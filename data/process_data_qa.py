import os
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from tqdm import tqdm
import logging
from transformers import CLIPProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MediqaQADataset(Dataset):
    def __init__(self, data_dir, query_file, closed_qa_file, mode='train', transform=None):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, 'images')
        self.mode = mode
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.269, 0.271, 0.282]),
        ])
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        with open(query_file, 'r') as f:
            self.queries = json.load(f)
        with open(closed_qa_file, 'r') as f:
            self.closed_qa_dict = json.load(f)
        
        self.image_files = []
        self.qa_data = []
        self.skipped_samples = []
        
        self.qid_groups = {}
        for qa in self.closed_qa_dict:
            qid = qa['qid']
            parent_qid = qid.split('-')[0]
            if parent_qid not in self.qid_groups:
                self.qid_groups[parent_qid] = []
            self.qid_groups[parent_qid].append(qa)
        
        with tqdm(total=len(self.queries), desc="Processing queries", unit="query") as pbar:
            for query in self.queries:
                encounter_id = query.get('encounter_id', '')
                if not encounter_id:
                    logger.warning(f"Missing encounter_id in query: {query}")
                    self.skipped_samples.append((encounter_id, "Missing encounter_id"))
                    pbar.update(1)
                    continue
                
                query_content = (query.get('query_title_en', '') + " " + query.get('query_content_en', '')).lower().strip()
                if not query_content:
                    query_content = "skin issue"
                
                keywords = query_content
                
                image_ids = []
                if mode == 'train':
                    for img_file in os.listdir(self.image_dir):
                        if img_file.startswith(f'IMG_{encounter_id}_') and (img_file.endswith('.png') or img_file.endswith('.jpg')):
                            img_id = img_file.replace(f'IMG_{encounter_id}_', '').rsplit('.', 1)[0]
                            image_ids.append(img_id)
                            break  # Chỉ lấy ảnh đầu tiên
                else:
                    image_ids = query.get('image_ids', query.get('image_id', []))
                    if isinstance(image_ids, str):
                        image_ids = [image_ids]
                    image_ids = [
                        img_id.replace(f'IMG_{encounter_id}_', '').rsplit('.', 1)[0]
                        if img_id.startswith(f'IMG_{encounter_id}_') else img_id
                        for img_id in image_ids if img_id
                    ]
                    if image_ids:
                        image_ids = [image_ids[0]]  # Chỉ lấy ảnh đầu tiên
                
                if not image_ids:
                    logger.warning(f"No valid image_ids for encounter_id: {encounter_id}")
                    self.skipped_samples.append((encounter_id, "No valid image_ids"))
                    pbar.update(1)
                    continue
                
                for img_id in image_ids:
                    img_path = os.path.join(self.image_dir, f'IMG_{encounter_id}_{img_id}.png')
                    if not os.path.exists(img_path):
                        img_path = os.path.join(self.image_dir, f'IMG_{encounter_id}_{img_id}.jpg')
                    
                    try:
                        Image.open(img_path).convert('RGB')
                    except Exception as e:
                        logger.warning(f"Failed to load image {img_path}: {e}")
                        self.skipped_samples.append((encounter_id, f"Failed to load image_id: {img_id}, error: {e}"))
                        continue
                    
                    self.image_files.append(img_path)
                    
                    for qa in self.closed_qa_dict:
                        qid = qa['qid']
                        question_text = qa['question_en']
                        options = [str(opt) for opt in qa['options_en'] if opt]
                        parent_qid = qid.split('-')[0]
                        question_index = self.qid_groups[parent_qid].index(qa) + 1
                        
                        self.qa_data.append({
                            'encounter_id': encounter_id,
                            'image_id': img_id,
                            'query': query_content,
                            'keywords': keywords,
                            'qid': qid,
                            'question_index': question_index,
                            'options': options,
                            'question_text': question_text
                        })
                
                pbar.update(1)
        
        logger.info(f"Total queries processed: {len(self.queries)}")
        logger.info(f"Total images in dataset: {len(self.image_files)}")
        logger.info(f"Total QA entries: {len(self.qa_data)}")
        if self.skipped_samples:
            logger.warning(f"Skipped samples: {len(self.skipped_samples)}")
            for enc_id, reason in self.skipped_samples:
                logger.warning(f"Skipped encounter_id: {enc_id}, reason: {reason}")
        if not self.qa_data:
            logger.error("No valid QA samples generated. Check data consistency.")

    def __len__(self):
        return len(self.qa_data)

    def __getitem__(self, idx):
        qa_info = self.qa_data[idx]
        img_path = os.path.join(self.image_dir, f'IMG_{qa_info["encounter_id"]}_{qa_info["image_id"]}.png')
        if not os.path.exists(img_path):
            img_path = os.path.join(self.image_dir, f'IMG_{qa_info["encounter_id"]}_{qa_info["image_id"]}.jpg')
        
        try:
            image = Image.open(img_path).convert('RGB')
            transformed_image = self.transform(image)
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            transformed_image = torch.zeros(3, 224, 224)
        
        # Thêm số thứ tự câu hỏi vào prompt để phân biệt CQID011-001, CQID011-002, ...
        question_number = qa_info['qid'].split('-')[1] if '-' in qa_info['qid'] else "1"
        prompt = (
            f"Context: {qa_info['query']}\n"
            f"Keywords: {qa_info['keywords']}\n"
            f"Question {qa_info['question_index']} (Number {question_number}): {qa_info['question_text']}\n"
            f"Options: {', '.join([f'{i+1}. {opt}' for i, opt in enumerate(qa_info['options'])])}"
        )
        
        return {
            'image': transformed_image,
            'prompt': prompt,
            'qid': qa_info['qid'],
            'question_index': qa_info['question_index'],
            'options': qa_info['options'],
            'encounter_id': qa_info['encounter_id'],
            'image_id': qa_info['image_id']
        }