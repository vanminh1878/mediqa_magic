import os
import json
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import Blip2Processor
import torch
from tqdm import tqdm
import logging
from sentence_transformers import SentenceTransformer, util

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def infer_qid(query_content, closed_qa_dict, model, threshold=0.5):
    """
    Suy ra danh sách qid, question_text, và options từ query_content dựa trên độ tương đồng ngữ nghĩa.
    Args:
        query_content: Nội dung câu hỏi từ test.json.
        closed_qa_dict: Danh sách câu hỏi từ closedquestions_definitions_imageclef2025.json.
        model: Mô hình Sentence-BERT.
        threshold: Ngưỡng độ tương đồng để chọn câu hỏi.
    """
    query_content = query_content.lower().strip()
    if not query_content:
        return [(None, None, None)]
    
    # Mã hóa query_content, tắt thanh tiến trình
    query_embedding = model.encode(query_content, convert_to_tensor=True, show_progress_bar=False)
    
    # Tìm các qid khớp
    matched_qids = []
    for qa in closed_qa_dict:
        qid = qa["qid"]
        question_text = qa["question_en"].lower()
        options = qa["options_en"]
        
        # Kết hợp question_text và options, mã hóa với show_progress_bar=False
        combined_text = question_text + " " + " ".join(options).lower()
        question_embedding = model.encode(combined_text, convert_to_tensor=True, show_progress_bar=False)
        
        # Tính độ tương đồng cosine
        similarity = util.cos_sim(query_embedding, question_embedding).item()
        
        # Nếu độ tương đồng vượt ngưỡng, thêm vào danh sách
        if similarity > threshold:
            matched_qids.append((qid, qa["question_en"], options))
    
    if not matched_qids:
        logger.info(f"No qid inferred for query_content: {query_content}")
        return [(None, None, None)]
    
    # Sắp xếp theo độ tương đồng
    matched_qids.sort(key=lambda x: util.cos_sim(
        model.encode(query_content, convert_to_tensor=True, show_progress_bar=False),
        model.encode(x[1] + " " + " ".join(x[2]).lower(), convert_to_tensor=True, show_progress_bar=False)
    ).item(), reverse=True)
    
    return matched_qids

class MediqaDataset(Dataset):
    def __init__(self, data_dir, query_file, closed_qa_file, mode='train', transform=None):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, 'images')
        self.mask_dir = os.path.join(data_dir, 'masks', mode)
        self.mode = mode
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")

        # Tải mô hình Sentence-BERT
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

        with open(query_file, 'r') as f:
            self.queries = json.load(f)
        
        with open(closed_qa_file, 'r') as f:
            self.closed_qa_dict = json.load(f)
        
        self.image_files = []
        self.masks = []
        self.qa_data = []
        
        # Thanh tiến trình duy nhất
        with tqdm(total=len(self.queries), desc="Processing queries", unit="query") as pbar:
            for query in self.queries:
                encounter_id = query['encounter_id']
                query_content = query.get('query_content_en', '')
                
                # Lấy danh sách image_ids
                image_ids = []
                if mode == 'train':
                    for img_file in os.listdir(self.image_dir):
                        if img_file.startswith(f'IMG_{encounter_id}_') and (img_file.endswith('.png') or img_file.endswith('.jpg')):
                            img_id = img_file.replace(f'IMG_{encounter_id}_', '').rsplit('.', 1)[0]
                            image_ids.append(img_id)
                else:
                    image_ids = query.get('image_ids', query.get('image_id', []))
                    if isinstance(image_ids, str):
                        image_ids = [image_ids]
                    image_ids = [img_id.replace(f'IMG_{encounter_id}_', '').rsplit('.', 1)[0] if img_id.startswith(f'IMG_{encounter_id}_') else img_id for img_id in image_ids]
                
                # Suy ra qid dùng Sentence-BERT
                matched_qids = infer_qid(query_content, self.closed_qa_dict, self.sentence_model)
                
                for img_id in image_ids:
                    img_path = os.path.join(self.image_dir, f'IMG_{encounter_id}_{img_id}.png')
                    if not os.path.exists(img_path):
                        img_path = os.path.join(self.image_dir, f'IMG_{encounter_id}_{img_id}.jpg')
                    
                    # Kiểm tra file hình ảnh và mặt nạ
                    try:
                        Image.open(img_path).convert('RGB')
                        mask_path = None
                        if mode == 'train':
                            for suffix in ['_mask_ann0.tiff', '_mask_ann1.tiff', '_mask_ann2.tiff', '_mask_ann3.tiff']:
                                temp_path = os.path.join(self.mask_dir, f'IMG_{encounter_id}_{img_id}{suffix}')
                                if os.path.exists(temp_path):
                                    mask_path = temp_path
                                    break
                            if mask_path:
                                mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                                if mask_img is None:
                                    continue
                            else:
                                continue
                    except Exception as e:
                        logger.info(f"Failed to load image {img_path}: {e}")
                        continue
                    
                    if mode == 'train':
                        if os.path.exists(img_path) and mask_path:
                            self.image_files.append(img_path)
                            self.masks.append(mask_path)
                    else:
                        if os.path.exists(img_path):
                            self.image_files.append(img_path)
                    
                    # Thêm dữ liệu QA cho mỗi qid
                    for qid, question_text, options in matched_qids:
                        if qid is None:
                            logger.info(f"No qid inferred for encounter_id: {encounter_id}, image_id: {img_id}")
                            qid = ''
                            question_text = ''
                            options = []
                        
                        self.qa_data.append({
                            'encounter_id': encounter_id,
                            'image_id': img_id,
                            'query': query_content,
                            'qid': qid,
                            'options': options,
                            'question_text': question_text
                        })
                
                pbar.update(1)
        
        logger.info(f"Total images in dataset: {len(self.image_files)}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.info(f"Error loading image {img_path}: {e}")
            return None
        
        qa_info = self.qa_data[idx]
        query = qa_info['query']
        qid = qa_info['qid']
        options = qa_info['options'] or []
        question_text = qa_info['question_text'] or ""
        
        transformed_image = self.transform(image)
        
        try:
            inputs = self.processor(images=image, text=f"Question: {question_text}\nContext: {query}\nOptions: {', '.join(options)}", return_tensors="pt")
        except Exception as e:
            logger.info(f"Error processing image {img_path} with Blip2Processor: {e}")
            return None
        
        mask = None
        if self.mode == 'train':
            mask_img = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
            if mask_img is not None:
                mask = mask_img / 255.0
                mask = Image.fromarray(mask)
                if self.transform:
                    mask = self.transform(mask)
                    mask = mask.squeeze()
                    mask = mask.unsqueeze(0)
            else:
                logger.info(f"Warning: Failed to load mask {self.masks[idx]}")
                mask = torch.zeros((1, 256, 256))
        
        return {
            'image': transformed_image,
            'prompt': f"Question: {question_text}\nContext: {query}\nOptions: {', '.join(options)}",
            'qid': qid,
            'options': options,
            'mask': mask if mask is not None else torch.zeros((1, 256, 256)),
            'encounter_id': qa_info['encounter_id'],
            'image_id': qa_info['image_id']
        }