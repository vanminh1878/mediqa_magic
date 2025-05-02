import os
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MediqaQADataset(Dataset):
    def __init__(self, data_dir, query_file, closed_qa_file, mode='test', transform=None):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, 'images')
        self.mode = mode
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Chuẩn hóa cho BLIP
        ])
        
        # Load queries and closed QA definitions
        try:
            with open(query_file, 'r') as f:
                self.queries = json.load(f)
            logger.info(f"Loaded {len(self.queries)} queries from {query_file}")
        except Exception as e:
            logger.error(f"Error loading query file {query_file}: {e}")
            raise
        
        try:
            with open(closed_qa_file, 'r') as f:
                self.closed_qa_dict = json.load(f)
            logger.info(f"Loaded {len(self.closed_qa_dict)} closed QA definitions from {closed_qa_file}")
        except Exception as e:
            logger.error(f"Error loading closed QA file {closed_qa_file}: {e}")
            raise
        
        # Create qid to options mapping
        self.qid_to_options = {qa['qid']: [str(opt) for opt in qa['options_en'] if opt] for qa in self.closed_qa_dict}
        
        # Group questions by parent qid (e.g., CQID011)
        self.qid_groups = {}
        for qa in self.closed_qa_dict:
            qid = qa['qid']
            parent_qid = qid.split('-')[0]
            if parent_qid not in self.qid_groups:
                self.qid_groups[parent_qid] = []
            self.qid_groups[parent_qid].append(qa)
        
        self.image_files = []
        self.qa_data = []
        self.skipped_samples = []
        
        # Process queries
        with tqdm(total=len(self.queries), desc=f"Processing queries ({mode})", unit="query") as pbar:
            for query in self.queries:
                encounter_id = query.get('encounter_id', '')
                if not encounter_id:
                    logger.warning(f"Missing encounter_id in query: {query}")
                    self.skipped_samples.append((encounter_id, "Missing encounter_id"))
                    pbar.update(1)
                    continue
                
                # Combine title and content
                query_content = (query.get('query_title_en', '') + " " + query.get('query_content_en', '')).lower().strip()
                if not query_content:
                    query_content = "skin issue"
                
                # Get image IDs
                image_ids = query.get('image_ids', [])
                if isinstance(image_ids, str):
                    image_ids = [image_ids]
                image_ids = [
                    img_id.replace(f'IMG_{encounter_id}_', '').rsplit('.', 1)[0]
                    if img_id.startswith(f'IMG_{encounter_id}_') else img_id
                    for img_id in image_ids if img_id
                ]
                if not image_ids:
                    logger.warning(f"No valid image_ids for encounter_id: {encounter_id}")
                    self.skipped_samples.append((encounter_id, "No valid image_ids"))
                    pbar.update(1)
                    continue
                
                # Select only the first image
                image_ids = [image_ids[0]]
                
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
                    
                    # Prepare QA data based on mode
                    if mode == 'valid':
                        questions = query.get('questions', {})
                        if not questions:
                            logger.warning(f"No questions found for encounter_id: {encounter_id}")
                            self.skipped_samples.append((encounter_id, "No questions found"))
                            continue
                        for qid in questions.keys():
                            if qid not in self.qid_to_options:
                                logger.warning(f"Skipping qid={qid} not found in closed_qa_dict for encounter_id: {encounter_id}")
                                self.skipped_samples.append((encounter_id, f"Invalid qid: {qid}"))
                                continue
                            parent_qid = qid.split('-')[0]
                            qa = next((qa for qa in self.closed_qa_dict if qa['qid'] == qid), None)
                            if not qa:
                                continue
                            question_text = qa['question_en']
                            options = self.qid_to_options[qid]
                            question_index = self.qid_groups[parent_qid].index(qa) + 1
                            
                            self.qa_data.append({
                                'encounter_id': encounter_id,
                                'image_id': img_id,
                                'query': query_content,
                                'qid': qid,
                                'question_index': question_index,
                                'options': options,
                                'question_text': question_text
                            })
                    else:  # mode='test'
                        qids = query.get('qids', [])
                        if not qids:
                            # Fallback: Use all qids from closed_qa_dict
                            qids = [qa['qid'] for qa in self.closed_qa_dict]
                            logger.info(f"No qids specified for encounter_id: {encounter_id}, using all {len(qids)} qids")
                        for qid in qids:
                            if qid not in self.qid_to_options:
                                logger.warning(f"Skipping qid={qid} not found in closed_qa_dict for encounter_id: {encounter_id}")
                                self.skipped_samples.append((encounter_id, f"Invalid qid: {qid}"))
                                continue
                            parent_qid = qid.split('-')[0]
                            qa = next((qa for qa in self.closed_qa_dict if qa['qid'] == qid), None)
                            if not qa:
                                continue
                            question_text = qa['question_en']
                            options = self.qid_to_options[qid]
                            question_index = self.qid_groups[parent_qid].index(qa) + 1
                            
                            self.qa_data.append({
                                'encounter_id': encounter_id,
                                'image_id': img_id,
                                'query': query_content,
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
            # Đặt đường dẫn ghi tệp vào /kaggle/working/
            output_dir = "/kaggle/working/"
            skipped_samples_file = os.path.join(output_dir, f'skipped_samples_init_{mode}.txt')
            
            # Đảm bảo thư mục tồn tại
            os.makedirs(output_dir, exist_ok=True)
            
            # Ghi đè tệp (tạo mới nếu chưa có)
            with open(skipped_samples_file, 'w') as f:
                for enc_id, reason in self.skipped_samples:
                    f.write(f"Skipped encounter_id={enc_id}, reason={reason}\n")
            logger.info(f"Skipped samples saved to {skipped_samples_file}")
        if not self.qa_data:
            logger.error("No valid QA samples generated. Check data consistency.")
            raise ValueError("No valid QA samples generated")

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
        
        # Prepare prompt for VQA
        question_number = qa_info['qid'].split('-')[1] if '-' in qa_info['qid'] else "1"
        prompt = (
            f"Question {qa_info['question_index']} (Number {question_number}): {qa_info['question_text']}\n"
            f"Options: {', '.join([f'{i+1}. {opt}' for i, opt in enumerate(qa_info['options'])])}"
        )
        
        # Log sample info
        logger.debug(f"Sample idx={idx}: encounter_id={qa_info['encounter_id']}, image_id={qa_info['image_id']}, "
                     f"qid={qa_info['qid']}, query={qa_info['query']}, options={qa_info['options']}")
        
        return {
            'image': transformed_image,
            'query': qa_info['query'],
            'prompt': prompt,
            'qid': qa_info['qid'],
            'question_index': qa_info['question_index'],
            'options': qa_info['options'],
            'encounter_id': qa_info['encounter_id'],
            'image_id': qa_info['image_id'],
            'question_text': qa_info['question_text']
        }