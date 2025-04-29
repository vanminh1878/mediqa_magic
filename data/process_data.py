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

def build_question_keywords(closed_qa_dict):
    """
    Xây dựng từ khóa cho từng qid dựa trên question_en và options_en trong closed_qa_dict.
    """
    question_keywords = {}
    
    for qa in closed_qa_dict:
        qid = qa["qid"]
        question_text = qa["question_en"].lower()
        options = qa["options_en"]
        
        # Tạo danh sách từ khóa từ câu hỏi và các lựa chọn
        keywords = []
        
        # Từ khóa từ question_en (tách các từ quan trọng)
        question_words = question_text.split()
        keywords.extend([word for word in question_words if len(word) > 3])  # Lọc từ ngắn
        
        # Từ khóa từ options_en
        for option in options:
            option_words = option.lower().split()
            keywords.extend([word for word in option_words if len(word) > 3])
        
        # Thêm từ khóa đặc trưng theo loại câu hỏi
        if "site" in question_text or "area" in question_text:
            keywords.extend(["area", "spot", "region", "circle", "limited", "widespread"])
        if "where" in question_text or "location" in question_text:
            keywords.extend(["thigh", "palm", "leg", "arm", "head", "neck", "chest", "back", "hand", "foot"])
        if "when" in question_text or "onset" in question_text or "time" in question_text:
            keywords.extend(["hour", "day", "week", "month", "year", "since", "ago", "time"])
        
        # Loại bỏ trùng lặp
        keywords = list(set(keywords))
        
        question_keywords[qid] = keywords
    
    return question_keywords

def infer_qid(query_content, closed_qa_dict, question_keywords):
    """
    Suy ra danh sách qid, question_text, và options từ query_content dựa trên từ khóa.
    """
    query_content = query_content.lower()
    
    # Tìm các qid khớp
    matched_qids = []
    for qid, keywords in question_keywords.items():
        if any(keyword in query_content for keyword in keywords):
            for qa in closed_qa_dict:
                if qa["qid"] == qid:
                    matched_qids.append((qid, qa["question_en"], qa["options_en"]))
    
    return matched_qids if matched_qids else [(None, None, None)]

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

        with open(query_file, 'r') as f:
            self.queries = json.load(f)
        
        with open(closed_qa_file, 'r') as f:
            self.closed_qa_dict = json.load(f)
        
        # Xây dựng từ khóa từ closed_qa_dict
        self.question_keywords = build_question_keywords(self.closed_qa_dict)
        
        self.image_files = []
        self.masks = []
        self.qa_data = []
        
        # Xử lý queries
        for query in tqdm(self.queries, desc="Processing queries"):
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
            
            # Suy ra qid từ query_content
            matched_qids = infer_qid(query_content, self.closed_qa_dict, self.question_keywords)
            
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
                    print(f"Failed to load image {img_path}: {e}")
                    continue
                
                if mode == 'train':
                    if os.path.exists(img_path) and mask_path:
                        self.image_files.append(img_path)
                        self.masks.append(mask_path)
                else:
                    if os.path.exists(img_path):
                        self.image_files.append(img_path)
                
                # Thêm dữ liệu QA cho mỗi qid khớp
                for qid, question_text, options in matched_qids:
                    if qid is None:
                        print(f"No qid inferred for encounter_id: {encounter_id}, image_id: {img_id}")
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
        
        print(f"Total images in dataset: {len(self.image_files)}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None
        
        qa_info = self.qa_data[idx]
        query = qa_info['query']
        qid = qa_info['qid']
        options = qa_info['options'] or []
        question_text = qa_info['question_text'] or ""
        
        # Áp dụng transform cho hình ảnh
        transformed_image = self.transform(image)
        
        try:
            inputs = self.processor(images=image, text=f"Question: {question_text}\nContext: {query}\nOptions: {', '.join(options)}", return_tensors="pt")
        except Exception as e:
            print(f"Error processing image {img_path} with Blip2Processor: {e}")
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
                print(f"Warning: Failed to load mask {self.masks[idx]}")
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