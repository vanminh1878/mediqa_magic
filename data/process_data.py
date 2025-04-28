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

class MediqaDataset(Dataset):
    def __init__(self, data_dir, query_file, closed_qa_file, mode='train', transform=None):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, 'images')
        self.mask_dir = os.path.join(data_dir, 'masks', mode)
        self.mode = mode
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),  # Chuẩn hóa kích thước
            transforms.ToTensor(),
        ])
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        
        with open(query_file, 'r') as f:
            self.queries = json.load(f)
        
        with open(closed_qa_file, 'r') as f:
            self.closed_qa_dict = json.load(f)
        
        self.image_files = []
        self.masks = []
        self.qa_data = []
        
        # Sử dụng tqdm để hiển thị thanh tiến trình khi xử lý queries
        for query in tqdm(self.queries, desc="Processing queries"):
            encounter_id = query['encounter_id']
            
            # Lấy danh sách image_ids từ thư mục images/
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
            
            for img_id in image_ids:
                img_path = os.path.join(self.image_dir, f'IMG_{encounter_id}_{img_id}.png')
                if not os.path.exists(img_path):
                    img_path = os.path.join(self.image_dir, f'IMG_{encounter_id}_{img_id}.jpg')
                
                # Thử nhiều hậu tố cho file mặt nạ
                mask_path = None
                for suffix in ['_mask_ann0.tiff', '_mask_ann1.tiff', '_mask_ann2.tiff', '_mask_ann3.tiff']:
                    temp_path = os.path.join(self.mask_dir, f'IMG_{encounter_id}_{img_id}{suffix}')
                    if os.path.exists(temp_path):
                        mask_path = temp_path
                        break
                
                # Kiểm tra file hình ảnh và mặt nạ trước khi thêm
                try:
                    Image.open(img_path).convert('RGB')
                    if mode == 'train' and mask_path:
                        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        if mask_img is None:
                            continue
                    elif mode == 'train' and not mask_path:
                        continue
                except Exception:
                    continue
                
                if mode == 'train':
                    if os.path.exists(img_path) and mask_path:
                        self.image_files.append(img_path)
                        self.masks.append(mask_path)
                else:
                    if os.path.exists(img_path):
                        self.image_files.append(img_path)
                
                # Suy ra qid từ query hoặc closed_qa_dict
                qid = None
                options = []
                question_text = ""
                for key in query:
                    if key.startswith('CQID'):
                        qid = key
                        for qa in self.closed_qa_dict:
                            if qa['qid'] == qid:
                                options = qa['options_en']
                                question_text = qa['question_en']
                                break
                        if options and question_text:
                            break
                
                self.qa_data.append({
                    'encounter_id': encounter_id,
                    'image_id': img_id,
                    'query': '',
                    'qid': qid if qid else '',
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
                    mask = mask.squeeze()  # Loại bỏ chiều kênh
                    mask = mask.unsqueeze(0)  # Thêm chiều kênh [1, 256, 256]
            else:
                print(f"Warning: Failed to load mask {self.masks[idx]}")
                mask = torch.zeros((1, 256, 256))  # Giá trị mặc định [1, 256, 256]
        
        return {
            'image': transformed_image,  # [3, 256, 256]
            'prompt': f"Question: {question_text}\nContext: {query}\nOptions: {', '.join(options)}",
            'qid': qid,
            'options': options,
            'mask': mask if mask is not None else torch.zeros((1, 256, 256)),
            'encounter_id': qa_info['encounter_id'],
            'image_id': qa_info['image_id']
        }