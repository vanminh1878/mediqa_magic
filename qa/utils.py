import json
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class MediQAQADataset(Dataset):
    def __init__(self, query_file, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_dir = os.path.join(data_dir, 'images')
        self.mask_dir = os.path.join(data_dir, f'masks_{split}')
        
        with open(query_file, 'r') as f:
            self.data = json.load(f)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        encounter_id = item['encounter_id']
        query = f"{item['query_title_en']} {item['query_content_en']}"
        
        # Chọn ảnh đầu tiên
        img_id = item['image_ids'][0]
        img_path = os.path.join(self.image_dir, img_id)
        image = Image.open(img_path).convert('RGB')
        
        # Load mask nếu có
        mask_path = os.path.join(self.mask_dir, img_id.replace('.jpg', '_mask.png'))
        mask = Image.open(mask_path).convert('L') if os.path.exists(mask_path) else None
        
        if self.transform:
            image = self.transform(image)
        
        # Câu hỏi mở
        response = item['responses'][0]['content_en'] if 'responses' in item else ""
        
        # Câu hỏi đóng (giả sử ánh xạ từ query/response)
        closed_qa = {}
        if 'query_content_en' in item:
            if 'thigh' in item['query_content_en'].lower():
                closed_qa['CQID011-001'] = 3  # lower extremities
            elif 'palm' in item['query_content_en'].lower():
                closed_qa['CQID011-001'] = 7  # other (please specify)
            else:
                closed_qa['CQID011-001'] = 8  # Not mentioned
        
        return {
            'image': image,
            'query': query,
            'response': response,
            'closed_qa': closed_qa,
            'encounter_id': encounter_id
        }

def load_closed_questions(closed_qa_file):
    with open(closed_qa_file, 'r') as f:
        return json.load(f)