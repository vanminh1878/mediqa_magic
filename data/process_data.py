import os
import json
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from transformers import Blip2Processor

class MediqaDataset(Dataset):
    def __init__(self, data_dir, query_file, closed_qa_file, mode='train', transform=None):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, 'images')
        self.mask_dir = os.path.join(data_dir, 'masks', mode)  # Tìm trong masks/train hoặc masks/valid
        self.mode = mode
        self.transform = transform
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        
        with open(query_file, 'r') as f:
            self.queries = json.load(f)
        
        with open(closed_qa_file, 'r') as f:
            self.closed_qa_dict = json.load(f)
        
        self.image_files = []
        self.masks = []
        self.qa_data = []
        
        print(f"Processing {len(self.queries)} queries from {query_file}")
        for query in self.queries:
            encounter_id = query['encounter_id']
            print(f"Encounter ID: {encounter_id}")
            
            # Lấy danh sách image_ids từ thư mục images/
            image_ids = []
            if mode == 'train':
                for img_file in os.listdir(self.image_dir):
                    if img_file.startswith(f'IMG_{encounter_id}_') and (img_file.endswith('.png') or img_file.endswith('.jpg')):
                        img_id = img_file.replace(f'IMG_{encounter_id}_', '').rsplit('.', 1)[0]
                        image_ids.append(img_id)
                print(f"Found {len(image_ids)} images for {encounter_id}: {image_ids}")
            else:
                image_ids = query.get('image_ids', query.get('image_id', []))
                if isinstance(image_ids, str):
                    image_ids = [image_ids]
            
            for img_id in image_ids:
                img_path = os.path.join(self.image_dir, f'IMG_{encounter_id}_{img_id}.png')
                if not os.path.exists(img_path):
                    img_path = os.path.join(self.image_dir, f'IMG_{encounter_id}_{img_id}.jpg')
                mask_path = os.path.join(self.mask_dir, f'IMG_{encounter_id}_{img_id}_mask_0.tiff')
                
                if mode == 'train':
                    if os.path.exists(img_path) and os.path.exists(mask_path):
                        self.image_files.append(img_path)
                        self.masks.append(mask_path)
                        print(f"Added image: {img_path}, mask: {mask_path}")
                    else:
                        print(f"Skipped: Image {img_path} or mask {mask_path} not found")
                else:
                    if os.path.exists(img_path):
                        self.image_files.append(img_path)
                        print(f"Added image: {img_path}")
                
                # Suy ra qid từ query hoặc closed_qa_dict
                qid = None
                options = None
                question_text = None
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
                    'query': '',  # train_cvqa.json không có query_content_en
                    'qid': qid,
                    'options': options,
                    'question_text': question_text
                })
        
        print(f"Total images in dataset: {len(self.image_files)}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        qa_info = self.qa_data[idx]
        query = qa_info['query']
        qid = qa_info['qid']
        options = qa_info['options']
        question_text = qa_info['question_text']
        
        prompt = f"Question: {question_text}\nContext: {query}\nOptions: {', '.join(options if options else [])}"
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        
        mask = None
        if self.mode == 'train':
            mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
            mask = mask / 255.0
            if self.transform:
                image = self.transform(image)
                mask = self.transform(mask)
        
        return {
            'image': inputs['pixel_values'].squeeze(),
            'prompt': prompt,
            'qid': qid,
            'options': options,
            'mask': mask,
            'encounter_id': qa_info['encounter_id'],
            'image_id': qa_info['image_id']
        }