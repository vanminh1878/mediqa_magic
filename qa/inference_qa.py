import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BlipForConditionalGeneration, AutoTokenizer, AutoModel
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import timm
import logging

# Giảm logging chi tiết của Hugging Face
logging.getLogger("transformers").setLevel(logging.ERROR)

# Tích hợp MediQAQADataset trực tiếp
class MediQAQADataset(Dataset):
    def __init__(self, query_file, data_dir, split='valid', transform=None):
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

class ClosedQAModel(nn.Module):
    def __init__(self):
        super(ClosedQAModel, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768 + 1000, 9)  # 9 options for CQID011-001
    
    def forward(self, images, queries):
        img_features = self.vit(images)
        bert_outputs = self.bert(input_ids=queries['input_ids'], attention_mask=queries['attention_mask'])
        text_features = bert_outputs.pooler_output
        combined = torch.cat([img_features, text_features], dim=1)
        return self.fc(combined)

def inference_qa(split='valid'):
    data_dir = "/kaggle/input/mediqa-data/mediqa-data/"
    query_file = f"/kaggle/input/mediqa-data/mediqa-data/{split}.json"
    output_dir = f"/kaggle/working/{split}_results"
    os.makedirs(output_dir, exist_ok=True)
    
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = MediQAQADataset(query_file, data_dir, split=split, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # QA mở
    model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base').cuda()
    model.load_state_dict(torch.load("/kaggle/working/blip_med.pth"))
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/blip-image-captioning-base')
    
    # QA đóng
    closed_model = ClosedQAModel().cuda()
    closed_model.load_state_dict(torch.load("/kaggle/working/closed_qa.pth"))
    
    results_open = []
    results_closed = []
    
    model.eval()
    closed_model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Inference {split}", leave=True):
            encounter_id = batch['encounter_id'][0]
            images = batch['image'].cuda()
            queries = batch['query']
            
            # QA mở
            inputs = tokenizer(queries, return_tensors='pt', padding=True, truncation=True).to('cuda')
            inputs['pixel_values'] = images
            outputs = model.generate(**inputs, max_length=200)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # QA đóng
            tokenized_queries = tokenizer(queries, return_tensors='pt', padding=True, truncation=True).to('cuda')
            closed_outputs = closed_model(images, tokenized_queries)
            closed_pred = torch.argmax(closed_outputs, dim=1).item()
            
            results_open.append({
                'encounter_id': encounter_id,
                'response': response
            })
            results_closed.append({
                'encounter_id': encounter_id,
                'CQID011-001': closed_pred
            })
    
    with open(os.path.join(output_dir, 'open_qa.json'), 'w') as f:
        json.dump(results_open, f, indent=2)
    with open(os.path.join(output_dir, 'closed_qa.json'), 'w') as f:
        json.dump(results_closed, f, indent=2)
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    for split in ['valid', 'test']:
        inference_qa(split)