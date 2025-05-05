import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

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
        query = f"{item['query_title_en']} {item['query_content_en']}"[:512]
        
        if not query.strip():
            query = "No query provided"
        
        img_id = item['image_ids'][0] if item['image_ids'] else None
        img_path = os.path.join(self.image_dir, img_id) if img_id else None
        try:
            image = Image.open(img_path).convert('RGB')
        except (FileNotFoundError, TypeError):
            print(f"Error: Image not found for encounter {encounter_id}. Using default image.")
            image = Image.new('RGB', (224, 224), color='gray')
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'query': query,
            'encounter_id': encounter_id
        }

class ClosedQAModel(nn.Module):
    def __init__(self, qids=[
        'CQID010-001', 'CQID011-001', 'CQID011-002', 'CQID011-003', 
        'CQID011-004', 'CQID011-005', 'CQID011-006', 'CQID012-001', 
        'CQID012-002', 'CQID012-003', 'CQID012-004', 'CQID012-005', 
        'CQID012-006', 'CQID015-001', 'CQID020-001', 'CQID020-002', 
        'CQID020-003', 'CQID020-004', 'CQID020-005', 'CQID020-006', 
        'CQID020-007', 'CQID020-008', 'CQID020-009', 'CQID025-001', 
        'CQID034-001', 'CQID035-001', 'CQID036-001'
    ]):
        super(ClosedQAModel, self).__init__()
        self.qids = qids
        self.clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch16')
        self.dropout = nn.Dropout(0.3)
        self.fc_layers = nn.ModuleDict({
            qid: nn.Linear(512 + 512, 
                           4 if qid.startswith('CQID010') else
                           8 if qid.startswith('CQID011') else
                           4 if qid.startswith('CQID012') else
                           7 if qid == 'CQID015-001' else
                           10 if qid.startswith('CQID020') else
                           3 if qid == 'CQID025-001' else
                           12 if qid == 'CQID034-001' else
                           3 if qid == 'CQID035-001' else
                           3) for qid in qids
        })
    
    def forward(self, images, queries):
        outputs = self.clip(pixel_values=images, input_ids=queries['input_ids'], 
                           attention_mask=queries['attention_mask'])
        img_features = outputs.image_embeds
        text_features = outputs.text_embeds
        combined = self.dropout(torch.cat([img_features, text_features], dim=1))
        outputs = {qid: fc(combined) for qid, fc in self.fc_layers.items()}
        return outputs

def inference_closed_qa(split='valid'):
    data_dir = "/kaggle/input/mediqa-data/mediqa-data/"
    query_file = f"/kaggle/input/mediqa-data/mediqa-data/{split}.json"
    output_dir = f"/kaggle/working/{split}_results"
    os.makedirs(output_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.269, 0.261, 0.276])
    ])
    
    dataset = MediQAQADataset(query_file, data_dir, split=split, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model = ClosedQAModel().cuda()
    model.load_state_dict(torch.load("/kaggle/working/closed_qa_clip.pth", weights_only=True))
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch16')
    
    results_closed = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Inference {split}", leave=True, ncols=100):
            encounter_id = batch['encounter_id'][0]
            images = batch['image'].cuda()
            queries = batch['query']
            
            inputs = processor(text=queries, images=None, return_tensors='pt', 
                             padding=True, truncation=True, max_length=77).to('cuda')
            inputs['pixel_values'] = images
            outputs = model(images, inputs)
            
            closed_pred = {}
            for qid in model.qids:
                closed_pred[qid] = torch.argmax(outputs[qid], dim=1).item()
            
            results_closed.append({
                'encounter_id': encounter_id,
                **closed_pred
            })
    
    with open(os.path.join(output_dir, 'closed_qa.json'), 'w') as f:
        json.dump(results_closed, f, indent=2)
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    for split in ['valid', 'test']:
        inference_closed_qa(split)