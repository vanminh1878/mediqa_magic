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
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

logging.getLogger("transformers").setLevel(logging.ERROR)

class MediQAQADataset(Dataset):
    def __init__(self, query_file, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_dir = os.path.join(data_dir, 'images')
        self.mask_dir = os.path.join(data_dir, f'masks_{split}')
        
        with open(query_file, 'r') as f:
            self.data = json.load(f)
        
        # Load closed questions definitions
        with open('closedquestions_definitions_imageclef2025.json', 'r') as f:
            self.q_definitions = json.load(f)
        
        self.qid_to_options = {q['qid']: q['options_en'] for q in self.q_definitions}
    
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
        
        # Ground truth closed QA labels (use provided labels or improve regex/NLP-based labeling)
        closed_qa = item.get('closed_qa', {})
        if not closed_qa:
            closed_qa = self.extract_closed_qa(item['query_content_en'].lower())
        
        return {
            'image': image,
            'query': query,
            'closed_qa': closed_qa,
            'encounter_id': encounter_id
        }
    
    def extract_closed_qa(self, query_content):
        # Improved label extraction (can be replaced with BERT-based NER or classification)
        closed_qa = {}
        
        # CQID010-001: How much of the body is affected
        closed_qa['CQID010-001'] = (
            0 if 'single spot' in query_content or 'single' in query_content else
            1 if 'limited area' in query_content or 'small area' in query_content else
            2 if 'widespread' in query_content or 'whole body' in query_content else
            3
        )
        
        # CQID011-001 to CQID011-006: Where is the affected area
        locations = ['head', 'neck', 'upper extremities|arm|hand|elbow|wrist', 
                     'lower extremities|thigh|leg|knee|ankle', 'chest|abdomen|torso', 
                     'back', 'palm|thumb|finger']
        found_locations = []
        for i, qid in enumerate(['CQID011-001', 'CQID011-002', 'CQID011-003', 
                                'CQID011-004', 'CQID011-005', 'CQID011-006']):
            if i < len(found_locations):
                closed_qa[qid] = found_locations[i]
            else:
                for j, loc in enumerate(locations):
                    if any(word in query_content for word in loc.split('|')):
                        closed_qa[qid] = j
                        found_locations.append(j)
                        break
                else:
                    closed_qa[qid] = 7
        
        # CQID012-001 to CQID012-006: Size of affected areas
        for i, qid in enumerate(['CQID012-001', 'CQID012-002', 'CQID012-003', 
                                'CQID012-004', 'CQID012-005', 'CQID012-006']):
            if i == 0:
                closed_qa[qid] = (
                    0 if 'thumb nail' in query_content or 'nail' in query_content else
                    1 if 'palm' in query_content else
                    2 if 'large' in query_content or 'cm' in query_content else
                    3
                )
            else:
                closed_qa[qid] = 3
        
        # CQID015-001: Onset
        times = ['hour', 'day', 'week', 'month', 'year']
        for i, time in enumerate(times):
            if time in query_content:
                closed_qa['CQID015-001'] = i
                break
        else:
            closed_qa['CQID015-001'] = 5 if 'years' in query_content else 6
        
        # CQID020-001 to CQID020-009: Skin description
        descs = ['raised|bumpy|blister', 'flat', 'sunken', 'thick', 'thin', 
                 'warty', 'crust', 'scab', 'weeping|oozing']
        for i, qid in enumerate(['CQID020-001', 'CQID020-002', 'CQID020-003', 
                                'CQID020-004', 'CQID020-005', 'CQID020-006', 
                                'CQID020-007', 'CQID020-008', 'CQID020-009']):
            if i == 0:
                for j, desc in enumerate(descs):
                    if any(word in query_content for word in desc.split('|')):
                        closed_qa[qid] = j
                        break
                else:
                    closed_qa[qid] = 9
            else:
                closed_qa[qid] = 9
        
        # CQID025-001: Itch
        closed_qa['CQID025-001'] = (
            0 if 'itch' in query_content or 'itchy' in query_content else
            1 if 'no itch' in query_content else
            2
        )
        
        # CQID034-001: Lesion color
        colors = ['normal', 'pink', 'red', 'brown', 'blue', 'purple', 'black', 
                  'white', 'combination', 'hyperpigmentation', 'hypopigmentation']
        for i, color in enumerate(colors):
            if color in query_content:
                closed_qa[qid] = i
                break
        else:
            closed_qa['CQID034-001'] = 11
        
        # CQID035-001: Lesion count
        closed_qa['CQID035-001'] = (
            0 if 'single' in query_content else
            1 if 'multiple' in query_content or 'many' in query_content else
            2
        )
        
        # CQID036-001: Texture
        closed_qa['CQID036-001'] = (
            0 if 'smooth' in query_content else
            1 if 'rough' in query_content else
            2
        )
        
        return closed_qa

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
        img_features = outputs.image_embeds  # [batch, 512]
        text_features = outputs.text_embeds  # [batch, 512]
        combined = self.dropout(torch.cat([img_features, text_features], dim=1))
        outputs = {qid: fc(combined) for qid, fc in self.fc_layers.items()}
        return outputs

def train_closed_qa():
    data_dir = "/kaggle/input/mediqa-data/mediqa-data/"
    query_file = "/kaggle/input/mediqa-data/mediqa-data/train_cvqa.json"
    synthetic_file = "/kaggle/working/synthetic_train.json"
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.269, 0.261, 0.276])
    ])
    
    dataset = MediQAQADataset(query_file, data_dir, split='train', transform=transform)
    synthetic_dataset = MediQAQADataset(synthetic_file, data_dir, split='train', transform=transform)
    dataloader = DataLoader(dataset + synthetic_dataset, batch_size=8, shuffle=True, num_workers=4)
    
    model = ClosedQAModel().cuda()
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch16')
    
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([2.0, 2.0, 2.0, 1.0]).cuda())  # Weight non-"Not mentioned" classes higher
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=20)
    
    for epoch in range(20):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=True, ncols=100):
            images = batch['image'].cuda()
            queries = batch['query']
            closed_qa = batch['closed_qa']
            
            with autocast():
                inputs = processor(text=queries, images=None, return_tensors='pt', 
                                 padding=True, truncation=True, max_length=77).to('cuda')
                inputs['pixel_values'] = images
                outputs = model(images, inputs)
                
                loss = 0
                for qid in model.qids:
                    if qid in closed_qa and closed_qa[qid] is not None:
                        labels = torch.tensor([closed_qa[qid][i] for i in range(len(closed_qa[qid]))]).cuda()
                        loss += criterion(outputs[qid], labels.long())
            
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        print(f"Epoch {epoch+1}, Average Loss: {total_loss / len(dataloader):.4f}")
    
    torch.save(model.state_dict(), "/kaggle/working/closed_qa_clip.pth")

if __name__ == "__main__":
    train_closed_qa()