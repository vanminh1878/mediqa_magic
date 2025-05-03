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
from torch.cuda.amp import autocast
import re

logging.getLogger("transformers").setLevel(logging.ERROR)

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
        query = f"{item['query_title_en']} {item['query_content_en']}"[:200]
        
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
        
        response = item['responses'][0]['content_en'] if 'responses' in item else ""
        
        closed_qa = {}
        if 'query_content_en' in item:
            query_content = item['query_content_en'].lower()
            
            # CQID010-001: How much of the body is affected
            closed_qa['CQID010-001'] = (
                0 if re.search(r'\bsingle|spot\b', query_content) else
                1 if re.search(r'\blimited|small area|circle\b', query_content) else
                2 if re.search(r'\bwidespread|whole body\b', query_content) else
                3
            )
            
            # CQID011-001 to CQID011-006: Where is the affected area
            locations = [
                r'\bhead\b', r'\bneck\b', r'\barm|hand|upper|elbow|wrist\b',
                r'\bthigh|leg|lower|knee|ankle\b', r'\bchest|abdomen|torso\b',
                r'\bback\b', r'\bpalm|thumb|finger\b'
            ]
            found_locations = []
            for i, qid in enumerate(['CQID011-001', 'CQID011-002', 'CQID011-003', 'CQID011-004', 'CQID011-005', 'CQID011-006']):
                if i < len(found_locations):
                    closed_qa[qid] = found_locations[i]
                else:
                    for j, pattern in enumerate(locations):
                        if re.search(pattern, query_content):
                            closed_qa[qid] = j
                            found_locations.append(j)
                            break
                    else:
                        closed_qa[qid] = 7
            
            # CQID012-001 to CQID012-006: How large are the affected areas
            for i, qid in enumerate(['CQID012-001', 'CQID012-002', 'CQID012-003', 'CQID012-004', 'CQID012-005', 'CQID012-006']):
                if i == 0:
                    closed_qa[qid] = (
                        0 if re.search(r'\bthumb|nail|small\b', query_content) else
                        1 if re.search(r'\bpalm\b', query_content) else
                        2 if re.search(r'\blarge|\d+\s*cm\b', query_content) else
                        3
                    )
                else:
                    closed_qa[qid] = 3
            
            # CQID015-001: When did the patient first notice the issue
            times = [r'\bhour', r'\bday', r'\bweek', r'\bmonth', r'\byear\b']
            for i, pattern in enumerate(times):
                if re.search(pattern, query_content):
                    closed_qa['CQID015-001'] = i
                    break
            else:
                closed_qa['CQID015-001'] = 5 if re.search(r'\byears\b', query_content) else 6
            
            # CQID020-001 to CQID020-009: What label best describes the affected area
            descs = [
                r'\bbump|raised|lump|blister\b', r'\bflat\b', r'\bsunken\b',
                r'\bthick\b', r'\bthin\b', r'\bwart\b', r'\bcrust\b',
                r'\bscab\b', r'\bweep|oozing\b'
            ]
            for i, qid in enumerate(['CQID020-001', 'CQID020-002', 'CQID020-003', 'CQID020-004', 'CQID020-005', 'CQID020-006', 'CQID020-007', 'CQID020-008', 'CQID020-009']):
                if i == 0:
                    for j, pattern in enumerate(descs):
                        if re.search(pattern, query_content):
                            closed_qa[qid] = j
                            break
                    else:
                        closed_qa[qid] = 9
                else:
                    closed_qa[qid] = 9
            
            # CQID025-001: Is there any associated itching
            closed_qa['CQID025-001'] = (
                0 if re.search(r'\bitch|itchy\b', query_content) else
                1 if re.search(r'\bno itch\b', query_content) else
                2
            )
            
            # CQID034-001: Color of the skin lesion
            colors = ['normal', 'pink', 'red', 'brown', 'blue', 'purple', 'black', 'white', 'combination', 'hyperpigmentation', 'hypopigmentation']
            for i, color in enumerate(colors):
                if color in query_content:
                    closed_qa['CQID034-001'] = i
                    break
            else:
                closed_qa['CQID034-001'] = 11
            
            # CQID035-001: How many skin lesions
            closed_qa['CQID035-001'] = (
                0 if re.search(r'\bsingle\b', query_content) else
                1 if re.search(r'\bmultiple|many|several|circle\b', query_content) else
                2
            )
            
            # CQID036-001: Skin lesion texture
            closed_qa['CQID036-001'] = (
                0 if re.search(r'\bsmooth\b', query_content) else
                1 if re.search(r'\brough\b', query_content) else
                2
            )
        
        return {
            'image': image,
            'query': query,
            'response': response,
            'closed_qa': closed_qa,
            'encounter_id': encounter_id
        }

class ClosedQAModel(nn.Module):
    def __init__(self, qids=[
        'CQID010-001', 'CQID011-001', 'CQID011-002', 'CQID011-003', 'CQID011-004', 'CQID011-005', 'CQID011-006',
        'CQID012-001', 'CQID012-002', 'CQID012-003', 'CQID012-004', 'CQID012-005', 'CQID012-006', 'CQID015-001',
        'CQID020-001', 'CQID020-002', 'CQID020-003', 'CQID020-004', 'CQID020-005', 'CQID020-006', 'CQID020-007',
        'CQID020-008', 'CQID020-009', 'CQID025-001', 'CQID034-001', 'CQID035-001', 'CQID036-001'
    ]):
        super(ClosedQAModel, self).__init__()
        self.qids = qids
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.fc_layers = nn.ModuleDict({
            qid: nn.Linear(768 + 1000, 
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
        img_features = self.vit(images)
        bert_outputs = self.bert(input_ids=queries['input_ids'], attention_mask=queries['attention_mask'])
        text_features = bert_outputs.pooler_output
        combined = self.dropout(torch.cat([img_features, text_features], dim=1))
        outputs = {qid: fc(combined) for qid, fc in self.fc_layers.items()}
        return outputs

def train_qa():
    data_dir = "/kaggle/input/mediqa-data/mediqa-data/"
    query_file = "/kaggle/input/mediqa-data/mediqa-data/train_cvqa.json"
    synthetic_file = "/kaggle/working/synthetic_train.json"
    
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = MediQAQADataset(query_file, data_dir, split='train', transform=transform)
    synthetic_dataset = MediQAQADataset(synthetic_file, data_dir, split='train', transform=transform)
    dataloader = DataLoader(dataset + synthetic_dataset, batch_size=2, shuffle=True)
    
    model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base').cuda()
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/blip-image-captioning-base')
    
    closed_model = ClosedQAModel().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(closed_model.parameters()), lr=1e-4)
    
    for epoch in range(20):  # Tăng lên 20 epoch
        model.train()
        closed_model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=True, ncols=100):
            images = batch['image'].cuda()
            queries = batch['query']
            responses = batch['response']
            closed_qa = batch['closed_qa']
            
            with autocast():
                inputs = tokenizer(queries, return_tensors='pt', padding=True, truncation=True, max_length=512).to('cuda')
                inputs['pixel_values'] = images
                outputs = model(**inputs, labels=inputs['input_ids'])
                loss_open = outputs.loss if outputs.loss is not None else torch.tensor(0.0).cuda()
            
            tokenized_queries = tokenizer(queries, return_tensors='pt', padding=True, truncation=True, max_length=512).to('cuda')
            closed_outputs = closed_model(images, tokenized_queries)
            loss_closed = 0
            for qid in closed_model.qids:
                if qid in closed_qa and closed_qa[qid] is not None:
                    labels = torch.tensor([closed_qa[qid][i] for i in range(len(closed_qa[qid]))]).cuda()
                    loss_closed += criterion(closed_outputs[qid], labels.long())
            
            loss = loss_open + loss_closed
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}, Average Loss: {total_loss / len(dataloader):.4f}")
    
    torch.save(model.state_dict(), "/kaggle/working/blip_med.pth")
    torch.save(closed_model.state_dict(), "/kaggle/working/closed_qa.pth")

if __name__ == "__main__":
    train_qa()