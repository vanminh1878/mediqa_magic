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
            if 'single' in query_content or 'spot' in query_content:
                closed_qa['CQID010-001'] = 0
            elif 'limited' in query_content or 'area' in query_content:
                closed_qa['CQID010-001'] = 1
            elif 'widespread' in query_content:
                closed_qa['CQID010-001'] = 2
            else:
                closed_qa['CQID010-001'] = 3  # Not mentioned
            
            # CQID011-001 to CQID011-006: Where is the affected area
            locations = ['head', 'neck', 'upper extremities', 'lower extremities', 'chest', 'back', 'other']
            for i, qid in enumerate(['CQID011-001', 'CQID011-002', 'CQID011-003', 'CQID011-004', 'CQID011-005', 'CQID011-006']):
                if i == 0:  # Chỉ xử lý vị trí đầu tiên
                    if 'head' in query_content:
                        closed_qa[qid] = 0
                    elif 'neck' in query_content:
                        closed_qa[qid] = 1
                    elif 'arm' in query_content or 'hand' in query_content:
                        closed_qa[qid] = 2
                    elif 'thigh' in query_content or 'leg' in query_content:
                        closed_qa[qid] = 3
                    elif 'chest' in query_content or 'abdomen' in query_content:
                        closed_qa[qid] = 4
                    elif 'back' in query_content:
                        closed_qa[qid] = 5
                    elif 'palm' in query_content or 'thumb' in query_content:
                        closed_qa[qid] = 6
                    else:
                        closed_qa[qid] = 7  # Not mentioned
                else:
                    closed_qa[qid] = 7  # Not mentioned for additional locations
            
            # CQID012-001 to CQID012-006: How large are the affected areas
            for qid in ['CQID012-001', 'CQID012-002', 'CQID012-003', 'CQID012-004', 'CQID012-005', 'CQID012-006']:
                if 'thumb' in query_content or 'nail' in query_content:
                    closed_qa[qid] = 0
                elif 'palm' in query_content:
                    closed_qa[qid] = 1
                elif 'large' in query_content or 'cm' in query_content:
                    closed_qa[qid] = 2
                else:
                    closed_qa[qid] = 3  # Not mentioned
            
            # CQID015-001: When did the patient first notice the issue
            if 'hour' in query_content:
                closed_qa['CQID015-001'] = 0
            elif 'day' in query_content:
                closed_qa['CQID015-001'] = 1
            elif 'week' in query_content:
                closed_qa['CQID015-001'] = 2
            elif 'month' in query_content:
                closed_qa['CQID015-001'] = 3
            elif 'year' in query_content:
                closed_qa['CQID015-001'] = 4
            else:
                closed_qa['CQID015-001'] = 5  # Multiple years or Not mentioned
            
            # CQID020-001 to CQID020-009: What label best describes the affected area
            for qid in ['CQID020-001', 'CQID020-002', 'CQID020-003', 'CQID020-004', 'CQID020-005', 'CQID020-006', 'CQID020-007', 'CQID020-008', 'CQID020-009']:
                if 'bump' in query_content or 'raised' in query_content:
                    closed_qa[qid] = 0
                elif 'flat' in query_content:
                    closed_qa[qid] = 1
                elif 'sunken' in query_content:
                    closed_qa[qid] = 2
                elif 'thick' in query_content:
                    closed_qa[qid] = 3
                elif 'thin' in query_content:
                    closed_qa[qid] = 4
                elif 'wart' in query_content:
                    closed_qa[qid] = 5
                elif 'crust' in query_content:
                    closed_qa[qid] = 6
                elif 'scab' in query_content:
                    closed_qa[qid] = 7
                elif 'weep' in query_content:
                    closed_qa[qid] = 8
                else:
                    closed_qa[qid] = 9  # Not mentioned
            
            # CQID025-001: Is there any associated itching
            if 'itch' in query_content:
                closed_qa['CQID025-001'] = 0
            elif 'no itch' in query_content:
                closed_qa['CQID025-001'] = 1
            else:
                closed_qa['CQID025-001'] = 2  # Not mentioned
            
            # CQID034-001: Color of the skin lesion
            colors = ['normal', 'pink', 'red', 'brown', 'blue', 'purple', 'black', 'white', 'combination', 'hyperpigmentation', 'hypopigmentation']
            for i, color in enumerate(colors):
                if color in query_content:
                    closed_qa['CQID034-001'] = i
                    break
            else:
                closed_qa['CQID034-001'] = 11  # Not mentioned
            
            # CQID035-001: How many skin lesions
            if 'single' in query_content:
                closed_qa['CQID035-001'] = 0
            elif 'multiple' in query_content or 'many' in query_content:
                closed_qa['CQID035-001'] = 1
            else:
                closed_qa['CQID035-001'] = 2  # Not mentioned
            
            # CQID036-001: Skin lesion texture
            if 'smooth' in query_content:
                closed_qa['CQID036-001'] = 0
            elif 'rough' in query_content:
                closed_qa['CQID036-001'] = 1
            else:
                closed_qa['CQID036-001'] = 2  # Not mentioned
        
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
        combined = torch.cat([img_features, text_features], dim=1)
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
    
    for epoch in range(3):
        model.train()
        closed_model.train()
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
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    
    torch.save(model.state_dict(), "/kaggle/working/blip_med.pth")
    torch.save(closed_model.state_dict(), "/kaggle/working/closed_qa.pth")

if __name__ == "__main__":
    train_qa()