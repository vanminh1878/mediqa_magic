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
            if 'single' in query_content or 'spot' in query_content:
                closed_qa['CQID010-001'] = 0
            elif 'limited' in query_content or 'area' in query_content:
                closed_qa['CQID010-001'] = 1
            elif 'widespread' in query_content:
                closed_qa['CQID010-001'] = 2
            else:
                closed_qa['CQID010-001'] = 3
            
            locations = ['head', 'neck', 'upper extremities', 'lower extremities', 'chest', 'back', 'other']
            for i, qid in enumerate(['CQID011-001', 'CQID011-002', 'CQID011-003', 'CQID011-004', 'CQID011-005', 'CQID011-006']):
                if i == 0:
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
                        closed_qa[qid] = 7
                else:
                    closed_qa[qid] = 7
            
            for qid in ['CQID012-001', 'CQID012-002', 'CQID012-003', 'CQID012-004', 'CQID012-005', 'CQID012-006']:
                if 'thumb' in query_content or 'nail' in query_content:
                    closed_qa[qid] = 0
                elif 'palm' in query_content:
                    closed_qa[qid] = 1
                elif 'large' in query_content or 'cm' in query_content:
                    closed_qa[qid] = 2
                else:
                    closed_qa[qid] = 3
            
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
                closed_qa['CQID015-001'] = 5
            
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
                    closed_qa[qid] = 9
            
            if 'itch' in query_content:
                closed_qa['CQID025-001'] = 0
            elif 'no itch' in query_content:
                closed_qa['CQID025-001'] = 1
            else:
                closed_qa['CQID025-001'] = 2
            
            colors = ['normal', 'pink', 'red', 'brown', 'blue', 'purple', 'black', 'white', 'combination', 'hyperpigmentation', 'hypopigmentation']
            for i, color in enumerate(colors):
                if color in query_content:
                    closed_qa['CQID034-001'] = i
                    break
            else:
                closed_qa['CQID034-001'] = 11
            
            if 'single' in query_content:
                closed_qa['CQID035-001'] = 0
            elif 'multiple' in query_content or 'many' in query_content:
                closed_qa['CQID035-001'] = 1
            else:
                closed_qa['CQID035-001'] = 2
            
            if 'smooth' in query_content:
                closed_qa['CQID036-001'] = 0
            elif 'rough' in query_content:
                closed_qa['CQID036-001'] = 1
            else:
                closed_qa['CQID036-001'] = 2
        
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

def inference_qa(split='valid'):
    data_dir = "/kaggle/input/mediqa-data/mediqa-data/"
    query_file = f"/kaggle/input/mediqa-data/mediqa-data/{split}.json"
    output_dir = f"/kaggle/working/{split}_results"
    os.makedirs(output_dir, exist_ok=True)
    
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = MediQAQADataset(query_file, data_dir, split=split, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base').cuda()
    model.load_state_dict(torch.load("/kaggle/working/blip_med.pth", weights_only=True))
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/blip-image-captioning-base')
    
    closed_model = ClosedQAModel().cuda()
    closed_model.load_state_dict(torch.load("/kaggle/working/closed_qa.pth", weights_only=True))
    
    results_open = []
    results_closed = []
    
    model.eval()
    closed_model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Inference {split}", leave=True, ncols=100):
            encounter_id = batch['encounter_id'][0]
            images = batch['image'].cuda()
            queries = batch['query']
            
            inputs = tokenizer(queries, return_tensors='pt', padding=True, truncation=True, max_length=512).to('cuda')
            input_ids_len = inputs['input_ids'].shape[1]
            print(f"Input IDs length for {encounter_id}: {input_ids_len}")
            if input_ids_len == 0:
                print(f"Skipping {encounter_id}: Empty input_ids")
                continue
            inputs['pixel_values'] = images
            outputs = model.generate(**inputs, max_new_tokens=200)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            tokenized_queries = tokenizer(queries, return_tensors='pt', padding=True, truncation=True, max_length=512).to('cuda')
            closed_outputs = closed_model(images, tokenized_queries)
            closed_pred = {}
            for qid in closed_model.qids:
                closed_pred[qid] = torch.argmax(closed_outputs[qid], dim=1).item()
            
            results_open.append({
                'encounter_id': encounter_id,
                'response': response
            })
            results_closed.append({
                'encounter_id': encounter_id,
                **closed_pred
            })
    
    with open(os.path.join(output_dir, 'open_qa.json'), 'w') as f:
        json.dump(results_open, f, indent=2)
    with open(os.path.join(output_dir, 'closed_qa.json'), 'w') as f:
        json.dump(results_closed, f, indent=2)
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    for split in ['valid', 'test']:
        inference_qa(split)