import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel, DistilBertTokenizer, DistilBertForSequenceClassification
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import logging

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
logging.getLogger("transformers").setLevel(logging.ERROR)

class MediQAQADataset(Dataset):
    def __init__(self, query_file, data_dir, split='valid', transform=None, bert_model_dir="/kaggle/working/bert_models"):
        self.data_dir = data_dir
        self.transform = transform
        self.image_dir = os.path.join(data_dir, 'images')
        self.mask_dir = os.path.join(data_dir, f'masks_{split}')
        self.bert_model_dir = bert_model_dir
        
        with open(query_file, 'r') as f:
            self.data = json.load(f)
        
        with open('closedquestions_definitions_imageclef2025.json', 'r') as f:
            self.q_definitions = json.load(f)
        
        self.qids = [
            'CQID010-001', 'CQID011-001', 'CQID011-002', 'CQID011-003', 
            'CQID011-004', 'CQID011-005', 'CQID011-006', 'CQID012-001', 
            'CQID012-002', 'CQID012-003', 'CQID012-004', 'CQID012-005', 
            'CQID012-006', 'CQID015-001', 'CQID020-001', 'CQID020-002', 
            'CQID020-003', 'CQID020-004', 'CQID020-005', 'CQID020-006', 
            'CQID020-007', 'CQID020-008', 'CQID020-009', 'CQID025-001', 
            'CQID034-001', 'CQID035-001', 'CQID036-001'
        ]
        # Load BERT models and tokenizers on CPU
        self.bert_models = {}
        self.bert_tokenizers = {}
        for qid in self.qids:
            self.bert_models[qid] = DistilBertForSequenceClassification.from_pretrained(os.path.join(bert_model_dir, f"bert_{qid}"))
            self.bert_tokenizers[qid] = DistilBertTokenizer.from_pretrained(os.path.join(bert_model_dir, f"bert_{qid}"))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        encounter_id = item['encounter_id']
        query = f"{item['query_title_en']} {item['query_content_en']}"[:512]
        
        if not query.strip():
            query = "No query provided"
        
        # Load all images for the encounter
        images = []
        loaded_image_ids = []
        for img_id in item.get('image_ids', []):
            img_path = os.path.join(self.image_dir, img_id)
            try:
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                images.append(image)
                loaded_image_ids.append(img_id)
            except (FileNotFoundError, TypeError):
                print(f"Error: Image {img_path} not found for encounter {encounter_id}. Skipping this image.")
                continue
        
        if not images:
            print(f"Warning: No valid images loaded for encounter {encounter_id}. Using empty image tensor.")
            images = [torch.zeros(3, 224, 224)]  # Placeholder tensor
        
        # Stack images into a tensor
        images = torch.stack(images)
        
        # Extract closed QA labels using BERT
        closed_qa = self.extract_closed_qa(query)
        
        # Debug log
        print(f"Encounter {encounter_id}: Loaded images {loaded_image_ids}, Query: {query[:100]}...")
        
        return {
            'images': images,
            'query': query,
            'closed_qa': closed_qa,
            'encounter_id': encounter_id
        }
    
    def extract_closed_qa(self, query):
        closed_qa = {}
        for qid in self.qids:
            model = self.bert_models[qid]
            tokenizer = self.bert_tokenizers[qid]
            model.eval()
            with torch.no_grad():
                inputs = tokenizer(query, return_tensors='pt', padding=True, truncation=True, max_length=128)
                outputs = model(**inputs)
                closed_qa[qid] = torch.argmax(outputs.logits, dim=1).item()
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
        # Process multiple images and average their features
        batch_size, num_images, c, h, w = images.shape
        images = images.view(-1, c, h, w)  # Flatten for CLIP processing
        outputs = self.clip(pixel_values=images, input_ids=queries['input_ids'].repeat(num_images, 1), 
                           attention_mask=queries['attention_mask'].repeat(num_images, 1))
        img_features = outputs.image_embeds
        text_features = outputs.text_embeds
        
        # Average image features across all images for the encounter
        img_features = img_features.view(batch_size, num_images, -1).mean(dim=1)
        text_features = text_features.view(batch_size, num_images, -1).mean(dim=1)
        
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
            images = batch['images'].cuda()  # Shape: (1, num_images, 3, 224, 224)
            queries = batch['query']
            closed_qa_bert = batch['closed_qa']
            
            inputs = processor(text=queries, images=None, return_tensors='pt', 
                             padding=True, truncation=True, max_length=77).to('cuda')
            outputs = model(images, inputs)
            
            closed_pred = {}
            for qid in model.qids:
                pred = torch.argmax(outputs[qid], dim=1).item()
                # Combine CLIP prediction with BERT if BERT provides a non-"Not mentioned" label
                bert_label = closed_qa_bert[qid][0] if qid in closed_qa_bert else None
                if bert_label is not None and bert_label != (
                    3 if qid.startswith('CQID012') else
                    9 if qid.startswith('CQID020') else
                    7 if qid.startswith('CQID011') else
                    2
                ):
                    closed_pred[qid] = bert_label  # Prioritize BERT for non-"Not mentioned"
                else:
                    closed_pred[qid] = pred  # Use CLIP prediction
            
            results_closed.append({
                'encounter_id': encounter_id,
                **closed_pred
            })
    
    output_file = os.path.join(output_dir, 'closed_qa.json')
    with open(output_file, 'w') as f:
        json.dump(results_closed, f, indent=2)
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    for split in ['valid', 'test']:
        inference_closed_qa(split)