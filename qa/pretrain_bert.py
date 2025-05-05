import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from tqdm import tqdm
import os

class BERTQADataset(Dataset):
    def __init__(self, json_file, qids):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.qids = qids
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        query = f"{item['query_title_en']} {item['query_content_en']}"[:512]
        labels = {qid: item['closed_qa'][qid] for qid in self.qids if qid in item.get('closed_qa', {})}
        inputs = self.tokenizer(query, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels
        }

def train_bert_model(qid, num_labels, train_file, output_dir, epochs=2):
    os.makedirs(output_dir, exist_ok=True)
    dataset = BERTQADataset(train_file, [qid])
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1} for {qid}"):
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = torch.tensor([batch['labels'][qid][i] for i in range(len(batch['labels'][qid]))]).cuda()
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1} for {qid}, Average Loss: {total_loss / len(dataloader):.4f}")
    
    model.save_pretrained(os.path.join(output_dir, f"bert_{qid}"))
    dataset.tokenizer.save_pretrained(os.path.join(output_dir, f"bert_{qid}"))

if __name__ == "__main__":
    synthetic_file = "/kaggle/working/synthetic_train.json"
    output_dir = "/kaggle/working/bert_models"
    
    qid_configs = {
        'CQID010-001': 4,
        'CQID011-001': 8,
        'CQID011-002': 8,
        'CQID011-003': 8,
        'CQID011-004': 8,
        'CQID011-005': 8,
        'CQID011-006': 8,
        'CQID012-001': 4,
        'CQID012-002': 4,
        'CQID012-003': 4,
        'CQID012-004': 4,
        'CQID012-005': 4,
        'CQID012-006': 4,
        'CQID015-001': 7,
        'CQID020-001': 10,
        'CQID020-002': 10,
        'CQID020-003': 10,
        'CQID020-004': 10,
        'CQID020-005': 10,
        'CQID020-006': 10,
        'CQID020-007': 10,
        'CQID020-008': 10,
        'CQID020-009': 10,
        'CQID025-001': 3,
        'CQID034-001': 12,
        'CQID035-001': 3,
        'CQID036-001': 3
    }
    
    for qid, num_labels in qid_configs.items():
        train_bert_model(qid, num_labels, synthetic_file, output_dir)