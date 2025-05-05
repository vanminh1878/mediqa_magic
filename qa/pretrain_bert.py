import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score

class FocalLoss(nn.Module):
    def __init__(self, alpha=7.0, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class BERTQADataset(Dataset):
    def __init__(self, json_file, qids, split='train'):
        self.split = split
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

def train_bert_model(qid, num_labels, train_file, valid_file, output_dir, epochs=15):
    os.makedirs(output_dir, exist_ok=True)
    train_dataset = BERTQADataset(train_file, [qid], split='train')
    valid_dataset = BERTQADataset(valid_file, [qid], split='valid')
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=False)
    
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = FocalLoss(alpha=7.0, gamma=2)
    
    best_valid_acc = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1} for {qid}"):
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = torch.tensor(batch['labels'][qid]).cuda()
            
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels.long())
            total_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1} for {qid}, Average Loss: {total_loss / len(train_dataloader):.4f}")
        
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in valid_dataloader:
                input_ids = batch['input_ids'].cuda()
                attention_mask = batch['attention_mask'].cuda()
                labels = torch.tensor(batch['labels'][qid]).cuda()
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        valid_acc = accuracy_score(all_labels, all_preds)
        print(f"Validation Accuracy for {qid}: {valid_acc:.4f}")
        
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            model.save_pretrained(os.path.join(output_dir, f"bert_{qid}"))
            train_dataset.tokenizer.save_pretrained(os.path.join(output_dir, f"bert_{qid}"))
    
    print(f"Best model for {qid} saved with validation accuracy: {best_valid_acc:.4f}")

if __name__ == "__main__":
    synthetic_file = "/kaggle/working/synthetic_train.json"
    valid_file = "/kaggle/input/mediqa-data/mediqa-data/valid.json"
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
        train_bert_model(qid, num_labels, synthetic_file, valid_file, output_dir)