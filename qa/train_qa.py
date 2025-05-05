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
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
logging.getLogger("transformers").setLevel(logging.ERROR)

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

class MediQAQADataset(Dataset):
    def __init__(self, query_file, data_dir, split='train', transform=None, bert_model_dir="/kaggle/working/bert_models"):
        self.data_dir = data_dir
        self.transform = transform
        self.image_dir = os.path.join(data_dir, 'images')
        self.mask_dir = os.path.join(data_dir, f'masks_{split}')
        self.bert_model_dir = bert_model_dir
        
        with open(query_file, 'r') as f:
            self.data = json.load(f)
        
        with open('/kaggle/input/mediqa-data/mediqa-data/closedquestions_definitions_imageclef2025.json', 'r') as f:
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
        
        images = []
        loaded_image_ids = []
        for img_id in item.get('image_ids', [])[:3]:  # Limit to 3 images
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
            images = [torch.zeros(3, 224, 224)]
        
        images = torch.stack(images)
        
        closed_qa = item.get('closed_qa', {})
        if not closed_qa:
            closed_qa = self.extract_closed_qa(query)
        
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
        batch_size, num_images, c, h, w = images.shape
        images = images.view(-1, c, h, w)
        outputs = self.clip(pixel_values=images, input_ids=queries['input_ids'].repeat(num_images, 1), 
                           attention_mask=queries['attention_mask'].repeat(num_images, 1))
        img_features = outputs.image_embeds
        text_features = outputs.text_embeds
        
        img_features = img_features.view(batch_size, num_images, -1).mean(dim=1)
        text_features = text_features.view(batch_size, num_images, -1).mean(dim=1)
        
        print(f"img_features mean: {img_features.mean().item()}, text_features mean: {text_features.mean().item()}")
        
        combined = self.dropout(torch.cat([img_features, text_features], dim=1))
        outputs = {qid: fc(combined) for qid, fc in self.fc_layers.items()}
        return outputs

def evaluate(model, dataloader, processor, criteria):
    model.eval()
    all_preds = {qid: [] for qid in model.qids}
    all_labels = {qid: [] for qid in model.qids}
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['images'].cuda()
            queries = batch['query']
            closed_qa = batch['closed_qa']
            inputs = processor(text=queries, images=None, return_tensors='pt', 
                             padding=True, truncation=True, max_length=77).to('cuda')
            outputs = model(images, inputs)
            for qid in model.qids:
                if qid in closed_qa and closed_qa[qid] is not None:
                    preds = torch.argmax(outputs[qid], dim=1).cpu().numpy()
                    labels = [closed_qa[qid]] * len(preds)  # Repeat label for batch
                    all_preds[qid].extend(preds)
                    all_labels[qid].extend(labels)
    accuracies = {qid: accuracy_score(all_labels[qid], all_preds[qid]) for qid in model.qids if all_labels[qid]}
    f1_scores = {qid: f1_score(all_labels[qid], all_preds[qid], average='weighted') for qid in model.qids if all_labels[qid]}
    return accuracies, f1_scores

def train_closed_qa():
    data_dir = "/kaggle/input/mediqa-data/mediqa-data/"
    query_file = "/kaggle/input/mediqa-data/mediqa-data/train_cvqa.json"
    synthetic_file = "/kaggle/working/synthetic_train.json"
    valid_file = "/kaggle/input/mediqa-data/mediqa-data/valid.json"
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.269, 0.261, 0.276])
    ])
    
    train_dataset = MediQAQADataset(query_file, data_dir, split='train', transform=transform)
    synthetic_dataset = MediQAQADataset(synthetic_file, data_dir, split='train', transform=transform)
    valid_dataset = MediQAQADataset(valid_file, data_dir, split='valid', transform=transform)
    
    train_dataloader = DataLoader(train_dataset + synthetic_dataset, batch_size=2, shuffle=True, num_workers=0)
    valid_dataloader = DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=0)
    
    model = ClosedQAModel().cuda()
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch16')
    
    criteria = {
        4: FocalLoss(alpha=7.0, gamma=2),
        8: FocalLoss(alpha=7.0, gamma=2),
        7: FocalLoss(alpha=7.0, gamma=2),
        10: FocalLoss(alpha=7.0, gamma=2),
        3: FocalLoss(alpha=7.0, gamma=2),
        12: FocalLoss(alpha=7.0, gamma=2)
    }
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = GradScaler()
    
    best_valid_loss = float('inf')
    patience_counter = 0
    patience_limit = 5
    
    for param in model.clip.vision_model.parameters():
        param.requires_grad = False
    
    for epoch in range(10):
        if epoch == 3:
            for param in model.clip.vision_model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
        
        model.train()
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}", leave=True, ncols=100):
            images = batch['images'].cuda()
            batch_size, num_images, c, h, w = images.shape
            queries = batch['query']
            closed_qa = batch['closed_qa']
            
            with autocast():
                inputs = processor(text=queries, images=None, return_tensors='pt', 
                                 padding=True, truncation=True, max_length=77).to('cuda')
                outputs = model(images, inputs)
                
                loss = 0
                for qid in model.qids:
                    if qid in closed_qa and closed_qa[qid] is not None:
                        labels = torch.tensor([closed_qa[qid]]).cuda()
                        num_classes = (
                            4 if qid.startswith('CQID010') else
                            8 if qid.startswith('CQID011') else
                            4 if qid.startswith('CQID012') else
                            7 if qid == 'CQID015-001' else
                            10 if qid.startswith('CQID020') else
                            3 if qid == 'CQID025-001' else
                            12 if qid == 'CQID034-001' else
                            3 if qid == 'CQID035-001' else
                            3
                        )
                        criterion = criteria[num_classes]
                        loss += criterion(outputs[qid], labels.long())
            
            total_loss += loss.item()
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
        
        accuracies, f1_scores = evaluate(model, valid_dataloader, processor, criteria)
        print(f"Validation Accuracies: {accuracies}")
        print(f"Validation F1 Scores: {f1_scores}")
        
        valid_loss = sum(1 - acc for acc in accuracies.values()) / len(accuracies)
        scheduler.step(valid_loss)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
            torch.save(model.state_dict(), "/kaggle/working/closed_qa_clip_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print("Early stopping triggered")
                break
    
    torch.save(model.state_dict(), "/kaggle/working/closed_qa_clip.pth")

if __name__ == "__main__":
    train_closed_qa()