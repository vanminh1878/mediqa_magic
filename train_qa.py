import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from data.process_data_qa import MediqaQADataset
from models.bert_vqa import BertVQA
from tqdm import tqdm
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    images = torch.stack([item['image'] for item in batch])
    queries = [item['query'] for item in batch]
    prompts = [item['prompt'] for item in batch]
    qids = [item['qid'] for item in batch]
    question_indices = torch.tensor([item['question_index'] for item in batch])
    options = [item['options'] for item in batch]
    encounter_ids = [item['encounter_id'] for item in batch]
    image_ids = [item['image_id'] for item in batch]
    
    return {
        'image': images,
        'query': queries,
        'prompt': prompts,
        'qid': qids,
        'question_index': question_indices,
        'options': options,
        'encounter_id': encounter_ids,
        'image_id': image_ids
    }

def train_bert(data_dir, train_query_file, valid_query_file, closed_qa_file, epochs=3, batch_size=8, lr=2e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = MediqaQADataset(data_dir, train_query_file, closed_qa_file, mode='train', transform=transform)
    valid_dataset = MediqaQADataset(data_dir, valid_query_file, closed_qa_file, mode='valid', transform=transform)
    
    if len(train_dataset) == 0 or len(valid_dataset) == 0:
        raise ValueError("Dataset is empty. Check data files.")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
    
    # Load model
    bert_vqa = BertVQA(device=device)
    model = bert_vqa.bert_model  # Only fine-tune BERT for mapping
    model.train()
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Load ground truth labels
    with open(train_query_file, 'r') as f:
        train_queries = json.load(f)
    with open(valid_query_file, 'r') as f:
        valid_queries = json.load(f)
    
    train_labels = {q['encounter_id']: {k: v for k, v in q.items() if k.startswith('CQID')} for q in train_queries}
    valid_labels = {q['encounter_id']: {k: v for k, v in q.items() if k.startswith('CQID')} for q in valid_queries}
    
    # Map qids to indices
    with open(closed_qa_file, 'r') as f:
        closed_qa_dict = json.load(f)
    qid_to_idx = {qa['qid']: idx for idx, qa in enumerate(closed_qa_dict)}
    
    best_valid_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for batch in train_loader:
                if batch is None:
                    pbar.update(1)
                    continue
                
                try:
                    queries = batch['query']
                    qids = batch['qid']
                    encounter_ids = batch['encounter_id']
                    
                    optimizer.zero_grad()
                    
                    # Encode queries
                    inputs = model.encode(queries, convert_to_tensor=True, device=device)
                    
                    # Get labels
                    labels = []
                    valid_indices = []
                    for idx, (enc_id, qid) in enumerate(zip(encounter_ids, qids)):
                        label = train_labels.get(enc_id, {}).get(qid, 0)  # Default to 0
                        if label is None:
                            logger.warning(f"Invalid label for encounter_id: {enc_id}, qid: {qid}. Using default label 0.")
                            label = 0
                        labels.append(label)
                        valid_indices.append(idx)
                    
                    if not labels:
                        logger.warning("No valid labels in batch. Skipping.")
                        pbar.update(1)
                        continue
                    
                    labels = torch.tensor(labels).to(device)
                    if len(valid_indices) < batch_size:
                        inputs = inputs[valid_indices]
                    
                    # Forward pass (use embeddings as logits for classification)
                    logits = model(inputs)  # Assume SentenceTransformer outputs embeddings
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    preds = logits.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += len(labels)
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        "Loss": f"{running_loss/(pbar.n+1):.4f}",
                        "Acc": f"{correct/total:.4f}" if total > 0 else "N/A"
                    })
                except Exception as e:
                    logger.info(f"Batch processing error: {e}")
                    pbar.update(1)
                    continue
        
        epoch_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        epoch_acc = correct / total if total > 0 else 0.0
        logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")
        
        # Evaluate on valid
        model.eval()
        valid_correct = 0
        valid_total = 0
        with torch.no_grad():
            for batch in valid_loader:
                if batch is None:
                    continue
                
                try:
                    queries = batch['query']
                    qids = batch['qid']
                    encounter_ids = batch['encounter_id']
                    
                    inputs = model.encode(queries, convert_to_tensor=True, device=device)
                    
                    labels = []
                    valid_indices = []
                    for idx, (enc_id, qid) in enumerate(zip(encounter_ids, qids)):
                        label = valid_labels.get(enc_id, {}).get(qid, 0)  # Default to 0
                        if label is None:
                            logger.warning(f"Invalid label for encounter_id: {enc_id}, qid: {qid}. Using default label 0.")
                            label = 0
                        labels.append(label)
                        valid_indices.append(idx)
                    
                    if not labels:
                        continue
                    
                    labels = torch.tensor(labels).to(device)
                    if len(valid_indices) < batch_size:
                        inputs = inputs[valid_indices]
                    
                    logits = model(inputs)
                    preds = logits.argmax(dim=1)
                    valid_correct += (preds == labels).sum().item()
                    valid_total += len(labels)
                except Exception as e:
                    logger.info(f"Valid batch processing error: {e}")
                    continue
        
        valid_acc = valid_correct / valid_total if valid_total > 0 else 0.0
        logger.info(f"Epoch {epoch+1}/{epochs}, Valid Acc: {valid_acc:.4f}")
        
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), '/kaggle/working/bert_model.pth')
            logger.info(f"Saved best model at epoch {epoch+1}")

if __name__ == "__main__":
    data_dir = "/kaggle/input/mediqa-data/mediqa-data/"
    train_query_file = "/kaggle/working/train_cvqa_labeled.json"
    valid_query_file = "/kaggle/working/valid_cvqa_labeled.json"
    closed_qa_file = "/kaggle/input/mediqa-data/mediqa-data/closedquestions_definitions_imageclef2025.json"
    train_bert(data_dir, train_query_file, valid_query_file, closed_qa_file)