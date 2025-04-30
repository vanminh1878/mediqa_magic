import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from data.process_data_qa import MediqaQADataset
from models.clip_qa import CLIPQA
from tqdm import tqdm
import logging
import json

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def train_clip(data_dir, train_query_file, valid_query_file, closed_qa_file, epochs=5, batch_size=4, lr=1e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Tải dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.269, 0.271, 0.282])
    ])
    
    train_dataset = MediqaQADataset(data_dir, train_query_file, closed_qa_file, mode='train', transform=transform)
    valid_dataset = MediqaQADataset(data_dir, valid_query_file, closed_qa_file, mode='valid', transform=transform)
    
    if len(train_dataset) == 0 or len(valid_dataset) == 0:
        raise ValueError("Dataset is empty. Check data files.")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Tải mô hình
    clip_qa = CLIPQA(device=device)
    model = clip_qa.model
    model.train()
    
    # Loss và optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Tải nhãn ground truth
    with open(train_query_file, 'r') as f:
        train_queries = json.load(f)
    with open(valid_query_file, 'r') as f:
        valid_queries = json.load(f)
    
    train_labels = {q['encounter_id']: {k: v for k, v in q.items() if k.startswith('CQID')} for q in train_queries}
    valid_labels = {q['encounter_id']: {k: v for k, v in q.items() if k.startswith('CQID')} for q in valid_queries}
    
    best_valid_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for batch in train_loader:
                try:
                    images = batch['image'].to(device)
                    prompts = batch['prompt']
                    qids = batch['qid']
                    options = batch['options']
                    encounter_ids = batch['encounter_id']
                    question_indices = batch['question_index']
                    
                    optimizer.zero_grad()
                    
                    # Chuẩn bị inputs
                    batch_size = len(prompts)
                    all_text_inputs = []
                    for i in range(batch_size):
                        text_inputs = [prompts[i] + f"\nAnswer: {opt}" for opt in options[i]]
                        all_text_inputs.extend(text_inputs)
                    
                    inputs = clip_qa.processor(
                        text=all_text_inputs,
                        images=images.repeat_interleave(len(options[0]), dim=0),
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).to(device)
                    
                    # Dự đoán
                    outputs = model(**inputs)
                    logits = outputs.logits_per_image.view(batch_size, len(options[0]))
                    
                    # Tính loss
                    labels = torch.tensor([
                        train_labels[enc_id][qid] for enc_id, qid in zip(encounter_ids, qids)
                    ]).to(device)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    preds = logits.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += batch_size
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        "Loss": f"{running_loss/(pbar.n+1):.4f}",
                        "Acc": f"{correct/total:.4f}"
                    })
                except Exception as e:
                    logger.info(f"Batch processing error: {e}")
                    pbar.update(1)
                    continue
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")
        
        # Đánh giá trên valid
        model.eval()
        valid_correct = 0
        valid_total = 0
        with torch.no_grad():
            for batch in valid_loader:
                try:
                    images = batch['image'].to(device)
                    prompts = batch['prompt']
                    qids = batch['qid']
                    options = batch['options']
                    encounter_ids = batch['encounter_id']
                    
                    batch_size = len(prompts)
                    all_text_inputs = []
                    for i in range(batch_size):
                        text_inputs = [prompts[i] + f"\nAnswer: {opt}" for opt in options[i]]
                        all_text_inputs.extend(text_inputs)
                    
                    inputs = clip_qa.processor(
                        text=all_text_inputs,
                        images=images.repeat_interleave(len(options[0]), dim=0),
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).to(device)
                    
                    outputs = model(**inputs)
                    logits = outputs.logits_per_image.view(batch_size, len(options[0]))
                    
                    labels = torch.tensor([
                        valid_labels[enc_id][qid] for enc_id, qid in zip(encounter_ids, qids)
                    ]).to(device)
                    preds = logits.argmax(dim=1)
                    valid_correct += (preds == labels).sum().item()
                    valid_total += batch_size
                except Exception as e:
                    logger.info(f"Valid batch processing error: {e}")
                    continue
        
        valid_acc = valid_correct / valid_total if valid_total > 0 else 0.0
        logger.info(f"Epoch {epoch+1}/{epochs}, Valid Acc: {valid_acc:.4f}")
        
        # Lưu mô hình tốt nhất
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), '/kaggle/working/clip_model.pth')
            logger.info(f"Saved best model at epoch {epoch+1}")

if __name__ == "__main__":
    data_dir = "/kaggle/input/mediqa-data/mediqa-data/"
    train_query_file = "/kaggle/input/mediqa-data/mediqa-data/train_cvqa.json"
    valid_query_file = "/kaggle/input/mediqa-data/mediqa-data/valid.json"
    closed_qa_file = "/kaggle/input/mediqa-data/mediqa-data/closedquestions_definitions_imageclef2025.json"
    train_clip(data_dir, train_query_file, valid_query_file, closed_qa_file)