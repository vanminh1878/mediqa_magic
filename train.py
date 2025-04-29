import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from data.process_data import MediqaDataset
from models.unet_seg import UNet, DiceLoss
from utils.metrics import jaccard_index, dice_score
from tqdm import tqdm
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def train_unet(data_dir, query_file, closed_qa_file, epochs=10, batch_size=2, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = MediqaDataset(data_dir, query_file, closed_qa_file, mode='train')
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Check image and mask files.")
    
    def collate_fn(batch):
        # Lọc bỏ mẫu không hợp lệ
        batch = [item for item in batch if item is not None]
        if not batch:
            return None
        
        # Đảm bảo cấu trúc đồng nhất
        try:
            return torch.utils.data.dataloader.default_collate(batch)
        except Exception as e:
            logger.info(f"Collate error: {e}")
            return None
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    
    model = UNet().to(device)
    bce_loss = nn.BCELoss()
    dice_loss = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        jaccard_total = 0.0
        dice_total = 0.0
        batch_count = 0
        
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for batch in dataloader:
                if batch is None:
                    pbar.update(1)
                    continue
                
                try:
                    images = batch['image'].to(device)
                    masks = batch['mask'].to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = 0.5 * bce_loss(outputs, masks) + 0.5 * dice_loss(outputs, masks)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    jaccard_total += jaccard_index(outputs, masks).item()
                    dice_total += dice_score(outputs, masks).item()
                    batch_count += 1
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        "Loss": f"{running_loss/batch_count:.4f}" if batch_count > 0 else "N/A",
                        "Jaccard": f"{jaccard_total/batch_count:.4f}" if batch_count > 0 else "N/A",
                        "Dice": f"{dice_total/batch_count:.4f}" if batch_count > 0 else "N/A"
                    })
                except Exception as e:
                    logger.info(f"Batch processing error: {e}")
                    pbar.update(1)
                    continue
        
        if batch_count == 0:
            logger.info(f"Epoch {epoch+1}: No valid batches processed")
            continue
        
        epoch_loss = running_loss / batch_count
        epoch_jaccard = jaccard_total / batch_count
        epoch_dice = dice_total / batch_count
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Jaccard: {epoch_jaccard:.4f}, Dice: {epoch_dice:.4f}")
    
    torch.save(model.state_dict(), '/kaggle/working/unet_model.pth')

if __name__ == "__main__":
    data_dir = "/kaggle/input/mediqa-data/mediqa-data/"
    query_file = "/kaggle/input/mediqa-data/mediqa-data/train_cvqa.json"
    closed_qa_file = "/kaggle/input/mediqa-data/mediqa-data/closedquestions_definitions_imageclef2025.json"
    train_unet(data_dir, query_file, closed_qa_file)