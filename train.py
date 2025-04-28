import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from data.process_data import MediqaDataset
from models.unet_seg import UNet, DiceLoss
from utils.metrics import jaccard_index, dice_score
from tqdm import tqdm

def train_unet(data_dir, query_file, closed_qa_file, epochs=10, batch_size=2, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2)
    ])
    
    dataset = MediqaDataset(data_dir, query_file, closed_qa_file, mode='train', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = UNet().to(device)
    bce_loss = nn.BCELoss()
    dice_loss = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        jaccard_total = 0.0
        dice_total = 0.0
        for batch in tqdm(dataloader):
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
        
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}, "
              f"Jaccard: {jaccard_total/len(dataloader)}, Dice: {dice_total/len(dataloader)}")
    
    torch.save(model.state_dict(), '/kaggle/working/unet_model.pth')

if __name__ == "__main__":
    data_dir = "/kaggle/input/mediqa-data/mediqa-data/"
    query_file = "/kaggle/input/mediqa-data/mediqa-data/train.json"
    closed_qa_file = "/kaggle/input/mediqa-data/mediqa-data/closedquestions_definitions_imageclef2025.json"
    train_unet(data_dir, query_file, closed_qa_file)