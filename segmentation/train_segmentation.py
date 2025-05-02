import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from segmentation.utils import MediQADataset
from tqdm import tqdm
import timm

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        self.encoder = timm.create_model('resnet18', pretrained=True, features_only=True)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        enc = self.encoder(x)
        x = self.decoder(enc[-1])
        return x

def train_segmentation():
    data_dir = "/kaggle/input/mediqa-data/mediqa-data/"
    transform = transforms.Compose([transforms.ToTensor()])
    
    dataset = MediQADataset(data_dir, split='train', transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    model = UNet().cuda()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(10):
        model.train()
        for images, masks in tqdm(dataloader):
            images, masks = images.cuda(), masks.cuda()
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    
    torch.save(model.state_dict(), "/kaggle/working/unet_segmentation.pth")

if __name__ == "__main__":
    train_segmentation()