import torch
import torch.nn as nn
from torchvision.models import resnet18

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        return 1 - ((2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth))

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        # Sử dụng ResNet18 làm encoder
        resnet = resnet18(weights='IMAGENET1K_V1')
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64 channels, 256x256
        self.encoder2 = resnet.layer1  # 64 channels, 256x256
        self.encoder3 = resnet.layer2  # 128 channels, 128x128
        self.encoder4 = resnet.layer3  # 256 channels, 64x64
        self.encoder5 = resnet.layer4  # 512 channels, 32x32
        
        self.pool = nn.MaxPool2d(2)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )  # 1024 channels, 32x32
        
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)  # 512 channels, 64x64
        self.decoder4 = self.double_conv(768, 512)  # 512 (upconv4) + 256 (encoder4) = 768, 64x64
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)  # 256 channels, 128x128
        self.decoder3 = self.double_conv(384, 256)  # 256 (upconv3) + 128 (encoder3) = 384, 128x128
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)  # 128 channels, 256x256
        self.decoder2 = self.double_conv(192, 128)  # 128 (upconv2) + 64 (encoder2) = 192, 256x256
        self.decoder1 = self.double_conv(128, 64)  # 64 channels, 256x256
        
        # Tầng upsample để đạt kích thước 256x256
        self.upsample = nn.ConvTranspose2d(64, 64, 2, stride=2)  # 64 channels, 256x256
        self.final_conv = nn.Conv2d(64, out_channels, 1)  # 1 channel, 256x256
    
    def double_conv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Input: [batch_size, 3, 256, 256]
        e1 = self.encoder1(x)  # 64 channels, 256x256
        e2 = self.encoder2(e1)  # 64 channels, 256x256
        e3 = self.encoder3(e2)  # 128 channels, 128x128
        e4 = self.encoder4(e3)  # 256 channels, 64x64
        e5 = self.encoder5(e4)  # 512 channels, 32x32
        
        b = self.bottleneck(e5)  # 1024 channels, 32x32
        
        d4 = self.upconv4(b)  # 512 channels, 64x64
        d4 = torch.cat([d4, e4], dim=1)  # 512 + 256 = 768, 64x64
        d4 = self.decoder4(d4)  # 512 channels, 64x64
        d3 = self.upconv3(d4)  # 256 channels, 128x128
        d3 = torch.cat([d3, e3], dim=1)  # 256 + 128 = 384, 128x128
        d3 = self.decoder3(d3)  # 256 channels, 128x128
        d2 = self.upconv2(d3)  # 128 channels, 256x256
        d2 = torch.cat([d2, e2], dim=1)  # 128 + 64 = 192, 256x256
        d2 = self.decoder2(d2)  # 128 channels, 256x256
        d1 = self.decoder1(d2)  # 64 channels, 256x256
        
        # Upsample để đạt kích thước 256x256
        out = self.upsample(d1)  # 64 channels, 256x256
        out = self.final_conv(out)  # 1 channel, 256x256
        return torch.sigmoid(out)  # [batch_size, 1, 256, 256]