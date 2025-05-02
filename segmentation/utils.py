import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import tifffile

class MediQADataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.image_dir = os.path.join(data_dir, 'images')
        self.mask_dir = os.path.join(data_dir, split, 'masks')
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith(('.png', '.jpg'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = img_name.replace('.jpg', '_mask_ann0.tiff').replace('.png', '_mask_ann0.tiff')
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        image = Image.open(img_path).convert('RGB')
        mask = tifffile.imread(mask_path) if os.path.exists(mask_path) else np.zeros((image.height, image.width))
        
        if self.transform:
            image = self.transform(image)
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
        
        return image, mask