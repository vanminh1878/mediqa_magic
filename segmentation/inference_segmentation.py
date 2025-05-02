import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from segmentation.train_segmentation import UNet
from tqdm import tqdm

def inference_segmentation(split='valid'):
    data_dir = "/kaggle/input/mediqa-data/mediqa-data/"
    output_dir = f"/kaggle/working/masks_{split}"
    os.makedirs(output_dir, exist_ok=True)
    
    transform = transforms.Compose([transforms.ToTensor()])
    model = UNet().cuda()
    model.load_state_dict(torch.load("/kaggle/working/unet_segmentation.pth"))
    model.eval()
    
    image_dir = os.path.join(data_dir, 'images')
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))]
    
    with torch.no_grad():
        for img_name in tqdm(image_files):
            img_path = os.path.join(image_dir, img_name)
            image = Image.open(img_path).convert('RGB')
            image = transform(image).unsqueeze(0).cuda()
            
            mask = model(image)
            mask = (mask.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
            Image.fromarray(mask).save(os.path.join(output_dir, img_name.replace('.jpg', '_mask.png')))
    
    print(f"Masks saved to {output_dir}")

if __name__ == "__main__":
    for split in ['valid', 'test']:
        inference_segmentation(split)