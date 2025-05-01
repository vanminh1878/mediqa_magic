import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import logging
import json

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MediqaSegDataset(Dataset):
    def __init__(self, data_dir, query_file, mode='train', transform=None):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, 'images')
        # Sửa mask_dir để trỏ đến images/masks/{mode}
        self.mask_dir = os.path.join(data_dir, 'images', 'masks', mode)
        self.mode = mode
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        # Kiểm tra sự tồn tại của thư mục
        logger.info(f"Image dir: {self.image_dir}")
        logger.info(f"Mask dir: {self.mask_dir}")
        if not os.path.exists(self.image_dir):
            logger.error(f"Image directory {self.image_dir} does not exist")
            raise FileNotFoundError(f"Image directory {self.image_dir} does not exist")
        if not os.path.exists(self.mask_dir):
            logger.error(f"Mask directory {self.mask_dir} does not exist")
            raise FileNotFoundError(f"Mask directory {self.mask_dir} does not exist")
        
        # Đọc file query
        with open(query_file, 'r') as f:
            self.queries = json.load(f)
        
        self.image_files = []
        self.masks = []
        self.skipped_samples = []
        
        with tqdm(total=len(self.queries), desc="Processing queries", unit="query") as pbar:
            for query in self.queries:
                encounter_id = query.get('encounter_id', '')
                if not encounter_id:
                    logger.warning(f"Missing encounter_id in query: {query}")
                    self.skipped_samples.append((encounter_id, "Missing encounter_id"))
                    pbar.update(1)
                    continue
                
                image_ids = []
                for img_file in os.listdir(self.image_dir):
                    if img_file.startswith(f'IMG_{encounter_id}_') and (img_file.endswith('.png') or img_file.endswith('.jpg')):
                        img_id = img_file.replace(f'IMG_{encounter_id}_', '').rsplit('.', 1)[0]
                        image_ids.append(img_id)
                
                if not image_ids:
                    logger.warning(f"No valid image_ids for encounter_id: {encounter_id}")
                    self.skipped_samples.append((encounter_id, "No valid image_ids"))
                    pbar.update(1)
                    continue
                
                for img_id in image_ids:
                    img_path = os.path.join(self.image_dir, f'IMG_{encounter_id}_{img_id}.png')
                    if not os.path.exists(img_path):
                        img_path = os.path.join(self.image_dir, f'IMG_{encounter_id}_{img_id}.jpg')
                    
                    try:
                        Image.open(img_path).convert('RGB')
                        mask_path = None
                        for suffix in ['_mask_ann0.tiff', '_mask_ann1.tiff', '_mask_ann2.tiff', '_mask_ann3.tiff']:
                            temp_path = os.path.join(self.mask_dir, f'IMG_{encounter_id}_{img_id}{suffix}')
                            if os.path.exists(temp_path):
                                mask_path = temp_path
                                break
                        if mask_path:
                            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                            if mask_img is None:
                                logger.warning(f"Invalid mask {mask_path}")
                                self.skipped_samples.append((encounter_id, f"Invalid mask for image_id: {img_id}"))
                                continue
                        else:
                            logger.warning(f"No mask found for {img_path}")
                            self.skipped_samples.append((encounter_id, f"No mask for image_id: {img_id}"))
                            continue
                    except Exception as e:
                        logger.warning(f"Failed to load image {img_path}: {e}")
                        self.skipped_samples.append((encounter_id, f"Failed to load image_id: {img_id}, error: {e}"))
                        continue
                    
                    if os.path.exists(img_path) and mask_path:
                        self.image_files.append(img_path)
                        self.masks.append(mask_path)
                
                pbar.update(1)
        
        logger.info(f"Total images in dataset: {len(self.image_files)}")
        logger.info(f"Image files: {self.image_files}")
        logger.info(f"Mask files: {self.masks}")
        if self.skipped_samples:
            logger.warning(f"Skipped samples: {len(self.skipped_samples)}")
            for enc_id, reason in self.skipped_samples:
                logger.warning(f"Skipped encounter_id: {enc_id}, reason: {reason}")
        if len(self.image_files) == 0:
            logger.error("No valid samples found in dataset")
            raise ValueError("No valid samples found in dataset")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            transformed_image = self.transform(image)
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            raise ValueError(f"Invalid image at index {idx}")
        
        try:
            mask_img = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                logger.error(f"Invalid mask {self.masks[idx]}")
                raise ValueError(f"Invalid mask at index {idx}")
            mask = mask_img / 255.0
            mask = Image.fromarray(mask)
            if self.transform:
                mask = self.transform(mask)
                mask = mask.squeeze()
                mask = mask.unsqueeze(0)
        except Exception as e:
            logger.error(f"Error loading mask {self.masks[idx]}: {e}")
            raise ValueError(f"Invalid mask at index {idx}")
        
        return {
            'image': transformed_image,
            'mask': mask,
            'image_path': img_path
        }