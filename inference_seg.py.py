import os
import torch
from torch.utils.data import DataLoader
from data.process_data_seg import MediqaSegDataset
from models.unet_seg import UNet
from utils.helpers import save_mask
from torchvision import transforms
from tqdm import tqdm
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

def run_seg_inference(data_dir, query_file, output_dir, mode='val'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.join(output_dir, 'masks_preds'), exist_ok=True)

    # Tải mô hình UNet
    unet = UNet().to(device)
    try:
        unet.load_state_dict(torch.load('/kaggle/working/unet_model.pth', map_location=device))
        logger.info("UNet model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading UNet model: {e}")
        raise
    unet.eval()

    # Đồng bộ transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Tải dataset
    try:
        dataset = MediqaSegDataset(data_dir, query_file, mode=mode, transform=transform)
        logger.info(f"Dataset loaded with {len(dataset)} samples")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    processed_samples = 0
    with tqdm(total=len(dataloader), desc="Running segmentation inference", unit="sample") as pbar:
        for batch in dataloader:
            try:
                image = batch['image'].to(device)
                image_path = batch['image_path'][0]
                
                # Trích xuất encounter_id và image_id
                filename = os.path.basename(image_path)
                encounter_id = filename.split('_')[1]
                image_id = filename.split('_')[2].split('.')[0]
                
                # Dự đoán mặt nạ
                with torch.no_grad():
                    mask_pred = unet(image)
                    mask_pred = mask_pred.squeeze().cpu().numpy()
                    save_mask(mask_pred, encounter_id, image_id, os.path.join(output_dir, 'masks_preds'))
                    logger.debug(f"Saved mask for encounter_id: {encounter_id}, image_id: {image_id}")
                
                processed_samples += 1
                pbar.update(1)
                
                del image
                torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {e}")
                pbar.update(1)
                continue

    logger.info(f"Processed samples: {processed_samples}/{len(dataloader)}")

if __name__ == "__main__":
    data_dir = "/kaggle/input/mediqa-data/mediqa-data/"
    query_file = "/kaggle/input/mediqa-data/mediqa-data/valid.json"
    output_dir = "/kaggle/working/output/"
    run_seg_inference(data_dir, query_file, output_dir)