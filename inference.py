import os
import torch
from torch.utils.data import DataLoader
from data.process_data import MediqaDataset
from models.unet_seg import UNet
from models.blip2_qa import BLIP2QA
from utils.helpers import save_mask, save_qa_results
from torchvision import transforms
from tqdm import tqdm
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

def run_inference(data_dir, query_file, closed_qa_file, output_dir, mode='test'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.join(output_dir, 'masks_preds'), exist_ok=True)

    # Tải mô hình UNet
    unet = UNet().to(device)
    try:
        unet.load_state_dict(torch.load('/kaggle/working/unet_model.pth'))
    except Exception as e:
        logger.error(f"Error loading UNet model: {e}")
        raise
    unet.eval()

    # Khởi tạo BLIP2QA
    blip2 = BLIP2QA(device=device)

    # Đồng bộ transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    dataset = MediqaDataset(data_dir, query_file, closed_qa_file, mode=mode, transform=transform)

    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Check if images exist or test.json contains valid image_ids.")
    
    expected_samples = 314  # Số mẫu mong đợi trong tập test
    logger.info(f"Expected samples: {expected_samples}, Actual samples: {len(dataset)}")
    if len(dataset) < expected_samples:
        logger.warning(f"Dataset has fewer samples than expected: {len(dataset)}/{expected_samples}")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    qa_results_dict = {}  # Lưu kết quả theo encounter_id
    processed_samples = 0
    skipped_samples = []

    # Chạy inference
    with tqdm(total=len(dataloader), desc="RunningSiemens inference", unit="image") as pbar:
        for batch in dataloader:
            try:
                image = batch['image'].to(device)
                prompt = batch['prompt'][0]  # Lấy chuỗi từ tensor
                qid = batch['qid'][0]  # Lấy chuỗi từ tensor
                options = batch['options'][0]  # Lấy danh sách từ tensor
                encounter_id = batch['encounter_id'][0]
                image_id = batch['image_id'][0]
                
                # Dự đoán mặt nạ
                with torch.no_grad():
                    mask_pred = unet(image)
                    mask_pred = mask_pred.squeeze().cpu().numpy()
                    save_mask(mask_pred, encounter_id, image_id, os.path.join(output_dir, 'masks_preds'))
                
                # Trả lời câu hỏi nếu có qid và options
                if qid and options:
                    option_idx = blip2.answer_question(image, prompt, options)
                    if option_idx >= 0:
                        if encounter_id not in qa_results_dict:
                            qa_results_dict[encounter_id] = {}
                        qa_results_dict[encounter_id][qid] = option_idx
                    else:
                        logger.info(f"Invalid option_idx for encounter_id: {encounter_id}, image_id: {image_id}, qid: {qid}")
                else:
                    logger.info(f"Skipping QA for encounter_id: {encounter_id}, image_id: {image_id}, qid: {qid}, options: {options}")
                
                processed_samples += 1
                pbar.update(1)
                
                # Xóa tensor để giải phóng bộ nhớ
                del image, mask_pred
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error processing encounter_id: {encounter_id}, image_id: {image_id}, error: {e}")
                skipped_samples.append((encounter_id, image_id, str(e)))
                pbar.update(1)
                continue

    # Báo cáo kết quả
    logger.info(f"Processed samples: {processed_samples}/{expected_samples}")
    if skipped_samples:
        logger.warning(f"Skipped samples: {len(skipped_samples)}")
        for enc_id, img_id, err in skipped_samples:
            logger.warning(f"Skipped encounter_id: {enc_id}, image_id: {img_id}, error: {err}")

    # Chuyển qa_results_dict thành danh sách theo định dạng yêu cầu
    qa_results = [
        {"encounter_id": enc_id, **qid_dict}
        for enc_id, qid_dict in qa_results_dict.items()
    ]

    # Lưu kết quả QA
    output_file = os.path.join(output_dir, 'data_cvqa_sys.json')
    save_qa_results(qa_results, output_file)
    logger.info(f"QA results saved to {output_file}")

if __name__ == "__main__":
    data_dir = "/kaggle/input/mediqa-data/mediqa-data/"
    query_file = "/kaggle/input/mediqa-data/mediqa-data/test.json"
    closed_qa_file = "/kaggle/input/mediqa-data/mediqa-data/closedquestions_definitions_imageclef2025.json"
    output_dir = "/kaggle/working/output/"
    run_inference(data_dir, query_file, closed_qa_file, output_dir)