import os
import json
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

def post_process_qa_results(qa_results_dict, closed_qa_dict):
    """Kiểm tra tính nhất quán giữa các câu hỏi liên quan"""
    for enc_id, qid_dict in qa_results_dict.items():
        # Nhóm câu hỏi CQID011 (vị trí tổn thương)
        location_qids = [qid for qid in qid_dict if qid.startswith('CQID011-')]
        if location_qids:
            # Nếu có câu hỏi vị trí chọn cụ thể (không phải "Not mentioned"), CQID010-001 không nên là "Not mentioned"
            location_selected = any(
                qid_dict[qid] != len(closed_qa_dict[[qa['qid'] for qa in closed_qa_dict].index(qid)]['options_en']) - 1
                for qid in location_qids
            )
            if 'CQID010-001' in qid_dict:
                not_mentioned_idx = len(closed_qa_dict[[qa['qid'] for qa in closed_qa_dict].index('CQID010-001')]['options_en']) - 1
                if location_selected and qid_dict['CQID010-001'] == not_mentioned_idx:
                    # Chọn "limited area" thay vì "Not mentioned"
                    qid_dict['CQID010-001'] = 1
                    logger.info(f"Updated CQID010-001 for encounter_id {enc_id} to 'limited area' for consistency")
    
    return qa_results_dict

def run_inference(data_dir, query_file, closed_qa_file, output_dir, mode='test'):
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

    # Khởi tạo BLIP2QA
    try:
        blip2 = BLIP2QA(device=device)
        logger.info("BLIP2QA model initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing BLIP2QA: {e}")
        raise

    # Đồng bộ transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Tải dataset
    try:
        dataset = MediqaDataset(data_dir, query_file, closed_qa_file, mode=mode, transform=transform)
        logger.info(f"Dataset loaded with {len(dataset)} samples")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

    # Kiểm tra số mẫu
    expected_samples = 314
    logger.info(f"Expected samples: {expected_samples}, Actual samples: {len(dataset)}")
    if len(dataset) < expected_samples:
        logger.warning(f"Dataset has fewer samples than expected: {len(dataset)}/{expected_samples}")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Tải test.json để lấy danh sách tất cả encounter_id
    try:
        with open(query_file, 'r') as f:
            test_queries = json.load(f)
        all_encounter_ids = set(query.get('encounter_id', '') for query in test_queries if query.get('encounter_id'))
        logger.info(f"Total encounter_ids in test.json: {len(all_encounter_ids)}")
    except Exception as e:
        logger.error(f"Error loading test.json: {e}")
        raise

    # Tải closed_qa_dict để dùng trong post-processing
    try:
        with open(closed_qa_file, 'r') as f:
            closed_qa_dict = json.load(f)
    except Exception as e:
        logger.error(f"Error loading closed_qa_file: {e}")
        raise

    qa_results_dict = {}
    processed_samples = 0
    skipped_samples = []

    # Lặp qua dataloader
    with tqdm(total=len(dataloader), desc="Running inference", unit="sample") as pbar:
        for batch in dataloader:
            try:
                encounter_id = batch['encounter_id'][0]
                image_id = batch['image_id'][0]
                qid = batch['qid'][0]
                options = batch['options'][0]
                prompt = batch['prompt'][0]
                image = batch['image'].to(device) if batch['image'] is not None else None

                # Dự đoán mặt nạ nếu có hình ảnh
                if image is not None:
                    try:
                        with torch.no_grad():
                            mask_pred = unet(image)
                            mask_pred = mask_pred.squeeze().cpu().numpy()
                            save_mask(mask_pred, encounter_id, image_id, os.path.join(output_dir, 'masks_preds'))
                            logger.debug(f"Saved mask for encounter_id: {encounter_id}, image_id: {image_id}")
                    except Exception as e:
                        logger.warning(f"Error predicting mask for encounter_id: {encounter_id}, image_id: {image_id}, error: {e}")

                # Trả lời câu hỏi
                if qid and options:
                    try:
                        option_idx = blip2.answer_question(image, prompt, options)
                        if option_idx >= 0:
                            if encounter_id not in qa_results_dict:
                                qa_results_dict[encounter_id] = {}
                            qa_results_dict[encounter_id][qid] = option_idx
                            logger.debug(f"QA result for encounter_id: {encounter_id}, qid: {qid}, option_idx: {option_idx}")
                        else:
                            logger.warning(f"Invalid option_idx for encounter_id: {encounter_id}, image_id: {image_id}, qid: {qid}")
                            skipped_samples.append((encounter_id, image_id, f"Invalid option_idx for qid: {qid}"))
                    except Exception as e:
                        logger.warning(f"Error answering question for encounter_id: {encounter_id}, image_id: {image_id}, qid: {qid}, error: {e}")
                        skipped_samples.append((encounter_id, image_id, f"QA error: {e}"))
                else:
                    logger.warning(f"Skipping QA for encounter_id: {encounter_id}, image_id: {image_id}, qid: {qid}, options: {options}")
                    skipped_samples.append((encounter_id, image_id, "Missing qid or options"))

                processed_samples += 1
                pbar.update(1)

                # Giải phóng bộ nhớ
                del image
                torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Error processing encounter_id: {encounter_id}, image_id: {image_id}, error: {e}")
                skipped_samples.append((encounter_id, image_id, f"Processing error: {e}"))
                pbar.update(1)
                continue

    # Xử lý các encounter_id không có trong dataset
    missing_encounter_ids = all_encounter_ids - set(qa_results_dict.keys())
    for enc_id in missing_encounter_ids:
        logger.warning(f"Missing encounter_id in results: {enc_id}")
        qa_results_dict[enc_id] = {}
        skipped_samples.append((enc_id, "", "Missing in dataset"))

    # Tối ưu hóa sau xử lý
    qa_results_dict = post_process_qa_results(qa_results_dict, closed_qa_dict)

    # Báo cáo kết quả
    logger.info(f"Processed samples: {processed_samples}/{expected_samples}")
    if skipped_samples:
        logger.warning(f"Skipped samples: {len(skipped_samples)}")
        for enc_id, img_id, err in skipped_samples:
            logger.warning(f"Skipped encounter_id: {enc_id}, image_id: {img_id}, error: {err}")

    # Chuyển qa_results_dict thành danh sách
    qa_results = [
        {"encounter_id": enc_id, **qid_dict}
        for enc_id, qid_dict in qa_results_dict.items()
    ]

    # Lưu kết quả QA
    output_file = os.path.join(output_dir, 'data_cvqa_sys.json')
    try:
        save_qa_results(qa_results, output_file)
        logger.info(f"QA results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving QA results: {e}")
        raise

    # Kiểm tra số encounter_id trong kết quả
    logger.info(f"Total encounter_ids in output: {len(qa_results)}/{expected_samples}")
    if len(qa_results) < expected_samples:
        logger.warning(f"Output contains fewer encounter_ids than expected: {len(qa_results)}/{expected_samples}")

if __name__ == "__main__":
    data_dir = "/kaggle/input/mediqa-data/mediqa-data/"
    query_file = "/kaggle/input/mediqa-data/mediqa-data/test.json"
    closed_qa_file = "/kaggle/input/mediqa-data/mediqa-data/closedquestions_definitions_imageclef2025.json"
    output_dir = "/kaggle/working/output/"
    run_inference(data_dir, query_file, closed_qa_file, output_dir)