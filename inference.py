import os
import torch
from torch.utils.data import DataLoader
from data.process_data import MediqaDataset
from models.unet_seg import UNet
from models.blip2_qa import BLIP2QA
from utils.helpers import save_mask, save_qa_results
from torchvision import transforms
from tqdm import tqdm

def run_inference(data_dir, query_file, closed_qa_file, output_dir, mode='test'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.join(output_dir, 'masks_preds'), exist_ok=True)
    
    # Tải mô hình UNet
    unet = UNet().to(device)
    unet.load_state_dict(torch.load('/kaggle/working/unet_model.pth'))
    unet.eval()
    
    # Khởi tạo BLIP2QA
    blip2 = BLIP2QA(device=device)
    
    # Đồng bộ transform với process_data.py
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    dataset = MediqaDataset(data_dir, query_file, closed_qa_file, mode=mode, transform=transform)
    
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Check if images exist in the data directory or if test.json contains valid image_ids.")
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    qa_results = []
    
    # Sử dụng tqdm để hiển thị thanh tiến trình
    with tqdm(total=len(dataloader), desc="Running inference", unit="image") as pbar:
        for batch in dataloader:
            image = batch['image'].to(device)
            prompt = batch['prompt']
            qid = batch['qid']
            options = batch['options']
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
                    qa_results.append({
                        'encounter_id': encounter_id,
                        qid: option_idx
                    })
            
            # Cập nhật thanh tiến trình
            pbar.update(1)
    
    # Lưu kết quả QA
    save_qa_results(qa_results, os.path.join(output_dir, 'data_cvqa_sys.json'))

if __name__ == "__main__":
    data_dir = "/kaggle/input/mediqa-data/mediqa-data/"
    query_file = "/kaggle/input/mediqa-data/mediqa-data/test.json"
    closed_qa_file = "/kaggle/input/mediqa-data/mediqa-data/closedquestions_definitions_imageclef2025.json"
    output_dir = "/kaggle/working/output/"
    run_inference(data_dir, query_file, closed_qa_file, output_dir)