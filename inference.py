import os
import torch
from torch.utils.data import DataLoader
from data.process_data import MediqaDataset
from models.unet_seg import UNet
from models.blip2_qa import BLIP2QA
from utils.helpers import save_mask, save_qa_results
from torchvision import transforms

def run_inference(data_dir, query_file, closed_qa_file, output_dir, mode='test'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.join(output_dir, 'masks_preds'), exist_ok=True)
    
    unet = UNet().to(device)
    unet.load_state_dict(torch.load('/kaggle/working/unet_model.pth'))
    unet.eval()
    
    blip2 = BLIP2QA(device=device)
    
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = MediqaDataset(data_dir, query_file, closed_qa_file, mode=mode, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    qa_results = []
    
    for batch in dataloader:
        image = batch['image'].to(device)
        prompt = batch['prompt']
        qid = batch['qid']
        options = batch['options']
        encounter_id = batch['encounter_id'][0]
        image_id = batch['image_id'][0]
        
        with torch.no_grad():
            mask_pred = unet(image)
            mask_pred = mask_pred.squeeze().cpu().numpy()
            save_mask(mask_pred, encounter_id, image_id, os.path.join(output_dir, 'masks_preds'))
        
        if qid and options:
            option_idx = blip2.answer_question(image, prompt, options)
            if option_idx >= 0:
                qa_results.append({
                    'encounter_id': encounter_id,
                    qid: option_idx
                })
    
    save_qa_results(qa_results, os.path.join(output_dir, 'data_cvqa_sys.json'))

if __name__ == "__main__":
    data_dir = "/kaggle/input/mediqa-data/"
    query_file = "/kaggle/input/mediqa-data/test.json"
    closed_qa_file = "/kaggle/input/mediqa-data/closedquestions_definitions_imageclef2025.json"
    output_dir = "/kaggle/working/output/"
    run_inference(data_dir, query_file, closed_qa_file, output_dir)