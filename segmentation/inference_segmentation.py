import json
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from qa.utils import MediQAQADataset
from qa.train_qa import ClosedQAModel
import torchvision.transforms as transforms
from tqdm import tqdm

def inference_qa(split='valid'):
    data_dir = "/kaggle/input/mediqa-data/mediqa-data/"
    query_file = f"/kaggle/input/mediqa-data/mediqa-data/{split}.json"
    output_dir = f"/kaggle/working/{split}_results"
    os.makedirs(output_dir, exist_ok=True)
    
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = MediQAQADataset(query_file, data_dir, split=split, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)  # Thêm DataLoader
    
    # QA mở
    model = AutoModelForCausalLM.from_pretrained('llava-hf/llava-7b-hf').cuda()
    model.load_state_dict(torch.load("/kaggle/working/llava_med.pth"))
    tokenizer = AutoTokenizer.from_pretrained('llava-hf/llava-7b-hf')
    
    # QA đóng
    closed_model = ClosedQAModel().cuda()
    closed_model.load_state_dict(torch.load("/kaggle/working/closed_qa.pth"))
    
    results_open = []
    results_closed = []
    
    model.eval()
    closed_model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            encounter_id = batch['encounter_id'][0]
            images = batch['image'].cuda()
            queries = batch['query']
            
            # QA mở
            inputs = tokenizer(queries, return_tensors='pt', padding=True, truncation=True).to('cuda')
            outputs = model.generate(**inputs, max_length=200)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # QA đóng
            tokenized_queries = tokenizer(queries, return_tensors='pt', padding=True, truncation=True).to('cuda')
            closed_outputs = closed_model(images, tokenized_queries)
            closed_pred = torch.argmax(closed_outputs, dim=1).item()
            
            results_open.append({
                'encounter_id': encounter_id,
                'response': response
            })
            results_closed.append({
                'encounter_id': encounter_id,
                'CQID011-001': closed_pred
            })
    
    with open(os.path.join(output_dir, 'open_qa.json'), 'w') as f:
        json.dump(results_open, f, indent=2)
    with open(os.path.join(output_dir, 'closed_qa.json'), 'w') as f:
        json.dump(results_closed, f, indent=2)
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    for split in ['valid', 'test']:
        inference_qa(split)