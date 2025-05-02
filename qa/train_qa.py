import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import timm

# Tích hợp MediQAQADataset trực tiếp
class MediQAQADataset(Dataset):
    def __init__(self, query_file, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_dir = os.path.join(data_dir, 'images')
        self.mask_dir = os.path.join(data_dir, f'masks_{split}')
        
        with open(query_file, 'r') as f:
            self.data = json.load(f)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        encounter_id = item['encounter_id']
        query = f"{item['query_title_en']} {item['query_content_en']}"
        
        # Chọn ảnh đầu tiên
        img_id = item['image_ids'][0]
        img_path = os.path.join(self.image_dir, img_id)
        image = Image.open(img_path).convert('RGB')
        
        # Load mask nếu có
        mask_path = os.path.join(self.mask_dir, img_id.replace('.jpg', '_mask.png'))
        mask = Image.open(mask_path).convert('L') if os.path.exists(mask_path) else None
        
        if self.transform:
            image = self.transform(image)
        
        # Câu hỏi mở
        response = item['responses'][0]['content_en'] if 'responses' in item else ""
        
        # Câu hỏi đóng (giả sử ánh xạ từ query/response)
        closed_qa = {}
        if 'query_content_en' in item:
            if 'thigh' in item['query_content_en'].lower():
                closed_qa['CQID011-001'] = 3  # lower extremities
            elif 'palm' in item['query_content_en'].lower():
                closed_qa['CQID011-001'] = 7  # other (please specify)
            else:
                closed_qa['CQID011-001'] = 8  # Not mentioned
        
        return {
            'image': image,
            'query': query,
            'response': response,
            'closed_qa': closed_qa,
            'encounter_id': encounter_id
        }

# Tích hợp load_closed_questions trực tiếp
def load_closed_questions(closed_qa_file):
    with open(closed_qa_file, 'r') as f:
        return json.load(f)

class ClosedQAModel(nn.Module):
    def __integrate__(self):
        super(ClosedQAModel, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.bert = AutoModelForCausalLM.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768 + 1000, 9)  # 9 options for CQID011-001
    
    def forward(self, images, queries):
        img_features = self.vit(images)
        text_features = self.bert(input_ids=queries['input_ids'], attention_mask=queries['attention_mask']).logits[:, 0, :]
        combined = torch.cat([img_features, text_features], dim=1)
        return self.fc(combined)

def train_qa():
    data_dir = "/kaggle/input/mediqa-data/mediqa-data/"
    query_file = "/kaggle/input/mediqa-data/mediqa-data/train_cvqa.json"
    synthetic_file = "/kaggle/working/synthetic_train.json"
    
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = MediQAQADataset(query_file, data_dir, split='train', transform=transform)
    synthetic_dataset = MediQAQADataset(synthetic_file, data_dir, split='train', transform=transform)
    dataloader = DataLoader(dataset + synthetic_dataset, batch_size=8, shuffle=True)
    
    # QA mở: Fine-tune LLaVA-Med
    model = AutoModelForCausalLM.from_pretrained('llava-hf/llava-7b-hf').cuda()
    tokenizer = AutoTokenizer.from_pretrained('llava-hf/llava-7b-hf')
    
    # QA đóng
    closed_model = ClosedQAModel().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(closed_model.parameters()), lr=1e-4)
    
    for epoch in range(5):
        model.train()
        closed_model.train()
        for batch in tqdm(dataloader):
            images = batch['image'].cuda()
            queries = batch['query']
            responses = batch['response']
            closed_qa_labels = batch['closed_qa']['CQID011-001'].cuda()
            
            # QA mở
            inputs = tokenizer([q + r for q, r in zip(queries, responses)], return_tensors='pt', padding=True, truncation=True).to('cuda')
            outputs = model(**inputs)
            loss_open = outputs.loss if outputs.loss is not None else torch.tensor(0.0).cuda()
            
            # QA đóng
            tokenized_queries = tokenizer(queries, return_tensors='pt', padding=True, truncation=True).to('cuda')
            closed_outputs = closed_model(images, tokenized_queries)
            loss_closed = criterion(closed_outputs, closed_qa_labels.long())
            
            loss = loss_open + loss_closed
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    
    torch.save(model.state_dict(), "/kaggle/working/llava_med.pth")
    torch.save(closed_model.state_dict(), "/kaggle/working/closed_qa.pth")

if __name__ == "__main__":
    train_qa()