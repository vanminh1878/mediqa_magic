import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from qa.utils import MediQAQADataset, load_closed_questions
import torchvision.transforms as transforms
from tqdm import tqdm
import timm

class ClosedQAModel(nn.Module):
    def __init__(self):
        super(ClosedQAModel, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.bert = AutoModelForCausalLM.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768 + 1000, 9)  # 9 options for CQID011-001
    
    def forward(self, images, queries):
        img_features = self.vit(images)
        # BERT expects tokenized input, so we need to process queries
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
    
    for epoch in range(5):  # Sửa lỗi cú pháp
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