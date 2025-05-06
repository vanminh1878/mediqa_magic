import json
import os
import torch
from torch import nn
import open_clip
from transformers import DistilBertTokenizer, DistilBertModel
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

# Cấu hình
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_DIR = "/kaggle/input/mediqa-data/mediqa-data/images"
TRAIN_FILE = "/kaggle/input/mediqa-data/mediqa-data/train_cvqa.json"
VALID_FILE = "/kaggle/input/mediqa-data/mediqa-data/valid.json"
TEST_FILE = "/kaggle/input/mediqa-data/mediqa-data/test.json"
QUESTION_FILE = "/kaggle/input/mediqa-data/mediqa-data/closedquestions_definitions_imageclef2025.json"
OUTPUT_DIR = "/kaggle/working"
BATCH_SIZE = 32  # Phù hợp với T4
LEARNING_RATE = 1e-5
EPOCHS = 3
WEIGHT_IMAGE = 0.3  # Trọng số nhúng hình ảnh
WEIGHT_TEXT = 0.7   # Trọng số nhúng văn bản (ưu tiên văn bản)

# Kiểm tra file tồn tại
for f in [QUESTION_FILE, TRAIN_FILE, VALID_FILE, TEST_FILE]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"File {f} not found")

# Tải mô hình CLIP và DistilBERT
try:
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    clip_model = clip_model.to(DEVICE).train()
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(DEVICE).train()
except Exception as e:
    raise RuntimeError(f"Failed to load models: {e}")

# Mô hình kết hợp CLIP và DistilBERT
class ClipBertModel(nn.Module):
    def __init__(self, clip_model, bert_model, num_options):
        super().__init__()
        self.clip_model = clip_model
        self.bert_model = bert_model
        self.fc = nn.Linear(512 + 768, num_options)  # CLIP: 512, DistilBERT: 768

    def forward(self, images, query_tokens, option_tokens):
        with torch.no_grad(), autocast():
            # Nhúng hình ảnh bằng CLIP
            image_embeds = self.clip_model.encode_image(images).float()
            # Nhúng văn bản truy vấn bằng DistilBERT
            query_embeds = self.bert_model(**query_tokens).last_hidden_state[:, 0, :].float()
            # Nhúng tùy chọn bằng DistilBERT
            option_embeds = self.bert_model(**option_tokens).last_hidden_state[:, 0, :].float()
        
        # Kết hợp nhúng
        combined_embed = WEIGHT_IMAGE * image_embeds + WEIGHT_TEXT * query_embeds
        logits = self.fc(torch.cat([combined_embed, option_embeds], dim=-1))
        return logits

# Dataset cho tinh chỉnh
class MediqaDataset(Dataset):
    def __init__(self, data_file, question_file, image_dir):
        with open(data_file, "r") as f:
            self.data = json.load(f)
        with open(question_file, "r") as f:
            self.questions = json.load(f)
        self.image_dir = image_dir
        self.question_dict = {q["qid"]: q for q in self.questions}
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        case = self.data[idx]
        encounter_id = case["encounter_id"]
        image_ids = case["image_ids"]
        query_content = case.get("query_content_en", "")
        image_paths = [os.path.join(self.image_dir, img_id) for img_id in image_ids]
        
        # Tải hình ảnh
        images = []
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(preprocess(img))
            except:
                continue
        if not images:
            images = [torch.zeros(3, 224, 224)]
        images = torch.stack(images).to(DEVICE)
        
        # Token hóa văn bản truy vấn
        query_tokens = self.tokenizer(query_content, return_tensors="pt", padding=True, truncation=True, max_length=128)
        query_tokens = {k: v.to(DEVICE) for k, v in query_tokens.items()}
        
        # Chuẩn bị nhãn và tùy chọn
        labels = {}
        option_texts = {}
        for qid, q_data in self.question_dict.items():
            options = q_data["options_en"]
            option_tokens = self.tokenizer(options, return_tensors="pt", padding=True, truncation=True, max_length=32)
            option_tokens = {k: v.to(DEVICE) for k, v in option_tokens.items()}
            option_texts[qid] = option_tokens
            # Nhãn từ train_cvqa.json (nếu có)
            label = case.get("answers", {}).get(qid, len(options) - 1)  # Mặc định "Not mentioned"
            labels[qid] = torch.tensor(label, dtype=torch.long).to(DEVICE)
        
        return {
            "images": images,
            "query_tokens": query_tokens,
            "option_texts": option_texts,
            "labels": labels,
            "encounter_id": encounter_id
        }

# Tinh chỉnh mô hình
def fine_tune_model():
    dataset = MediqaDataset(TRAIN_FILE, QUESTION_FILE, IMAGE_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = ClipBertModel(clip_model, bert_model, num_options=10).to(DEVICE)  # Giả sử max 10 tùy chọn
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            images = batch["images"]
            query_tokens = batch["query_tokens"]
            option_texts = batch["option_texts"]
            labels = batch["labels"]
            
            optimizer.zero_grad()
            with autocast():
                loss = 0
                for qid in option_texts:
                    logits = model(images, query_tokens, option_texts[qid])
                    loss += nn.CrossEntropyLoss()(logits, labels[qid])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")
    
    return model

# Hàm trả lời câu hỏi bằng CLIP + DistilBERT
def answer_with_clip_bert(model, image_paths, query_content, question, question_type, options, tokenizer):
    try:
        # Tải hình ảnh
        images = []
        valid_paths = []
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(preprocess(img))
                valid_paths.append(path)
            except:
                continue
        if not images:
            return options[-1], len(options) - 1
        images = torch.stack(images).to(DEVICE)
        
        # Token hóa văn bản
        query_tokens = tokenizer(query_content, return_tensors="pt", padding=True, truncation=True, max_length=128)
        query_tokens = {k: v.to(DEVICE) for k, v in query_tokens.items()}
        option_tokens = tokenizer(options, return_tensors="pt", padding=True, truncation=True, max_length=32)
        option_tokens = {k: v.to(DEVICE) for k, v in option_tokens.items()}
        
        # Dự đoán
        model.eval()
        with torch.no_grad(), autocast():
            logits = model(images, query_tokens, option_tokens)
            best_option_idx = torch.argmax(logits, dim=-1).item()
        
        return options[best_option_idx], best_option_idx
    except Exception as e:
        print(f"Model error for question {question}: {e}")
        return options[-1], len(options) - 1

# Hàm xử lý tập dữ liệu
def process_dataset(data_file, output_filename, question_file, model, tokenizer):
    try:
        with open(data_file, "r") as f:
            data = json.load(f)
        with open(question_file, "r") as f:
            questions = json.load(f)
    except Exception as e:
        raise FileNotFoundError(f"Error loading file: {e}")
    
    question_dict = {q["qid"]: q for q in questions}
    results = []
    
    for case in tqdm(data, desc=f"Processing {output_filename}"):
        encounter_id = case["encounter_id"]
        image_ids = case["image_ids"]
        image_paths = [os.path.join(IMAGE_DIR, img_id) for img_id in image_ids]
        query_content = case.get("query_content_en", "")
        
        result = {"encounter_id": encounter_id}
        
        for qid, q_data in question_dict.items():
            question = q_data["question_en"]
            question_type = q_data["question_type_en"]
            options = q_data["options_en"]
            
            # Dự đoán bằng CLIP + DistilBERT
            _, clip_idx = answer_with_clip_bert(model, image_paths, query_content, question, question_type, options, tokenizer)
            result[qid] = clip_idx
        
        results.append(result)
    
    # Lưu kết quả
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    try:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Error saving results to {output_path}: {e}")
        raise
    
    return results, data, question_dict

# Hàm đánh giá thủ công
def manual_evaluation(results, data, question_dict, num_samples=5):
    data_dict = {case["encounter_id"]: case for case in data}
    sample_ids = np.random.choice([r["encounter_id"] for r in results], min(num_samples, len(results)), replace=False)
    
    print("\nManual Evaluation (Random Samples):")
    for enc_id in sample_ids:
        result = next(r for r in results if r["encounter_id"] == enc_id)
        case = data_dict[enc_id]
        query_content = case.get("query_content_en", "")
        responses = case.get("responses", [])
        
        print(f"\nEncounter ID: {enc_id}")
        print(f"Query: {query_content}")
        print("Responses:", [r["content_en"] for r in responses])
        
        correct_count = 0
        total_questions = 0
        for qid, idx in result.items():
            if qid == "encounter_id":
                continue
            q_data = question_dict[qid]
            question = q_data["question_en"]
            options = q_data["options_en"]
            pred_option = options[idx]
            
            # Kiểm tra tính hợp lý
            is_correct = "Unknown"
            query_lower = query_content.lower()
            if q_data["question_type_en"] == "Onset" and any(k in query_lower for k in ["month", "year", "since", "ago"]):
                if idx in [3, 4, 5]:  # within months, within years, multiple years
                    is_correct = "Likely Correct"
                    correct_count += 1
                else:
                    is_correct = "Possibly Incorrect"
            elif q_data["question_type_en"] == "Itch" and "itch" in query_lower and idx == 0:
                is_correct = "Likely Correct"
                correct_count += 1
            elif q_data["question_type_en"] == "Itch" and "not itch" in query_lower and idx == 1:
                is_correct = "Likely Correct"
                correct_count += 1
            elif q_data["question_type_en"] == "Site Location" and any(k in query_lower for k in ["neck", "lip", "thigh", "chest"]):
                if ("neck" in query_lower and idx == 1) or ("lip" in query_lower and idx == 0) or ("thigh" in query_lower and idx == 3) or ("chest" in query_lower and idx == 4):
                    is_correct = "Likely Correct"
                    correct_count += 1
                else:
                    is_correct = "Possibly Incorrect"
            elif q_data["question_type_en"] == "Lesion Color" and "red" in query_lower and idx == 2:
                is_correct = "Likely Correct"
                correct_count += 1
            elif q_data["question_type_en"] == "Lesion Count" and "papule" in query_lower and idx == 0:
                is_correct = "Likely Correct"
                correct_count += 1
            elif idx == len(options) - 1 and not any(k in query_lower for k in ["month", "year", "itch", "neck", "lip", "thigh", "chest", "red", "papule"]):
                is_correct = "Likely Correct"
                correct_count += 1
            else:
                is_correct = "Possibly Incorrect"
            
            print(f"Q: {question} -> Predicted: {pred_option} ({is_correct})")
            total_questions += 1
        
        print(f"Estimated Accuracy for {enc_id}: {correct_count / total_questions:.2f}")

# Hàm kiểm tra định dạng JSON
def validate_output(output_file, question_file):
    try:
        with open(output_file, "r") as f:
            results = json.load(f)
        with open(question_file, "r") as f:
            questions = json.load(f)
        
        qids = set(q["qid"] for q in questions)
        for res in results:
            assert "encounter_id" in res, "Missing encounter_id"
            for qid in qids:
                assert qid in res, f"Missing {qid}"
                assert isinstance(res[qid], int), f"{qid} not integer"
                assert 0 <= res[qid] < len(questions[0]["options_en"]), f"Invalid index for {qid}"
        print(f"Output {output_file} is valid")
    except Exception as e:
        print(f"Validation error for {output_file}: {e}")

# Hàm chính
def main():
    try:
        # Tinh chỉnh mô hình
        print("Fine-tuning CLIP + DistilBERT...")
        model = fine_tune_model()
        
        # Xử lý tập valid
        valid_results, valid_data, question_dict = process_dataset(VALID_FILE, "closed_qa_valid_results.json", QUESTION_FILE, model, tokenizer)
        validate_output(os.path.join(OUTPUT_DIR, "closed_qa_valid_results.json"), QUESTION_FILE)
        
        # Xử lý tập test
        test_results, test_data, _ = process_dataset(TEST_FILE, "closed_qa_test_results.json", QUESTION_FILE, model, tokenizer)
        validate_output(os.path.join(OUTPUT_DIR, "closed_qa_test_results.json"), QUESTION_FILE)
        
        # Đánh giá thủ công
        print("\nEvaluating valid set samples:")
        manual_evaluation(valid_results, valid_data, question_dict, num_samples=5)
        
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()