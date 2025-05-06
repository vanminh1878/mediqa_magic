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
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp

# Đặt phương thức khởi tạo tiến trình
mp.set_start_method('spawn', force=True)

# Cấu hình
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_DIR = "/kaggle/input/mediqa-data/mediqa-data/images"
TRAIN_FILE = "/kaggle/input/mediqa-data/mediqa-data/train_cvqa.json"
VALID_FILE = "/kaggle/input/mediqa-data/mediqa-data/valid.json"
TEST_FILE = "/kaggle/input/mediqa-data/mediqa-data/test.json"
QUESTION_FILE = "/kaggle/input/mediqa-data/mediqa-data/closedquestions_definitions_imageclef2025.json"
OUTPUT_DIR = "/kaggle/working"
BATCH_SIZE = 16  # Tăng để huấn luyện nhanh hơn
LEARNING_RATE = 1e-5
EPOCHS = 5  # Tăng để học tốt hơn
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "model_checkpoint.pt")
MAX_IMAGES = 5
MAX_LENGTH = 128
PATIENCE = 2  # Early stopping

# Xóa cache CUDA và bật expandable_segments
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TF_LOGGING"] = "ERROR"

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

# Hàm trích xuất nhãn
def extract_labels(query_content, responses, question_dict):
    query_lower = query_content.lower()
    response_text = " ".join([r["content_en"].lower() for r in responses]).lower()
    text = query_lower + " " + response_text
    labels = {}
    
    for qid, q_data in question_dict.items():
        question_type = q_data["question_type_en"]
        options = q_data["options_en"]
        label = len(options) - 1  # Mặc định là Not mentioned
        
        if question_type == "Onset":
            if any(k in text for k in ["hour"]):
                label = 0
            elif any(k in text for k in ["day"]):
                label = 1
            elif any(k in text for k in ["week"]):
                label = 2
            elif any(k in text for k in ["month", "six months"]):
                label = 3
            elif any(k in text for k in ["year"]):
                label = 4
            elif any(k in text for k in ["ten years", "multiple years"]):
                label = 5
        elif question_type == "Itch":
            if any(k in text for k in ["itch", "pruritus", "scratch", "irritat"]):
                label = 0
            elif any(k in text for k in ["not itch", "no itch"]):
                label = 1
        elif question_type == "Site":
            if any(k in text for k in ["systemic", "widespread", "whole body"]):
                label = 2
            elif any(k in text for k in ["area", "spot", "region"]):
                label = 1
            elif any(k in text for k in ["single"]):
                label = 0
        elif question_type == "Site Location":
            locations = []
            location_map = {
                "head": 0, "neck": 1, "upper extremities": 2, "lower extremities": 3, "chest/abdomen": 4, "back": 5,
                "hand": 2, "foot": 3, "thigh": 3, "palm": 2, "lip": 0, "finger": 2, "thumb": 2, "arm": 2, "leg": 3,
                "chest": 4, "abdomen": 4, "eyebrow": 0, "sole": 3
            }
            for loc, idx in location_map.items():
                if loc in text and idx not in locations:
                    locations.append(idx)
            if qid.startswith("CQID011-"):
                q_num = int(qid.split("-")[1])
                label = locations[q_num - 1] if q_num <= len(locations) else len(options) - 1
        elif question_type == "Lesion Color":
            if any(k in text for k in ["red", "pink", "erythema"]):
                label = 2
            elif any(k in text for k in ["brown"]):
                label = 3
            elif any(k in text for k in ["blue"]):
                label = 4
            elif any(k in text for k in ["purple", "purplish"]):
                label = 5
            elif any(k in text for k in ["black"]):
                label = 6
            elif any(k in text for k in ["white", "pale", "leukoplakia"]):
                label = 7
            elif any(k in text for k in ["combination"]):
                label = 8
            elif any(k in text for k in ["hyperpigmentation"]):
                label = 9
            elif any(k in text for k in ["hypopigmentation"]):
                label = 10
        elif question_type == "Lesion Count":
            if any(k in text for k in ["rash", "rashes", "lesions", "spots", "multiple"]):
                label = 1
            elif any(k in text for k in ["single", "one"]):
                label = 0
        elif question_type == "Size":
            if any(k in text for k in ["large", "bigger"]):
                label = 2
            elif any(k in text for k in ["small", "nail"]):
                label = 0
            elif any(k in text for k in ["palm"]):
                label = 1
        elif question_type == "Skin Description":
            if any(k in text for k in ["bump", "raised", "swollen"]):
                label = 0
            elif any(k in text for k in ["flat", "macula"]):
                label = 1
            elif any(k in text for k in ["sunken", "atrophy", "loss"]):
                label = 2
            elif any(k in text for k in ["thick", "keratosis"]):
                label = 3
            elif any(k in text for k in ["thin"]):
                label = 4
            elif any(k in text for k in ["wart"]):
                label = 5
            elif any(k in text for k in ["crust", "peeling"]):
                label = 6
            elif any(k in text for k in ["scab"]):
                label = 7
            elif any(k in text for k in ["weeping", "exudation"]):
                label = 8
        elif question_type == "Texture":
            if any(k in text for k in ["smooth", "no scales"]):
                label = 0
            elif any(k in text for k in ["rough", "scaly", "scales", "peeling"]):
                label = 1
        
        labels[qid] = label
    
    return labels

# Hàm kiểm tra "Not mentioned"
def check_not_mentioned(query_content, question_type):
    query_lower = query_content.lower()
    if question_type == "Onset":
        return not any(k in query_lower for k in ["hour", "day", "week", "month", "year", "since", "ago", "ten years"])
    if question_type == "Itch":
        return not any(k in query_lower for k in ["itch", "scratch", "irritat", "pruritus", "not itch", "no itch"])
    if question_type == "Site":
        return not any(k in query_lower for k in ["area", "spot", "region", "widespread", "systemic", "single"])
    if question_type == "Site Location":
        return not any(k in query_lower for k in ["head", "neck", "arm", "leg", "chest", "abdomen", "back", "hand", "foot", "thigh", "palm", "lip", "finger", "eyebrow", "sole"])
    if question_type == "Size":
        return not any(k in query_lower for k in ["size", "large", "small", "thumb", "palm", "bigger", "nail"])
    if question_type == "Skin Description":
        return not any(k in query_lower for k in ["bump", "flat", "sunken", "thick", "thin", "wart", "crust", "scab", "weep", "lesion", "raised", "swollen", "atrophy", "keratosis", "peeling", "exudation"])
    if question_type == "Lesion Color":
        return not any(k in query_lower for k in ["color", "red", "pink", "brown", "blue", "purple", "black", "white", "pigment", "pale", "erythema", "purplish", "leukoplakia"])
    if question_type == "Lesion Count":
        return not any(k in query_lower for k in ["single", "multiple", "many", "few", "rash", "rashes", "lesions", "spots"])
    if question_type == "Texture":
        return not any(k in query_lower for k in ["smooth", "rough", "texture", "scales", "scaly", "peeling"])
    return True

# Mô hình kết hợp CLIP và DistilBERT
class ClipBertModel(nn.Module):
    def __init__(self, clip_model, bert_model, max_options=12):
        super().__init__()
        self.clip_model = clip_model
        self.bert_model = bert_model
        self.image_projection = nn.Linear(512, 768)
        self.dropout = nn.Dropout(0.3)  # Thêm dropout
        self.fc = nn.Linear(768 + 768, max_options)

    def forward(self, image_embeds, query_tokens, option_tokens, num_options):
        with autocast(device_type=DEVICE_TYPE):
            # Nhúng query
            query_embeds = self.bert_model(**query_tokens).last_hidden_state[:, 0, :].float()
            
            # Biến đổi image_embeds
            image_embeds = self.image_projection(image_embeds).float()
            
            # Nhúng option_tokens
            batch_size = image_embeds.size(0)
            option_input_ids = option_tokens["input_ids"].reshape(batch_size * num_options, -1)
            option_attention_mask = option_tokens["attention_mask"].reshape(batch_size * num_options, -1)
            option_embeds = self.bert_model(
                input_ids=option_input_ids,
                attention_mask=option_attention_mask
            ).last_hidden_state[:, 0, :].float()
            
            # Reshape option_embeds
            option_embeds = option_embeds.reshape(batch_size, num_options, -1).mean(dim=1)
        
        # Kết hợp embedding
        combined_embed = image_embeds + query_embeds
        combined_embed = self.dropout(combined_embed)
        logits = self.fc(torch.cat([combined_embed, option_embeds], dim=-1))
        return logits[:, :num_options]  # Cắt logits theo num_options thực tế

# Dataset cho tinh chỉnh
class MediqaDataset(Dataset):
    def __init__(self, data_file, question_file, image_dir, clip_model):
        with open(data_file, "r") as f:
            self.data = json.load(f)
        with open(question_file, "r") as f:
            self.questions = json.load(f)
        self.image_dir = image_dir
        self.question_dict = {q["qid"]: q for q in self.questions}
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.clip_model = clip_model

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        case = self.data[idx]
        encounter_id = case["encounter_id"]
        image_ids = case["image_ids"]
        query_content = case.get("query_content_en", "")
        responses = case.get("responses", [])
        image_paths = [os.path.join(self.image_dir, img_id) for img_id in image_ids]
        
        image_embeds = []
        for path in image_paths[:MAX_IMAGES]:
            try:
                img = Image.open(path).convert("RGB")
                img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad(), autocast(device_type=DEVICE_TYPE):
                    embed = self.clip_model.encode_image(img_tensor).float()
                image_embeds.append(embed)
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                continue
        if not image_embeds:
            image_embeds = [torch.zeros(1, 512).to(DEVICE)]
        image_embed = torch.mean(torch.cat(image_embeds, dim=0), dim=0, keepdim=True)
        
        query_tokens = self.tokenizer(
            query_content,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )
        query_tokens = {k: v.squeeze(0).to(DEVICE) for k, v in query_tokens.items()}
        
        labels = extract_labels(query_content, responses, self.question_dict)
        
        option_texts = {}
        for qid, q_data in self.question_dict.items():
            options = q_data["options_en"]
            option_tokens = self.tokenizer(
                options,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=32
            )
            option_tokens = {k: v.to(DEVICE) for k, v in option_tokens.items()}
            option_texts[qid] = option_tokens
            labels[qid] = torch.tensor(labels[qid], dtype=torch.long).to(DEVICE)
        
        return {
            "image_embeds": image_embed,
            "query_tokens": query_tokens,
            "option_texts": option_texts,
            "labels": labels,
            "encounter_id": encounter_id
        }

# Hàm collate_fn để chuẩn hóa batch
def collate_fn(batch):
    image_embeds = torch.cat([item["image_embeds"] for item in batch], dim=0)
    
    query_input_ids = torch.stack([item["query_tokens"]["input_ids"] for item in batch])
    query_attention_mask = torch.stack([item["query_tokens"]["attention_mask"] for item in batch])
    query_tokens = {
        "input_ids": query_input_ids,
        "attention_mask": query_attention_mask
    }
    
    option_texts = {}
    for qid in batch[0]["option_texts"]:
        option_input_ids = torch.stack([item["option_texts"][qid]["input_ids"] for item in batch])
        option_attention_mask = torch.stack([item["option_texts"][qid]["attention_mask"] for item in batch])
        option_texts[qid] = {
            "input_ids": option_input_ids,
            "attention_mask": option_attention_mask
        }
    
    labels = {}
    for qid in batch[0]["labels"]:
        labels[qid] = torch.stack([item["labels"][qid] for item in batch])
    
    encounter_ids = [item["encounter_id"] for item in batch]
    
    return {
        "image_embeds": image_embeds,
        "query_tokens": query_tokens,
        "option_texts": option_texts,
        "labels": labels,
        "encounter_id": encounter_ids
    }

# Tinh chỉnh mô hình
def fine_tune_model():
    dataset = MediqaDataset(TRAIN_FILE, QUESTION_FILE, IMAGE_DIR, clip_model)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)
    model = ClipBertModel(clip_model, bert_model, max_options=12).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda')
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            image_embeds = batch["image_embeds"]
            query_tokens = batch["query_tokens"]
            option_texts = batch["option_texts"]
            labels = batch["labels"]
            
            optimizer.zero_grad()
            with autocast(device_type=DEVICE_TYPE):
                loss = 0
                for qid in option_texts:
                    num_options = option_texts[qid]["input_ids"].size(1)
                    logits = model(image_embeds, query_tokens, option_texts[qid], num_options)
                    loss += nn.CrossEntropyLoss()(logits, labels[qid])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered")
                break
    
    # Tải mô hình tốt nhất
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    return model

# Hàm trả lời câu hỏi
def answer_with_clip_bert(model, image_paths, query_content, question, question_type, options, tokenizer, clip_model):
    try:
        if check_not_mentioned(query_content, question_type):
            return options[-1], len(options) - 1
        
        image_embeds = []
        for path in image_paths[:MAX_IMAGES]:
            try:
                img = Image.open(path).convert("RGB")
                img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad(), autocast(device_type=DEVICE_TYPE):
                    embed = clip_model.encode_image(img_tensor).float()
                image_embeds.append(embed)
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                continue
        if not image_embeds:
            image_embeds = [torch.zeros(1, 512).to(DEVICE)]
        image_embed = torch.mean(torch.cat(image_embeds, dim=0), dim=0, keepdim=True)
        
        query_tokens = tokenizer(
            query_content,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )
        query_tokens = {k: v.to(DEVICE) for k, v in query_tokens.items()}
        
        option_tokens = tokenizer(
            options,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=32
        )
        option_tokens = {k: v.to(DEVICE) for k, v in option_tokens.items()}
        
        num_options = len(options)
        
        # Tinh chỉnh trọng số dựa trên loại câu hỏi
        weight_image = 0.4
        weight_text = 0.6
        if question_type in ["Onset", "Itch", "Lesion Count"]:
            weight_image, weight_text = 0.1, 0.9
        elif question_type in ["Site Location", "Lesion Color", "Skin Description"]:
            weight_image, weight_text = 0.5, 0.5
        
        model.eval()
        with torch.no_grad(), autocast(device_type=DEVICE_TYPE):
            logits = model(image_embed, query_tokens, option_tokens, num_options)
            logits = weight_image * logits + weight_text * logits
            probs = torch.softmax(logits, dim=1)
            best_option_idx = torch.argmax(probs, dim=1).item()
            best_option_idx = min(best_option_idx, num_options - 1)  # Giới hạn chỉ số
        
        return options[best_option_idx], best_option_idx
    except Exception as e:
        print(f"Model error for question {question}: {e}")
        return options[-1], len(options) - 1

# Hàm xử lý tập dữ liệu
def process_dataset(data_file, output_filename, question_file, model, tokenizer, clip_model):
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
        
        locations = []
        if any(k in query_content.lower() for k in ["systemic", "widespread", "whole body"]):
            locations = [0, 1, 2, 3, 4, 5]
        else:
            location_map = {
                "head": 0, "neck": 1, "upper extremities": 2, "lower extremities": 3, "chest/abdomen": 4, "back": 5,
                "hand": 2, "foot": 3, "thigh": 3, "palm": 2, "lip": 0, "finger": 2, "eyebrow": 0, "sole": 3
            }
            for loc, idx in location_map.items():
                if loc in query_content.lower() and idx not in locations:
                    locations.append(idx)
        
        for qid, q_data in question_dict.items():
            question = q_data["question_en"]
            question_type = q_data["question_type_en"]
            options = q_data["options_en"]
            
            if question_type == "Site Location" and locations:
                q_num = int(qid.split("-")[1])
                if q_num <= len(locations):
                    result[qid] = locations[q_num - 1]
                    continue
                else:
                    result[qid] = len(options) - 1
            else:
                _, clip_idx = answer_with_clip_bert(model, image_paths, query_content, question, question_type, options, tokenizer, clip_model)
                result[qid] = clip_idx
        
        results.append(result)
    
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
            question_type = q_data["question_type_en"]
            options = q_data["options_en"]
            pred_option = options[idx]
            
            is_correct = "Unknown"
            query_lower = query_content.lower()
            response_lower = " ".join([r["content_en"].lower() for r in responses])
            text = query_lower + " " + response_lower
            
            matched_keywords = []
            
            if question_type == "Onset":
                keywords = ["hour", "day", "week", "month", "year", "since", "ago", "ten years"]
                matched_keywords = [k for k in keywords if k in text]
                if any(k in text for k in ["month", "year", "ten years"]) and idx in [3, 4, 5]:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif check_not_mentioned(query_content, question_type) and idx == len(options) - 1:
                    is_correct = "Likely Correct"
                    correct_count += 1
                else:
                    is_correct = "Possibly Incorrect"
            elif question_type == "Itch":
                keywords = ["itch", "pruritus", "scratch", "irritat", "not itch", "no itch"]
                matched_keywords = [k for k in keywords if k in text]
                if any(k in text for k in ["itch", "pruritus", "scratch"]) and idx == 0:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif any(k in text for k in ["not itch", "no itch"]) and idx == 1:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif check_not_mentioned(query_content, question_type) and idx == len(options) - 1:
                    is_correct = "Likely Correct"
                    correct_count += 1
                else:
                    is_correct = "Possibly Incorrect"
            elif question_type == "Site":
                keywords = ["systemic", "widespread", "area", "spot", "region", "single"]
                matched_keywords = [k for k in keywords if k in text]
                if any(k in text for k in ["systemic", "widespread"]) and idx == 2:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif any(k in text for k in ["area", "spot"]) and idx == 1:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif any(k in text for k in ["single"]) and idx == 0:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif check_not_mentioned(query_content, question_type) and idx == len(options) - 1:
                    is_correct = "Likely Correct"
                    correct_count += 1
                else:
                    is_correct = "Possibly Incorrect"
            elif question_type == "Site Location":
                locations = []
                location_map = {
                    "head": 0, "neck": 1, "upper extremities": 2, "lower extremities": 3, "chest/abdomen": 4, "back": 5,
                    "hand": 2, "foot": 3, "thigh": 3, "palm": 2, "lip": 0, "finger": 2, "eyebrow": 0, "sole": 3
                }
                matched_keywords = [loc for loc in location_map if loc in text]
                for loc, loc_idx in location_map.items():
                    if loc in text and loc_idx not in locations:
                        locations.append(loc_idx)
                q_num = int(qid.split("-")[1])
                if q_num <= len(locations) and idx == locations[q_num - 1]:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif q_num > len(locations) and idx == len(options) - 1:
                    is_correct = "Likely Correct"
                    correct_count += 1
                else:
                    is_correct = "Possibly Incorrect"
            elif question_type == "Lesion Color":
                keywords = ["red", "pink", "brown", "blue", "purple", "black", "white", "pale", "erythema", "purplish", "leukoplakia"]
                matched_keywords = [k for k in keywords if k in text]
                if any(k in text for k in ["red", "pink", "erythema"]) and idx == 2:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif any(k in text for k in ["brown"]) and idx == 3:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif any(k in text for k in ["purple", "purplish"]) and idx == 5:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif any(k in text for k in ["white", "pale", "leukoplakia"]) and idx == 7:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif check_not_mentioned(query_content, question_type) and idx == len(options) - 1:
                    is_correct = "Likely Correct"
                    correct_count += 1
                else:
                    is_correct = "Possibly Incorrect"
            elif question_type == "Lesion Count":
                keywords = ["rash", "rashes", "lesions", "spots", "multiple", "single"]
                matched_keywords = [k for k in keywords if k in text]
                if any(k in text for k in ["rash", "rashes", "lesions", "spots", "multiple"]) and idx == 1:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif any(k in text for k in ["single"]) and idx == 0:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif check_not_mentioned(query_content, question_type) and idx == len(options) - 1:
                    is_correct = "Likely Correct"
                    correct_count += 1
                else:
                    is_correct = "Possibly Incorrect"
            elif question_type == "Size":
                keywords = ["large", "small", "thumb", "palm", "bigger", "nail"]
                matched_keywords = [k for k in keywords if k in text]
                if any(k in text for k in ["large", "bigger"]) and idx == 2:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif any(k in text for k in ["small", "nail"]) and idx == 0:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif any(k in text for k in ["palm"]) and idx == 1:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif check_not_mentioned(query_content, question_type) and idx == len(options) - 1:
                    is_correct = "Likely Correct"
                    correct_count += 1
                else:
                    is_correct = "Possibly Incorrect"
            elif question_type == "Skin Description":
                keywords = ["bump", "flat", "sunken", "thick", "thin", "wart", "crust", "scab", "weeping", "raised", "swollen", "atrophy", "keratosis", "peeling", "exudation"]
                matched_keywords = [k for k in keywords if k in text]
                if any(k in text for k in ["bump", "raised", "swollen"]) and idx == 0:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif any(k in text for k in ["flat", "macula"]) and idx == 1:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif any(k in text for k in ["sunken", "atrophy"]) and idx == 2:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif any(k in text for k in ["thick", "keratosis"]) and idx == 3:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif any(k in text for k in ["weeping", "exudation"]) and idx == 8:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif check_not_mentioned(query_content, question_type) and idx == len(options) - 1:
                    is_correct = "Likely Correct"
                    correct_count += 1
                else:
                    is_correct = "Possibly Incorrect"
            elif question_type == "Texture":
                keywords = ["smooth", "rough", "scales", "scaly", "peeling"]
                matched_keywords = [k for k in keywords if k in text]
                if any(k in text for k in ["smooth", "no scales"]) and idx == 0:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif any(k in text for k in ["rough", "scales", "scaly", "peeling"]) and idx == 1:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif check_not_mentioned(query_content, question_type) and idx == len(options) - 1:
                    is_correct = "Likely Correct"
                    correct_count += 1
                else:
                    is_correct = "Possibly Incorrect"
            
            print(f"Q: {question} -> Predicted: {pred_option} ({is_correct}) | Matched Keywords: {matched_keywords}")
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
                q_data = next(q for q in questions if q["qid"] == qid)
                assert 0 <= res[qid] < len(q_data["options_en"]), f"Invalid index {res[qid]} for {qid}"
        print(f"Output {output_file} is valid")
    except Exception as e:
        print(f"Validation error for {output_file}: {e}")
        raise

# Hàm chính
def main():
    try:
        # Đặt phương thức spawn cho multiprocessing
        mp.set_start_method('spawn', force=True)
        print(f"PyTorch version: {torch.__version__}")
        
        print("Fine-tuning CLIP + DistilBERT...")
        model = fine_tune_model()
        
        # Tăng tốc inference nếu tương thích
        if torch.__version__ >= "2.0":
            model = torch.compile(model)
        
        valid_results, valid_data, question_dict = process_dataset(VALID_FILE, "closed_qa_valid_results.json", QUESTION_FILE, model, tokenizer, clip_model)
        validate_output(os.path.join(OUTPUT_DIR, "closed_qa_valid_results.json"), QUESTION_FILE)
        
        test_results, test_data, _ = process_dataset(TEST_FILE, "closed_qa_test_results.json", QUESTION_FILE, model, tokenizer, clip_model)
        validate_output(os.path.join(OUTPUT_DIR, "closed_qa_test_results.json"), QUESTION_FILE)
        
        print("\nEvaluating valid set samples:")
        manual_evaluation(valid_results, valid_data, question_dict, num_samples=5)
        
    except Exception as e:
        print(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()