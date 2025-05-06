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

# Cấu hình
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_DIR = "/kaggle/input/mediqa-data/mediqa-data/images"
TRAIN_FILE = "/kaggle/input/mediqa-data/mediqa-data/train_cvqa.json"
VALID_FILE = "/kaggle/input/mediqa-data/mediqa-data/valid.json"
TEST_FILE = "/kaggle/input/mediqa-data/mediqa-data/test.json"
QUESTION_FILE = "/kaggle/input/mediqa-data/mediqa-data/closedquestions_definitions_imageclef2025.json"
OUTPUT_DIR = "/kaggle/working"
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
EPOCHS = 3
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "model_checkpoint.pt")
MAX_IMAGES = 5
MAX_LENGTH = 128

# Xóa cache CUDA
torch.cuda.empty_cache()

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
    labels = {}
    
    for qid, q_data in question_dict.items():
        question_type = q_data["question_type_en"]
        options = q_data["options_en"]
        label = len(options) - 1
        
        if question_type == "Onset":
            if any(k in query_lower for k in ["hour"]):
                label = 0
            elif any(k in query_lower for k in ["day"]):
                label = 1
            elif any(k in query_lower for k in ["week"]):
                label = 2
            elif any(k in query_lower for k in ["month", "six months"]):
                label = 3
            elif any(k in query_lower for k in ["year"]):
                label = 4
            elif any(k in query_lower for k in ["ten years", "multiple years"]):
                label = 5
        elif question_type == "Itch":
            if any(k in query_lower for k in ["itch", "intense itching"]):
                label = 0
            elif any(k in query_lower for k in ["not itch"]):
                label = 1
        elif question_type == "Site":
            if any(k in query_lower for k in ["systemic", "widespread"]):
                label = 2
            elif any(k in query_lower for k in ["area", "spot"]):
                label = 1
        elif question_type == "Site Location":
            locations = []
            location_map = {
                "head": 0, "neck": 1, "arm": 2, "leg": 3, "chest": 4, "abdomen": 4, "back": 5,
                "hand": 2, "foot": 3, "thigh": 3, "palm": 2, "lip": 0, "finger": 2, "thumb": 2
            }
            for loc, idx in location_map.items():
                if loc in query_lower or loc in response_text:
                    if idx not in locations:
                        locations.append(idx)
            if qid.startswith("CQID011-"):
                q_num = int(qid.split("-")[1])
                label = locations[q_num - 1] if q_num <= len(locations) else len(options) - 1
        elif question_type == "Lesion Color":
            if any(k in query_lower for k in ["pale"]):
                label = 8
            elif "psoriasis" in response_text:
                label = 2
        elif question_type == "Lesion Count":
            if any(k in query_lower for k in ["rash", "lesions"]):
                label = 1
        
        labels[qid] = label
    
    return labels

# Hàm kiểm tra "Not mentioned"
def check_not_mentioned(query_content, question_type):
    query_content = query_content.lower()
    if question_type == "Onset":
        return not any(keyword in query_content for keyword in ["hour", "day", "week", "month", "year", "since", "ago", "ten years"])
    if question_type == "Itch":
        return not any(keyword in query_content for keyword in ["itch", "scratch", "irritat", "not itch", "intense"])
    if question_type == "Site":
        return not any(keyword in query_content for keyword in ["area", "spot", "region", "widespread", "systemic"])
    if question_type == "Site Location":
        return not any(keyword in query_content for keyword in ["head", "neck", "arm", "leg", "chest", "abdomen", "back", "hand", "foot", "thigh", "palm", "lip", "finger"])
    if question_type == "Size":
        return not any(keyword in query_content for keyword in ["size", "large", "small", "thumb", "palm"])
    if question_type == "Skin Description":
        return not any(keyword in query_content for keyword in ["bump", "flat", "sunken", "thick", "thin", "wart", "crust", "scab", "weep", "lesion"])
    if question_type == "Lesion Color":
        return not any(keyword in query_content for keyword in ["color", "red", "pink", "brown", "blue", "purple", "black", "white", "pigment", "pale"])
    if question_type == "Lesion Count":
        return not any(keyword in query_content for keyword in ["single", "multiple", "many", "few", "rash", "lesions"])
    if question_type == "Texture":
        return not any(keyword in query_content for keyword in ["smooth", "rough", "texture"])
    return True

# Mô hình kết hợp CLIP và DistilBERT
class ClipBertModel(nn.Module):
    def __init__(self, clip_model, bert_model, num_options):
        super().__init__()
        self.clip_model = clip_model
        self.bert_model = bert_model
        self.fc = nn.Linear(512 + 768, num_options)

    def forward(self, image_embeds, query_tokens, option_tokens):
        with autocast():
            query_embeds = self.bert_model(**query_tokens).last_hidden_state[:, 0, :].float()
            option_embeds = self.bert_model(**option_tokens).last_hidden_state[:, 0, :].float()
        
        combined_embed = image_embeds + query_embeds
        logits = self.fc(torch.cat([combined_embed, option_embeds], dim=-1))
        return logits

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
        
        # Tải và nhúng hình ảnh
        image_embeds = []
        for path in image_paths[:MAX_IMAGES]:
            try:
                img = Image.open(path).convert("RGB")
                img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad(), autocast():
                    embed = self.clip_model.encode_image(img_tensor).float()
                image_embeds.append(embed)
            except:
                continue
        if not image_embeds:
            image_embeds = [torch.zeros(1, 512).to(DEVICE)]
        image_embed = torch.mean(torch.cat(image_embeds, dim=0), dim=0, keepdim=True)
        
        # Token hóa văn bản truy vấn
        query_tokens = self.tokenizer(
            query_content,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )
        query_tokens = {k: v.squeeze(0).to(DEVICE) for k, v in query_tokens.items()}
        
        # Trích xuất nhãn
        labels = extract_labels(query_content, responses, self.question_dict)
        
        # Chuẩn bị tùy chọn
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
    
    # Chuẩn hóa query_tokens
    query_input_ids = torch.stack([item["query_tokens"]["input_ids"] for item in batch])
    query_attention_mask = torch.stack([item["query_tokens"]["attention_mask"] for item in batch])
    query_tokens = {
        "input_ids": query_input_ids,
        "attention_mask": query_attention_mask
    }
    
    # Chuẩn hóa option_texts
    option_texts = {}
    for qid in batch[0]["option_texts"]:
        option_input_ids = torch.stack([item["option_texts"][qid]["input_ids"] for item in batch])
        option_attention_mask = torch.stack([item["option_texts"][qid]["attention_mask"] for item in batch])
        option_texts[qid] = {
            "input_ids": option_input_ids,
            "attention_mask": option_attention_mask
        }
    
    # Chuẩn hóa labels
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
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    model = ClipBertModel(clip_model, bert_model, num_options=12).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda')
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            image_embeds = batch["image_embeds"]
            query_tokens = batch["query_tokens"]
            option_texts = batch["option_texts"]
            labels = batch["labels"]
            
            optimizer.zero_grad()
            with autocast():
                loss = 0
                for qid in option_texts:
                    logits = model(image_embeds, query_tokens, option_texts[qid])
                    loss += nn.CrossEntropyLoss()(logits, labels[qid])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")
        
        # Lưu checkpoint
        torch.save(model.state_dict(), CHECKPOINT_PATH)
    
    return model

# Hàm trả lời câu hỏi
def answer_with_clip_bert(model, image_paths, query_content, question, question_type, options, tokenizer, clip_model):
    try:
        if check_not_mentioned(query_content, question_type):
            return options[-1], len(options) - 1
        
        # Tải và nhúng hình ảnh
        image_embeds = []
        for path in image_paths[:MAX_IMAGES]:
            try:
                img = Image.open(path).convert("RGB")
                img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad(), autocast():
                    embed = clip_model.encode_image(img_tensor).float()
                image_embeds.append(embed)
            except:
                continue
        if not image_embeds:
            image_embeds = [torch.zeros(1, 512).to(DEVICE)]
        image_embed = torch.mean(torch.cat(image_embeds, dim=0), dim=0, keepdim=True)
        
        # Token hóa văn bản
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
        
        # Điều chỉnh trọng số
        weight_image = 0.3
        weight_text = 0.7
        if question_type in ["Onset", "Itch", "Lesion Count"]:
            weight_image, weight_text = 0.1, 0.9
        elif question_type in ["Site Location", "Lesion Color"]:
            weight_image, weight_text = 0.5, 0.5
        
        # Dự đoán
        model.eval()
        with torch.no_grad(), autocast():
            logits = model(image_embed, query_tokens, option_tokens)
            logits = weight_image * logits + weight_text * logits
            best_option_idx = torch.argmax(logits, dim=-1).item()
        
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
        if any(k in query_content.lower() for k in ["systemic", "widespread"]):
            locations = [0, 1, 2, 3, 4, 5]
        else:
            location_map = {
                "head": 0, "neck": 1, "arm": 2, "leg": 3, "chest": 4, "abdomen": 4, "back": 5,
                "hand": 2, "foot": 3, "thigh": 3, "palm": 2, "lip": 0, "finger": 2
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
            options = q_data["options_en"]
            pred_option = options[idx]
            
            is_correct = "Unknown"
            query_lower = query_content.lower()
            response_lower = " ".join([r["content_en"].lower() for r in responses])
            
            if q_data["question_type_en"] == "Onset":
                if any(k in query_lower for k in ["month", "year", "since", "ago"]) and idx in [3, 4, 5]:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif check_not_mentioned(query_content, q_data["question_type_en"]) and idx == len(options) - 1:
                    is_correct = "Likely Correct"
                    correct_count += 1
                else:
                    is_correct = "Possibly Incorrect"
            elif q_data["question_type_en"] == "Itch":
                if "itch" in query_lower and idx == 0:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif "not itch" in query_lower and idx == 1:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif check_not_mentioned(query_content, q_data["question_type_en"]) and idx == len(options) - 1:
                    is_correct = "Likely Correct"
                    correct_count += 1
                else:
                    is_correct = "Possibly Incorrect"
            elif q_data["question_type_en"] == "Site":
                if any(k in query_lower for k in ["systemic", "widespread"]) and idx == 2:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif check_not_mentioned(query_content, q_data["question_type_en"]) and idx == len(options) - 1:
                    is_correct = "Likely Correct"
                    correct_count += 1
                else:
                    is_correct = "Possibly Incorrect"
            elif q_data["question_type_en"] == "Site Location":
                locations = []
                location_map = {"head": 0, "neck": 1, "arm": 2, "leg": 3, "chest": 4, "back": 5, "foot": 3}
                for loc, loc_idx in location_map.items():
                    if loc in query_lower or loc in response_lower:
                        if loc_idx not in locations:
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
            elif q_data["question_type_en"] == "Lesion Color":
                if any(k in query_lower for k in ["pale"]) and idx == 8:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif "psoriasis" in response_lower and idx == 2:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif check_not_mentioned(query_content, q_data["question_type_en"]) and idx == len(options) - 1:
                    is_correct = "Likely Correct"
                    correct_count += 1
                else:
                    is_correct = "Possibly Incorrect"
            elif q_data["question_type_en"] == "Lesion Count":
                if any(k in query_lower for k in ["rash", "lesions"]) and idx == 1:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif check_not_mentioned(query_content, q_data["question_type_en"]) and idx == len(options) - 1:
                    is_correct = "Likely Correct"
                    correct_count += 1
                else:
                    is_correct = "Possibly Incorrect"
            else:
                if check_not_mentioned(query_content, q_data["question_type_en"]) and idx == len(options) - 1:
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
        valid_results, valid_data, question_dict = process_dataset(VALID_FILE, "closed_qa_valid_results.json", QUESTION_FILE, model, tokenizer, clip_model)
        validate_output(os.path.join(OUTPUT_DIR, "closed_qa_valid_results.json"), QUESTION_FILE)
        
        # Xử lý tập test
        test_results, test_data, _ = process_dataset(TEST_FILE, "closed_qa_test_results.json", QUESTION_FILE, model, tokenizer, clip_model)
        validate_output(os.path.join(OUTPUT_DIR, "closed_qa_test_results.json"), QUESTION_FILE)
        
        # Đánh giá thủ công
        print("\nEvaluating valid set samples:")
        manual_evaluation(valid_results, valid_data, question_dict, num_samples=5)
        
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()