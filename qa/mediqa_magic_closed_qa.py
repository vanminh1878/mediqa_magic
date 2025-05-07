import json
import os
import torch
import open_clip
from PIL import Image
from tqdm import tqdm
import numpy as np
import re
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

# Cấu hình môi trường CUDA
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Đồng bộ lỗi CUDA
os.environ["TORCH_USE_CUDA_DSA"] = "1"  # Kích hoạt kiểm tra lỗi phía thiết bị

# Kiểm tra môi trường
print(f"Phiên bản PyTorch: {torch.__version__}")
print(f"Phiên bản CUDA: {torch.version.cuda}")
print(f"GPU khả dụng: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Tên GPU: {torch.cuda.get_device_name(0)}")
    #print(f"VRAM khả dụng: {torch.cuda.memory_available(0) / 1e9:.2f} GB")

# Cấu hình thiết bị
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Thiết bị sử dụng ban đầu: {DEVICE}")

# Cấu hình chung
IMAGE_DIR = "/kaggle/input/mediqa-data/mediqa-data/images"
QUESTION_FILE = "/kaggle/input/mediqa-data/mediqa-data/closedquestions_definitions_imageclef2025.json"
TRAIN_FILE = "/kaggle/input/mediqa-data/mediqa-data/train_cvqa.json"
VALID_FILE = "/kaggle/input/mediqa-data/mediqa-data/valid.json"
TEST_FILE = "/kaggle/input/mediqa-data/mediqa-data/test.json"
OUTPUT_DIR = "/kaggle/working"
BATCH_SIZE = 1  # Giảm để tránh quá tải
MAX_IMAGES = 3  # Giảm số ảnh tối đa
MAX_LOCATIONS = 3
BERT_MODEL = "bert-base-uncased"
TRAIN_EPOCHS = 5
LEARNING_RATE = 2e-5

# Kiểm tra file tồn tại
for f in [QUESTION_FILE, TRAIN_FILE, VALID_FILE, TEST_FILE]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"File {f} không tồn tại")

# Kiểm tra trạng thái CUDA
def check_cuda_status():
    global DEVICE
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            print("Trạng thái CUDA ổn định")
            return True
        else:
            print("CUDA không khả dụng, sử dụng CPU")
            DEVICE = torch.device("cpu")
            return False
    except Exception as e:
        print(f"Lỗi trạng thái CUDA: {e}")
        print("Chuyển sang CPU do lỗi CUDA")
        DEVICE = torch.device("cpu")
        return False

# Tải câu hỏi để xác định num_labels tối đa
try:
    with open(QUESTION_FILE, "r") as f:
        questions = json.load(f)
    max_num_labels = max(len(q["options_en"]) for q in questions)
    print(f"Số lượng nhãn tối đa: {max_num_labels}")
except Exception as e:
    print(f"Lỗi khi tải câu hỏi: {e}")
    raise RuntimeError(f"Không thể tải file câu hỏi: {e}")

# Tải mô hình CLIP
clip_model = None
preprocess = None
try:
    print("Đang tải mô hình CLIP...")
    if check_cuda_status() and DEVICE.type == "cuda":
        with torch.amp.autocast('cuda'):
            clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
            clip_model = clip_model.to(DEVICE).eval()
        print(f"Mô hình CLIP đã được chuyển sang {DEVICE}")
        torch.cuda.empty_cache()  # Giải phóng bộ nhớ sau khi tải
    else:
        print("Chạy CLIP trên CPU...")
        clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        clip_model = clip_model.to(DEVICE).eval()
    print("Mô hình CLIP khởi tạo thành công")
except Exception as e:
    print(f"Lỗi khi tải mô hình CLIP: {e}")
    raise RuntimeError(f"Không thể tải mô hình CLIP: {e}")

# Tải mô hình BERT
try:
    print("Đang tải mô hình BERT...")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    bert_model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=max_num_labels)
    bert_model = bert_model.to(DEVICE)
    print(f"Mô hình BERT đã được chuyển sang {DEVICE}")
except Exception as e:
    print(f"Lỗi khi tải mô hình BERT: {e}")
    raise RuntimeError(f"Không thể tải mô hình BERT: {e}")

# Hàm kiểm tra "Not mentioned"
def check_not_mentioned(query_content, question_type):
    query_lower = query_content.lower()
    if question_type == "Onset":
        keywords = ["hour", "hours", "day", "days", "week", "weeks", "month", "months", "year", "years", "since", "ago", 
                    "ten years", "multiple years", "recently", "chronic", "long-term", r"\d+\s*(day|week|month|year)"]
        return not any(re.search(k, query_lower) for k in keywords)
    if question_type == "Itch":
        keywords = ["itch", "itchy", "pruritus", "scratch", "irritation", "irritated", "not itch", "no itch", "not itchy", 
                    "pruritic", "tingling"]
        return not any(k in query_lower for k in keywords)
    if question_type == "Site":
        keywords = ["area", "spot", "region", "widespread", "systemic", "single", "rashes", "palm", "ear", "all over", 
                    "whole body", "entire body", "everywhere", "diffuse", "generalized"]
        return not any(k in query_lower for k in keywords)
    if question_type == "Site Location":
        keywords = ["head", "neck", "arm", "leg", "chest", "abdomen", "back", "hand", "foot", "thigh", "palm", "lip", 
                    "finger", "eyebrow", "sole", "toe", "ear", "eyelash", "thumb", "wrist", "forearm", "shoulder", 
                    "knee", "ankle", "elbow", "shin", "calf"]
        return not any(k in query_lower for k in keywords)
    if question_type == "Size":
        keywords = ["size", "large", "small", "thumb", "bigger", "nail", "cm", "millimeter", "centimeter", "mm", 
                    "tiny", "huge", "diameter", r"\d+\s*(cm|mm)"]
        return not any(re.search(k, query_lower) for k in keywords)
    if question_type == "Skin Description":
        keywords = ["bump", "flat", "sunken", "thick", "thin", "wart", "crust", "scab", "weeping", "lesion", "raised", 
                    "swollen", "atrophy", "keratosis", "peeling", "exudation", "hives", "urticaria", "layered", 
                    "exfoliative", "eczema", "keratotic", "myxoid cyst", "blister", "macula", "papule", "nodule", 
                    "plaque", "pustule", "vesicle", "herpes", "acne", "rash", "patch"]
        return not any(k in query_lower for k in keywords)
    if question_type == "Lesion Color":
        keywords = ["color", "red", "pink", "brown", "blue", "purple", "black", "white", "pigment", "pale", "erythema", 
                    "purplish", "leukoplakia", "hematoma", "normal", "hypopigmentation", "hyperpigmentation", "yellow", 
                    "grey", "acne", "rash", "discoloration"]
        return not any(k in query_lower for k in keywords)
    if question_type == "Lesion Count":
        keywords = ["single", "multiple", "many", "few", "rash", "rashes", "lesions", "spots", "hives", "papula", 
                    "maculopapule", "urticaria", "numerous", "several", r"\d+\s*(piece|pieces|lesion|lesions|spot|spots)"]
        return not any(re.search(k, query_lower) for k in keywords)
    if question_type == "Texture":
        keywords = ["smooth", "rough", "texture", "scales", "scaly", "peeling", "exfoliative", "keratotic", "flaky", 
                    "granular", "eczema", "acne", "gritty", "velvety"]
        return not any(k in query_lower for k in keywords)
    return True

# Hàm kiểm tra văn bản trực tiếp để tạo nhãn giả
def check_text_directly(query_content, question_type, options):
    query_lower = query_content.lower()
    if question_type == "Onset":
        if re.search(r"\d+\s*hour|hours|recently", query_lower):
            return options[0], 0
        if re.search(r"\d+\s*day|days", query_lower):
            return options[1], 1
        if re.search(r"\d+\s*week|weeks", query_lower):
            return options[2], 2
        if re.search(r"\d+\s*month|months", query_lower):
            return options[3], 3
        if re.search(r"\d+\s*year|years|ago|chronic", query_lower):
            return options[4], 4
        if re.search(r"ten years|multiple years|long-term", query_lower):
            return options[5], 5
        if check_not_mentioned(query_content, question_type):
            return options[6], 6
    if question_type == "Itch":
        if any(k in query_lower for k in ["itch", "itchy", "pruritus", "scratch", "irritation", "irritated", "pruritic"]):
            return options[0], 0
        if any(k in query_lower for k in ["not itch", "no itch", "not itchy"]):
            return options[1], 1
        if check_not_mentioned(query_content, question_type):
            return options[2], 2
    if question_type == "Site":
        if any(k in query_lower for k in ["systemic", "widespread", "all over", "whole body", "entire body", "everywhere", "diffuse", "generalized"]):
            return options[2], 2
        if any(k in query_lower for k in ["area", "spot", "rashes", "palm", "ear"]):
            return options[1], 1
        if any(k in query_lower for k in ["single"]):
            return options[0], 0
        if check_not_mentioned(query_content, question_type):
            return options[3], 3
    if question_type == "Site Location":
        location_map = {
            "head": 0, "neck": 1, "upper extremities": 2, "lower extremities": 3, "chest/abdomen": 4, "back": 5,
            "hand": 2, "foot": 3, "thigh": 3, "palm": 2, "lip": 0, "finger": 2, "eyebrow": 0, "sole": 3, "toe": 3, 
            "thumb": 2, "arm": 2, "leg": 3, "chest": 4, "abdomen": 4, "wrist": 2, "forearm": 2, "shoulder": 2, 
            "knee": 3, "ankle": 3, "ear": 1, "elbow": 2, "shin": 3, "calf": 3
        }
        locations = []
        for loc, idx in location_map.items():
            if loc in query_lower and idx not in locations:
                locations.append(idx)
        if len(locations) > MAX_LOCATIONS:
            locations = locations[:MAX_LOCATIONS]
        if locations:
            return None, None
        if any(k in query_lower for k in ["all over", "widespread", "whole body", "entire body", "everywhere"]):
            return options[7], 7
        if check_not_mentioned(query_content, question_type):
            return options[7], 7
    if question_type == "Size":
        if (any(k in query_lower for k in ["small", "nail", "tiny"]) or re.search(r"\d+\s*mm|1\s*cm", query_lower)) and not any(k in query_lower for k in ["large", "bigger", "huge"]):
            return options[0], 0
        if any(k in query_lower for k in ["large", "bigger", "huge"]) or re.search(r"[2-9]\s*cm|\d{2,}\s*(cm|mm)", query_lower):
            return options[2], 2
        if check_not_mentioned(query_content, question_type):
            return options[3], 3
    if question_type == "Lesion Count":
        if re.search(r"\d+\s*(piece|pieces|lesion|lesions|spot|spots)|multiple|many|rashes|hives|papula|maculopapule|urticaria|numerous|several", query_lower):
            return options[1], 1
        if any(k in query_lower for k in ["single", "one"]) and not re.search(r"multiple|many|rashes", query_lower):
            return options[0], 0
        if check_not_mentioned(query_content, question_type):
            return options[2], 2
    if question_type == "Skin Description":
        if any(k in query_lower for k in ["hypopigmentation", "vitiligo", "macula", "eczema", "patch"]):
            return options[1], 1
        if any(k in query_lower for k in ["bump", "raised", "swollen", "hives", "urticaria", "blister", "papule", "nodule", "herpes", "acne", "myxoid cyst"]):
            return options[0], 0
        if any(k in query_lower for k in ["sunken", "atrophy"]):
            return options[2], 2
        if any(k in query_lower for k in ["thick", "keratosis", "keratotic"]):
            return options[3], 3
        if any(k in query_lower for k in ["thin"]):
            return options[4], 4
        if any(k in query_lower for k in ["wart"]):
            return options[5], 5
        if any(k in query_lower for k in ["crust", "peeling", "layered", "exfoliative", "herpes"]):
            return options[6], 6
        if any(k in query_lower for k in ["scab"]):
            return options[7], 7
        if any(k in query_lower for k in ["weeping", "exudation"]):
            return options[8], 8
        if check_not_mentioned(query_content, question_type):
            return options[9], 9
    if question_type == "Lesion Color":
        if any(k in query_lower for k in ["normal", "same as skin"]):
            return options[0], 0
        if any(k in query_lower for k in ["red", "pink", "erythema", "acne", "rash", "discoloration"]):
            return options[2], 2
        if any(k in query_lower for k in ["brown"]):
            return options[3], 3
        if any(k in query_lower for k in ["purple", "purplish", "hematoma"]):
            return options[5], 5
        if any(k in query_lower for k in ["white", "pale", "leukoplakia"]):
            return options[7], 7
        if any(k in query_lower for k in ["hyperpigmentation"]):
            return options[9], 9
        if any(k in query_lower for k in ["hypopigmentation"]):
            return options[10], 10
        if check_not_mentioned(query_content, question_type):
            return options[11], 11
    if question_type == "Texture":
        if any(k in query_lower for k in ["keratosis", "eczema", "rough", "scales", "scaly", "peeling", "exfoliative", "keratotic", "flaky", "granular", "acne", "gritty"]):
            return options[1], 1
        if any(k in query_lower for k in ["smooth", "no scales", "velvety"]) and not any(k in query_lower for k in ["acne", "eczema"]):
            return options[0], 0
        if check_not_mentioned(query_content, question_type):
            return options[2], 2
    return None, None

# Hàm collate tùy chỉnh để xử lý batch
def custom_collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    qids = [item["qid"] for item in batch]
    options = [item["options"] for item in batch]
    num_labels = [item["num_labels"] for item in batch]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label": labels,
        "qid": qids,
        "options": options,
        "num_labels": num_labels
    }

# Dataset cho huấn luyện và dự đoán
class QADataset(Dataset):
    def __init__(self, data, questions, tokenizer, max_length=128, is_train=False, is_test=False):
        self.data = data
        self.questions = {q["qid"]: q for q in questions}
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        self.items = []
        for case in data:
            encounter_id = case["encounter_id"]
            query_content = case.get("query_content_en", "")
            for qid, q_data in self.questions.items():
                question_type = q_data["question_type_en"]
                options = q_data["options_en"]
                num_labels = len(options)
                if is_test:
                    self.items.append({
                        "encounter_id": encounter_id,
                        "query_content": query_content,
                        "qid": qid,
                        "question": q_data["question_en"],
                        "options": options,
                        "label": 0,
                        "num_labels": num_labels
                    })
                elif is_train or not is_test:
                    _, label = check_text_directly(query_content, question_type, options)
                    if label is not None:
                        if not (0 <= label < num_labels):
                            print(f"Nhãn không hợp lệ {label} cho qid {qid} với {num_labels} tùy chọn")
                            continue
                        self.items.append({
                            "encounter_id": encounter_id,
                            "query_content": query_content,
                            "qid": qid,
                            "question": q_data["question_en"],
                            "options": options,
                            "label": label,
                            "num_labels": num_labels
                        })
        if not self.items and not is_test:
            raise ValueError(f"Không tìm thấy dữ liệu hợp lệ cho tập {'train' if is_train else 'valid'}.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        text = f"[CLS] {item['query_content']} [SEP] {item['question']} [SEP]"
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(item["label"], dtype=torch.long),
            "qid": item["qid"],
            "options": item["options"],
            "num_labels": item["num_labels"]
        }

# Huấn luyện BERT
def train_bert(model, train_data, valid_data, questions, epochs=TRAIN_EPOCHS):
    try:
        train_dataset = QADataset(train_data, questions, tokenizer, is_train=True)
        valid_dataset = QADataset(valid_data, questions, tokenizer)
    except ValueError as e:
        print(f"Lỗi khi tạo dataset: {e}")
        raise

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        if DEVICE.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()  # Chỉ gọi nếu CUDA khả dụng
        for batch in tqdm(train_loader, desc=f"Huấn luyện Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            num_labels = max(batch["num_labels"]) if batch["num_labels"] else max_num_labels

            if model.config.num_labels < num_labels:
                print(f"Cập nhật num_labels từ {model.config.num_labels} thành {num_labels}")
                model.config.num_labels = num_labels
                model.classifier = torch.nn.Linear(model.config.hidden_size, num_labels).to(DEVICE)
                model = model.to(DEVICE)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda' if DEVICE.type == 'cuda' else 'cpu'):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader)}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                num_labels = max(batch["num_labels"]) if batch["num_labels"] else max_num_labels
                
                if model.config.num_labels < num_labels:
                    print(f"Cập nhật num_labels từ {model.config.num_labels} thành {num_labels}")
                    model.config.num_labels = num_labels
                    model.classifier = torch.nn.Linear(model.config.hidden_size, num_labels).to(DEVICE)
                    model = model.to(DEVICE)

                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total if total > 0 else 0
        print(f"Độ chính xác tập valid: {accuracy:.2f}")
        model.train()

    return model

# Hàm nhúng văn bản bằng CLIP
def embed_text(texts, query_content, batch_size=BATCH_SIZE):
    combined_texts = [f"{query_content} {text}" for text in texts]
    embeddings = []
    for i in range(0, len(combined_texts), batch_size):
        batch = combined_texts[i:i+batch_size]
        try:
            inputs = open_clip.tokenize(batch).to(DEVICE)
            with torch.no_grad(), torch.amp.autocast('cuda' if DEVICE.type == 'cuda' else 'cpu'):
                text_embeds = clip_model.encode_text(inputs)
            embeddings.append(text_embeds.cpu().numpy())
        except Exception as e:
            print(f"Lỗi khi nhúng văn bản batch {i//batch_size}: {e}")
            embeddings.append(np.zeros((len(batch), clip_model.text_projection.shape[1])))
    return np.vstack(embeddings)

# Hàm nhúng hình ảnh bằng CLIP
def embed_images(image_paths):
    images = []
    valid_paths = []
    for path in image_paths[:MAX_IMAGES]:
        try:
            img = Image.open(path).convert("RGB")
            images.append(preprocess(img))
            valid_paths.append(path)
        except Exception as e:
            print(f"Lỗi khi tải hình ảnh {path}: {e}")
            continue
    if not images:
        return np.zeros((1, clip_model.visual.output_dim)), valid_paths
    images = torch.stack(images).to(DEVICE)
    with torch.no_grad(), torch.amp.autocast('cuda' if DEVICE.type == 'cuda' else 'cpu'):
        image_embeds = clip_model.encode_image(images)
    return image_embeds.cpu().numpy(), valid_paths

# Hàm trả lời câu hỏi bằng BERT
def answer_with_bert(bert_model, query_content, question, options, max_length=128):
    text = f"[CLS] {query_content} [SEP] {question} [SEP]"
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    num_labels = len(options)
    bert_model.eval()
    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :num_labels]  # Chỉ lấy logits cho số lượng tùy chọn hiện tại
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        max_prob = probs[0, pred_idx].item()

    # Kiểm tra pred_idx hợp lệ
    if pred_idx >= len(options):
        print(f"Cảnh báo: pred_idx {pred_idx} vượt quá len(options) {len(options)}. Chọn tùy chọn cuối cùng.")
        pred_idx = len(options) - 1
        max_prob = probs[0, pred_idx].item()

    return options[pred_idx], pred_idx, max_prob

# Hàm trả lời câu hỏi bằng CLIP
def answer_with_clip(image_paths, query_content, question, question_type, options):
    try:
        image_embeds, valid_paths = embed_images(image_paths)
        option_embeds = embed_text(options, query_content)

        image_embeds = image_embeds / np.linalg.norm(image_embeds, axis=1, keepdims=True)
        option_embeds = option_embeds / np.linalg.norm(option_embeds, axis=1, keepdims=True)
        similarities = image_embeds @ option_embeds.T

        weight_image = 0.8 if question_type in ["Lesion Color", "Skin Description", "Texture"] else 0.4
        weight_text = 1.0 - weight_image
        final_similarities = weight_image * similarities + weight_text * similarities
        best_option_idx = np.argmax(final_similarities.mean(axis=0))
        return options[best_option_idx], int(best_option_idx)
    except Exception as e:
        print(f"Lỗi CLIP cho câu hỏi {question}: {e}")
        return options[-1], int(len(options) - 1)

# Hàm trả lời câu hỏi kết hợp BERT và CLIP
def answer_question(bert_model, image_paths, query_content, question, question_type, options):
    bert_answer, bert_idx, bert_prob = answer_with_bert(bert_model, query_content, question, options)
    if bert_prob < 0.7 and question_type in ["Lesion Color", "Skin Description", "Texture", "Site Location", "Size"]:
        clip_answer, clip_idx = answer_with_clip(image_paths, query_content, question, question_type, options)
        return clip_answer, clip_idx
    return bert_answer, bert_idx

# Hàm xử lý tập dữ liệu
def process_dataset(data_file, output_filename, questions, bert_model, is_test=False):
    try:
        with open(data_file, "r") as f:
            data = json.load(f)
    except Exception as e:
        raise FileNotFoundError(f"Lỗi khi tải {data_file}: {e}")

    question_dict = {q["qid"]: q for q in questions}
    results = []

    for case in tqdm(data, desc=f"Xử lý {output_filename}"):
        encounter_id = case["encounter_id"]
        image_ids = case["image_ids"]
        image_paths = [os.path.join(IMAGE_DIR, img_id) for img_id in image_ids]
        query_content = case.get("query_content_en", "")

        locations = []
        query_lower = query_content.lower()
        location_map = {
            "head": 0, "neck": 1, "upper extremities": 2, "lower extremities": 3, "chest/abdomen": 4, "back": 5,
            "hand": 2, "foot": 3, "thigh": 3, "palm": 2, "lip": 0, "finger": 2, "eyebrow": 0, "sole": 3, "toe": 3,
            "thumb": 2, "arm": 2, "leg": 3, "chest": 4, "abdomen": 4, "wrist": 2, "forearm": 2, "shoulder": 2,
            "knee": 3, "ankle": 3, "ear": 1, "elbow": 2, "shin": 3, "calf": 3
        }
        for loc, idx in location_map.items():
            if loc in query_lower and idx not in locations:
                locations.append(idx)
        locations = sorted(list(set(locations)))[:MAX_LOCATIONS]

        result = {"encounter_id": encounter_id}

        for qid, q_data in question_dict.items():
            question = q_data["question_en"]
            question_type = q_data["question_type_en"]
            options = q_data["options_en"]
            num_labels = len(options)
            
            if question_type == "Site Location":
                q_num = int(qid.split("-")[1])
                if any(k in query_lower for k in ["all over", "widespread", "whole body", "entire body", "everywhere"]) and not locations:
                    result[qid] = 7
                    continue
                if q_num <= len(locations):
                    result[qid] = locations[q_num - 1]
                    continue
                else:
                    result[qid] = 7
                    continue
            else:
                answer, idx = answer_question(bert_model, image_paths, query_content, question, question_type, options)
                result[qid] = idx

        results.append(result)

    output_path = os.path.join(OUTPUT_DIR, output_filename)
    try:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Kết quả được lưu vào {output_path}")
    except Exception as e:
        print(f"Lỗi khi lưu kết quả vào {output_path}: {e}")
        raise

    return results, data, question_dict

# Hàm đánh giá thủ công (chỉ dùng nhãn giả)
def manual_score(results, data, question_dict, num_samples=15):
    data_dict = {case["encounter_id"]: case for case in data}
    sample_ids = np.random.choice([r["encounter_id"] for r in results], min(num_samples, len(results)), replace=False)

    print("\nĐánh giá thủ công (Mẫu ngẫu nhiên):")
    total_correct = 0
    total_questions = 0

    for enc_id in sample_ids:
        result = next(r for r in results if r["encounter_id"] == enc_id)
        case = data_dict[enc_id]
        query_content = case.get("query_content_en", "")

        print(f"\nEncounter ID: {enc_id}")
        print(f"Query: {query_content}")

        correct_count = 0
        q_count = 0
        for qid, idx in result.items():
            if qid == "encounter_id":
                continue
            q_data = question_dict[qid]
            question_type = q_data["question_type_en"]
            options = q_data["options_en"]
            pred_option = options[idx]
            _, true_idx = check_text_directly(query_content, question_type, options)
            if true_idx is None:
                continue

            is_correct = "Đúng" if idx == true_idx else "Sai"
            if is_correct == "Đúng":
                correct_count += 1
            print(f"Câu hỏi: {q_data['question_en']} -> Dự đoán: {pred_option} | Thật (giả): {options[true_idx]} ({is_correct})")
            q_count += 1

        accuracy = correct_count / q_count if q_count > 0 else 0
        print(f"Độ chính xác ước tính cho {enc_id}: {accuracy:.2f}")
        total_correct += correct_count
        total_questions += q_count

    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
    print(f"\nĐộ chính xác tổng thể ước tính: {overall_accuracy:.2f} ({total_correct}/{total_questions})")

# Hàm kiểm tra định dạng JSON
def validate_output(output_file, question_file):
    try:
        with open(output_file, "r") as f:
            results = json.load(f)
        with open(question_file, "r") as f:
            questions = json.load(f)

        qids = set(q["qid"] for q in questions)
        for res in results:
            assert "encounter_id" in res, "Thiếu encounter_id"
            for qid in qids:
                assert qid in res, f"Thiếu {qid}"
                assert isinstance(res[qid], int), f"{qid} không phải số nguyên"
                q_data = next(q for q in questions if q["qid"] == qid)
                assert 0 <= res[qid] < len(q_data["options_en"]), f"Chỉ số không hợp lệ {res[qid]} cho {qid} (tối đa: {len(q_data['options_en']) - 1})"
        print(f"Đầu ra {output_file} hợp lệ")
    except Exception as e:
        print(f"Lỗi xác thực cho {output_file}: {e}")
        raise

# Hàm chính
def main():
    try:
        with open(QUESTION_FILE, "r") as f:
            questions = json.load(f)
        with open(TRAIN_FILE, "r") as f:
            train_data = json.load(f)
        with open(VALID_FILE, "r") as f:
            valid_data = json.load(f)

        # Khởi tạo và huấn luyện BERT
        bert_model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=max_num_labels)
        bert_model = bert_model.to(DEVICE)
        bert_model = train_bert(bert_model, train_data, valid_data, questions)

        # Xử lý tập valid
        valid_results, valid_data, question_dict = process_dataset(
            VALID_FILE, "closed_qa_valid_results.json", questions, bert_model
        )
        validate_output(os.path.join(OUTPUT_DIR, "closed_qa_valid_results.json"), QUESTION_FILE)

        # Xử lý tập test
        test_results, test_data, _ = process_dataset(
            TEST_FILE, "closed_qa_test_results.json", questions, bert_model, is_test=True
        )
        validate_output(os.path.join(OUTPUT_DIR, "closed_qa_test_results.json"), QUESTION_FILE)

        print("\nĐánh giá mẫu tập valid:")
        manual_score(valid_results, valid_data, question_dict, num_samples=15)

    except Exception as e:
        print(f"Lỗi trong hàm main: {e}")
        raise

if __name__ == "__main__":
    main()