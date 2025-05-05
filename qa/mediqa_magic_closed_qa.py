import json
import os
import torch
import open_clip
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd

# Cấu hình
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_DIR = "/kaggle/input/mediqa-data/mediqa-data/images"
QUESTION_FILE = "/kaggle/input/mediqa-data/mediqa-data/closedquestions_definitions_imageclef2025.json"
VALID_FILE = "/kaggle/input/mediqa-data/mediqa-data/valid.json"
TEST_FILE = "/kaggle/input/mediqa-data/mediqa-data/test.json"
OUTPUT_DIR = "/kaggle/working"
BATCH_SIZE = 64  # Phù hợp với GPU T4

# Kiểm tra file tồn tại
for f in [QUESTION_FILE, VALID_FILE, TEST_FILE]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"File {f} not found")

# Tải mô hình CLIP
try:
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    clip_model = clip_model.to(DEVICE).eval()
except Exception as e:
    raise RuntimeError(f"Failed to load CLIP model: {e}")

# Hàm nhúng văn bản bằng CLIP
def embed_text(texts, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            inputs = open_clip.tokenize(batch).to(DEVICE)
            with torch.no_grad():
                text_embeds = clip_model.encode_text(inputs)
            embeddings.append(text_embeds.cpu().numpy())
        except Exception as e:
            print(f"Error embedding text batch {i//batch_size}: {e}")
            embeddings.append(np.zeros((len(batch), 512)))  # Giả lập embedding rỗng
    return np.vstack(embeddings)

# Hàm nhúng hình ảnh bằng CLIP
def embed_images(image_paths):
    images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            images.append(preprocess(img))
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            continue
    if not images:
        raise ValueError("No valid images loaded")
    images = torch.stack(images).to(DEVICE)
    with torch.no_grad():
        image_embeds = clip_model.encode_image(images)
    return image_embeds.cpu().numpy()

# Hàm trả lời câu hỏi đóng bằng CLIP
def answer_with_clip(image_paths, question, options):
    try:
        image_embeds = embed_images(image_paths)
        option_embeds = embed_text(options)
        
        # Chuẩn hóa embedding
        image_embeds = image_embeds / np.linalg.norm(image_embeds, axis=1, keepdims=True)
        option_embeds = option_embeds / np.linalg.norm(option_embeds, axis=1, keepdims=True)
        similarities = image_embeds @ option_embeds.T
        
        # Chọn tùy chọn có độ tương đồng cao nhất
        best_option_idx = np.argmax(similarities.mean(axis=0))
        return options[best_option_idx], best_option_idx
    except Exception as e:
        print(f"CLIP error for question {question}: {e}")
        return options[-1], len(options) - 1  # Trả về "Not mentioned" nếu lỗi

# Hàm xử lý một tập dữ liệu
def process_dataset(data_file, output_filename):
    try:
        with open(data_file, "r") as f:
            data = json.load(f)
    except Exception as e:
        raise FileNotFoundError(f"Error loading {data_file}: {e}")
    
    try:
        with open(QUESTION_FILE, "r") as f:
            questions = json.load(f)
    except Exception as e:
        raise FileNotFoundError(f"Error loading {QUESTION_FILE}: {e}")
    
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
            options = q_data["options_en"]
            
            # Chỉ dùng CLIP
            clip_answer, clip_idx = answer_with_clip(image_paths, question, options)
            result[qid] = clip_idx
        
        results.append(result)
    
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    try:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Error saving results to {output_path}: {e}")

# Hàm chính
def main():
    try:
        process_dataset(VALID_FILE, "closed_qa_valid_results.json")
        process_dataset(TEST_FILE, "closed_qa_test_results.json")
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()