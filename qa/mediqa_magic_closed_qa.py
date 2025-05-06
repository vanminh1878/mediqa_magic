import json
import os
import torch
import open_clip
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.cuda.amp import autocast

# Cấu hình
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_DIR = "/kaggle/input/mediqa-data/mediqa-data/images"
QUESTION_FILE = "/kaggle/input/mediqa-data/mediqa-data/closedquestions_definitions_imageclef2025.json"
VALID_FILE = "/kaggle/input/mediqa-data/mediqa-data/valid.json"
TEST_FILE = "/kaggle/input/mediqa-data/mediqa-data/test.json"
OUTPUT_DIR = "/kaggle/working"
BATCH_SIZE = 64  # Phù hợp với GPU T4
WEIGHT_IMAGE = 0.5  # Trọng số nhúng hình ảnh
WEIGHT_TEXT = 0.5   # Trọng số nhúng văn bản

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
            with torch.no_grad(), autocast():
                text_embeds = clip_model.encode_text(inputs)
            embeddings.append(text_embeds.cpu().numpy())
        except Exception as e:
            print(f"Error embedding text batch {i//batch_size}: {e}")
            embeddings.append(np.zeros((len(batch), 512)))
    return np.vstack(embeddings)

# Hàm nhúng hình ảnh bằng CLIP
def embed_images(image_paths):
    images = []
    valid_paths = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            images.append(preprocess(img))
            valid_paths.append(path)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            continue
    if not images:
        return np.zeros((1, 512)), []  # Trả về embedding rỗng
    images = torch.stack(images).to(DEVICE)
    with torch.no_grad(), autocast():
        image_embeds = clip_model.encode_image(images)
    return image_embeds.cpu().numpy(), valid_paths

# Hàm kiểm tra "Not mentioned" dựa trên văn bản truy vấn
def check_not_mentioned(query_content, question_type):
    query_content = query_content.lower()
    if question_type == "Onset":
        return not any(keyword in query_content for keyword in ["hour", "day", "week", "month", "year"])
    if question_type == "Itch":
        return not any(keyword in query_content for keyword in ["itch", "scratch", "irritat"])
    if question_type == "Site":
        return not any(keyword in query_content for keyword in ["area", "spot", "region", "widespread"])
    if question_type == "Site Location":
        return not any(keyword in query_content for keyword in ["head", "neck", "arm", "leg", "chest", "abdomen", "back", "hand", "foot", "thigh", "palm"])
    if question_type == "Size":
        return not any(keyword in query_content for keyword in ["size", "large", "small", "thumb", "palm"])
    if question_type == "Skin Description":
        return not any(keyword in query_content for keyword in ["bump", "flat", "sunken", "thick", "thin", "wart", "crust", "scab", "weep"])
    if question_type == "Lesion Color":
        return not any(keyword in query_content for keyword in ["color", "red", "pink", "brown", "blue", "purple", "black", "white", "pigment"])
    if question_type == "Lesion Count":
        return not any(keyword in query_content for keyword in ["single", "multiple", "many", "few"])
    if question_type == "Texture":
        return not any(keyword in query_content for keyword in ["smooth", "rough", "texture"])
    return True

# Hàm trả lời câu hỏi đóng bằng CLIP
def answer_with_clip(image_paths, query_content, question, question_type, options):
    try:
        # Kiểm tra "Not mentioned"
        if check_not_mentioned(query_content, question_type):
            return options[-1], len(options) - 1
        
        # Nhúng hình ảnh
        image_embeds, valid_paths = embed_images(image_paths)
        if len(valid_paths) == 0:
            return options[-1], len(options) - 1
        
        # Nhúng văn bản truy vấn
        query_embed = embed_text([query_content])[0]
        
        # Nhúng tùy chọn
        option_embeds = embed_text(options)
        
        # Chuẩn hóa embedding
        image_embeds = image_embeds / np.linalg.norm(image_embeds, axis=1, keepdims=True)
        query_embed = query_embed / np.linalg.norm(query_embed, keepdims=True)
        option_embeds = option_embeds / np.linalg.norm(option_embeds, axis=1, keepdims=True)
        
        # Kết hợp nhúng hình ảnh và truy vấn
        combined_embed = WEIGHT_IMAGE * image_embeds.mean(axis=0) + WEIGHT_TEXT * query_embed
        similarities = combined_embed @ option_embeds.T
        
        # Chọn tùy chọn có độ tương đồng cao nhất
        best_option_idx = np.argmax(similarities)
        return options[best_option_idx], int(best_option_idx)
    except Exception as e:
        print(f"CLIP error for question {question}: {e}")
        return options[-1], len(options) - 1

# Hàm xử lý một tập dữ liệu
def process_dataset(data_file, output_filename, question_file):
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
            
            # Dự đoán bằng CLIP
            _, clip_idx = answer_with_clip(image_paths, query_content, question, question_type, options)
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

# Hàm đánh giá thủ công (không có ground truth)
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
            if q_data["question_type_en"] == "Onset" and "month" in query_content.lower() and idx == 3:  # within months
                is_correct = "Likely Correct"
                correct_count += 1
            elif q_data["question_type_en"] == "Site Location" and "thigh" in query_content.lower() and idx == 3:  # lower extremities
                is_correct = "Likely Correct"
                correct_count += 1
            elif q_data["question_type_en"] == "Lesion Color" and "red" in query_content.lower() and idx == 2:  # red
                is_correct = "Likely Correct"
                correct_count += 1
            elif idx == len(options) - 1 and check_not_mentioned(query_content, q_data["question_type_en"]):
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
        print(f"Output  {output_file} is valid")
    except Exception as e:
        print(f"Validation error for {output_file}: {e}")

# Hàm chính
def main():
    try:
        # Xử lý tập valid
        valid_results, valid_data, question_dict = process_dataset(VALID_FILE, "closed_qa_valid_results.json", QUESTION_FILE)
        validate_output(os.path.join(OUTPUT_DIR, "closed_qa_valid_results.json"), QUESTION_FILE)
        
        # Xử lý tập test
        test_results, test_data, _ = process_dataset(TEST_FILE, "closed_qa_test_results.json", QUESTION_FILE)
        validate_output(os.path.join(OUTPUT_DIR, "closed_qa_test_results.json"), QUESTION_FILE)
        
        # Đánh giá thủ công
        print("\nEvaluating valid set samples:")
        manual_evaluation(valid_results, valid_data, question_dict, num_samples=5)
        
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()