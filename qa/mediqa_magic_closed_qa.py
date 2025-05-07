import json
import os
import torch
import open_clip
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Cấu hình
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_DIR = "/kaggle/input/mediqa-data/mediqa-data/images"
QUESTION_FILE = "/kaggle/input/mediqa-data/mediqa-data/closedquestions_definitions_imageclef2025.json"
VALID_FILE = "/kaggle/input/mediqa-data/mediqa-data/valid.json"
TEST_FILE = "/kaggle/input/mediqa-data/mediqa-data/test.json"
OUTPUT_DIR = "/kaggle/working"
BATCH_SIZE = 32

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

# Hàm kiểm tra "Not mentioned"
def check_not_mentioned(query_content, responses, question_type):
    query_lower = query_content.lower()
    response_text = " ".join([r["content_en"].lower() for r in responses]).lower() if responses else ""
    text = query_lower + " " + response_text
    if question_type == "Onset":
        return not any(k in text for k in ["hour", "day", "week", "month", "year", "since", "ago", "ten years"])
    if question_type == "Itch":
        return not any(k in text for k in ["itch", "scratch", "irritat", "pruritus", "not itch", "no itch"])
    if question_type == "Site":
        return not any(k in text for k in ["area", "spot", "region", "widespread", "systemic", "single", "rashes", "palm", "ear"])
    if question_type == "Site Location":
        return not any(k in text for k in ["head", "neck", "arm", "leg", "chest", "abdomen", "back", "hand", "foot", "thigh", "palm", "lip", "finger", "eyebrow", "sole", "toe", "ear", "eyelash", "thumb"])
    if question_type == "Size":
        return not any(k in text for k in ["size", "large", "small", "thumb", "palm", "bigger", "nail", "1 cm"])
    if question_type == "Skin Description":
        return not any(k in text for k in ["bump", "flat", "sunken", "thick", "thin", "wart", "crust", "scab", "weep", "lesion", "raised", "swollen", "atrophy", "keratosis", "peeling", "exudation", "hives", "urticaria", "layered", "exfoliative", "eczema", "keratotic", "myxoid cyst"])
    if question_type == "Lesion Color":
        return not any(k in text for k in ["color", "red", "pink", "brown", "blue", "purple", "black", "white", "pigment", "pale", "erythema", "purplish", "leukoplakia", "hematoma", "normal"])
    if question_type == "Lesion Count":
        return not any(k in text for k in ["single", "multiple", "many", "few", "rash", "rashes", "lesions", "spots", "hives"])
    if question_type == "Texture":
        return not any(k in text for k in ["smooth", "rough", "texture", "scales", "scaly", "peeling", "exfoliative", "keratotic"])
    return True

# Hàm nhúng văn bản bằng CLIP
def embed_text(texts, query_content, responses, batch_size=32):
    response_text = " ".join([r["content_en"] for r in responses]) if responses else ""
    combined_texts = [f"{query_content} {response_text} {text}" for text in texts]
    embeddings = []
    for i in range(0, len(combined_texts), batch_size):
        batch = combined_texts[i:i+batch_size]
        try:
            inputs = open_clip.tokenize(batch).to(DEVICE)
            with torch.no_grad(), torch.cuda.amp.autocast():
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
        raise ValueError("No valid images loaded")
    images = torch.stack(images).to(DEVICE)
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_embeds = clip_model.encode_image(images)
    return image_embeds.cpu().numpy(), valid_paths

# Hàm trả lời câu hỏi đóng
def answer_with_clip(image_paths, query_content, responses, question, question_type, options):
    try:
        if check_not_mentioned(query_content, responses, question_type):
            return options[-1], len(options) - 1
        
        image_embeds, valid_paths = embed_images(image_paths)
        option_embeds = embed_text(options, query_content, responses)
        
        image_embeds = image_embeds / np.linalg.norm(image_embeds, axis=1, keepdims=True)
        option_embeds = option_embeds / np.linalg.norm(option_embeds, axis=1, keepdims=True)
        similarities = image_embeds @ option_embeds.T
        
        weight_image = 0.5
        weight_text = 0.5
        if question_type in ["Onset", "Itch", "Lesion Count"]:
            weight_image, weight_text = 0.2, 0.8
        elif question_type in ["Site Location", "Lesion Color", "Skin Description"]:
            weight_image, weight_text = 0.7, 0.3
        
        final_similarities = weight_image * similarities + weight_text * similarities
        best_option_idx = np.argmax(final_similarities.mean(axis=0))
        return options[best_option_idx], int(best_option_idx)
    except Exception as e:
        print(f"CLIP error for question {question}: {e}")
        return options[-1], int(len(options) - 1)

# Hàm xử lý tập dữ liệu
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
        responses = case.get("responses", [])
        
        locations = []
        query_lower = query_content.lower()
        if any(k in query_lower for k in ["systemic", "widespread", "whole body"]):
            locations = [0, 1, 2, 3, 4, 5]
        else:
            location_map = {
                "head": 0, "neck": 1, "upper extremities": 2, "lower extremities": 3, "chest/abdomen": 4, "back": 5,
                "hand": 2, "foot": 3, "thigh": 3, "palm": 2, "lip": 0, "finger": 2, "eyebrow": 0, "sole": 3, "toe": 3, "ear": 0, "eyelash": 0, "thumb": 2
            }
            for loc, idx in location_map.items():
                if loc in query_lower and idx not in locations:
                    locations.append(idx)
        
        result = {"encounter_id": encounter_id}
        
        for qid, q_data in question_dict.items():
            question = q_data["question_en"]
            question_type = q_data["question_type_en"]
            options = q_data["options_en"]
            
            if question_type == "Site Location" and locations:
                q_num = int(qid.split("-")[1])
                if q_num <= len(locations):
                    result[qid] = locations[q_num - 1]
                    continue
            
            clip_answer, clip_idx = answer_with_clip(image_paths, query_content, responses, question, question_type, options)
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

# Hàm chính
def main():
    try:
        process_dataset(VALID_FILE, "closed_qa_valid_results.json")
        process_dataset(TEST_FILE, "closed_qa_test_results.json")
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()