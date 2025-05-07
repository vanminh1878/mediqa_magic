import json
import os
import torch
import open_clip
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np
import re

# Cấu hình
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_DIR = "/kaggle/input/mediqa-data/mediqa-data/images"
QUESTION_FILE = "/kaggle/input/mediqa-data/mediqa-data/closedquestions_definitions_imageclef2025.json"
VALID_FILE = "/kaggle/input/mediqa-data/mediqa-data/valid.json"
TEST_FILE = "/kaggle/input/mediqa-data/mediqa-data/test.json"
OUTPUT_DIR = "/kaggle/working"
BATCH_SIZE = 16
MAX_IMAGES = 5
MAX_LOCATIONS = 3

# Kiểm tra file tồn tại
for f in [QUESTION_FILE, VALID_FILE, TEST_FILE]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"File {f} not found")

# Tải mô hình CLIP
try:
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
    clip_model = clip_model.to(DEVICE).eval()
except Exception as e:
    raise RuntimeError(f"Failed to load CLIP model: {e}")

# Hàm kiểm tra "Not mentioned" với từ khóa mở rộng
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

# Hàm kiểm tra văn bản trực tiếp để chọn đáp án
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
        if any(k in query_lower for k in ["large", "bigger", "huge"]) or re.search(r"[2-9]\s*cm|\d{2,}\s*mm", query_lower):
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

# Hàm nhúng văn bản bằng CLIP
def embed_text(texts, query_content, batch_size=BATCH_SIZE):
    combined_texts = [f"{query_content} {text}" for text in texts]
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
            print(f"Error loading image {path}: {e}")
            continue
    if not images:
        return np.zeros((1, clip_model.visual.output_dim)), valid_paths
    images = torch.stack(images).to(DEVICE)
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_embeds = clip_model.encode_image(images)
    return image_embeds.cpu().numpy(), valid_paths

# Hàm trả lời câu hỏi đóng
def answer_with_clip(image_paths, query_content, question, question_type, options):
    try:
        # Kiểm tra văn bản trực tiếp trước
        text_answer, text_idx = check_text_directly(query_content, question_type, options)
        if text_answer is not None:
            return text_answer, text_idx
        
        # Sử dụng CLIP nếu không có đáp án từ văn bản
        image_embeds, valid_paths = embed_images(image_paths)
        option_embeds = embed_text(options, query_content)
        
        image_embeds = image_embeds / np.linalg.norm(image_embeds, axis=1, keepdims=True)
        option_embeds = option_embeds / np.linalg.norm(option_embeds, axis=1, keepdims=True)
        similarities = image_embeds @ option_embeds.T
        
        # Điều chỉnh trọng số hình ảnh/văn bản
        weight_image = 0.5
        weight_text = 0.5
        if question_type in ["Onset", "Itch", "Lesion Count"]:
            weight_image, weight_text = 0.3, 0.7  # Tăng weight_image khi văn bản thiếu thông tin
        elif question_type in ["Site", "Site Location", "Size"]:
            weight_image, weight_text = 0.4, 0.6
        elif question_type in ["Skin Description", "Lesion Color", "Texture"]:
            weight_image, weight_text = 0.5, 0.5  # Cân bằng hơn để xử lý đặc điểm hình ảnh
        
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
                clip_answer, clip_idx = answer_with_clip(image_paths, query_content, question, question_type, options)
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
def manual_score(results, data, question_dict, num_samples=15):
    data_dict = {case["encounter_id"]: case for case in data}
    sample_ids = np.random.choice([r["encounter_id"] for r in results], min(num_samples, len(results)), replace=False)
    
    print("\nManual Evaluation (Random Samples):")
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
            question = q_data["question_en"]
            question_type = q_data["question_type_en"]
            options = q_data["options_en"]
            pred_option = options[idx]
            
            is_correct = "Unknown"
            query_lower = query_content.lower()
            matched_keywords = []
            
            if question_type == "Onset":
                keywords = ["hour", "hours", "day", "days", "week", "weeks", "month", "months", "year", "years", "since", "ago", "ten years", "multiple years", "recently", "chronic", "long-term"]
                matched_keywords = [k for k in keywords if re.search(k, query_lower)]
                if any(k in query_lower for k in ["hour", "hours", "recently"]) and idx == 0:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif any(k in query_lower for k in ["day", "days"]) and idx == 1:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif any(k in query_lower for k in ["week", "weeks"]) and idx == 2:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif any(k in query_lower for k in ["month", "months"]) and idx == 3:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif any(k in query_lower for k in ["year", "years", "ago", "chronic"]) and idx == 4:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif any(k in query_lower for k in ["ten years", "multiple years", "long-term"]) and idx == 5:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif check_not_mentioned(query_content, question_type) and idx == 6:
                    is_correct = "Likely Correct"
                    correct_count += 1
                else:
                    is_correct = "Possibly Incorrect"
            elif question_type == "Itch":
                keywords = ["itch", "itchy", "pruritus", "scratch", "irritation", "irritated", "not itch", "no itch", "not itchy", "pruritic", "tingling"]
                matched_keywords = [k for k in keywords if k in query_lower]
                if any(k in query_lower for k in ["itch", "itchy", "pruritus", "scratch", "irritation", "irritated", "pruritic"]) and idx == 0:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif any(k in query_lower for k in ["not itch", "no itch", "not itchy"]) and idx == 1:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif check_not_mentioned(query_content, question_type) and idx == 2:
                    is_correct = "Likely Correct"
                    correct_count += 1
                else:
                    is_correct = "Possibly Incorrect"
            elif question_type == "Site":
                keywords = ["systemic", "widespread", "area", "spot", "region", "single", "rashes", "palm", "ear", "all over", "whole body", "entire body", "everywhere", "diffuse", "generalized"]
                matched_keywords = [k for k in keywords if k in query_lower]
                if any(k in query_lower for k in ["systemic", "widespread", "all over", "whole body", "entire body", "everywhere", "diffuse", "generalized"]) and idx == 2:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif any(k in query_lower for k in ["area", "spot", "rashes", "palm", "ear"]) and idx == 1:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif any(k in query_lower for k in ["single"]) and idx == 0:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif check_not_mentioned(query_content, question_type) and idx == 3:
                    is_correct = "Likely Correct"
                    correct_count += 1
                else:
                    is_correct = "Possibly Incorrect"
            elif question_type == "Site Location":
                locations = []
                location_map = {
                    "head": 0, "neck": 1, "upper extremities": 2, "lower extremities": 3, "chest/abdomen": 4, "back": 5,
                    "hand": 2, "foot": 3, "thigh": 3, "palm": 2, "lip": 0, "finger": 2, "eyebrow": 0, "sole": 3, "toe": 3, 
                    "thumb": 2, "arm": 2, "leg": 3, "chest": 4, "abdomen": 4, "wrist": 2, "forearm": 2, "shoulder": 2, 
                    "knee": 3, "ankle": 3, "ear": 1, "elbow": 2, "shin": 3, "calf": 3
                }
                matched_keywords = [loc for loc in location_map if loc in query_lower]
                for loc, loc_idx in location_map.items():
                    if loc in query_lower and loc_idx not in locations:
                        locations.append(loc_idx)
                locations = sorted(list(set(locations)))[:MAX_LOCATIONS]
                q_num = int(qid.split("-")[1])
                if any(k in query_lower for k in ["all over", "widespread", "whole body", "entire body", "everywhere"]) and not locations and idx == 7:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif q_num <= len(locations) and idx == locations[q_num - 1]:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif q_num > len(locations) and idx == 7:
                    is_correct = "Likely Correct"
                    correct_count += 1
                else:
                    is_correct = "Possibly Incorrect"
            elif question_type == "Size":
                keywords = ["large", "small", "thumb", "bigger", "nail", "cm", "millimeter", "centimeter", "mm", "tiny", "huge"]
                matched_keywords = [k for k in keywords if re.search(k, query_lower)]
                if (any(k in query_lower for k in ["small", "nail", "tiny"]) or re.search(r"\d+\s*mm|1\s*cm", query_lower)) and not any(k in query_lower for k in ["large", "bigger", "huge"]) and idx == 0:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif (any(k in query_lower for k in ["large", "bigger", "huge"]) or re.search(r"[2-9]\s*cm|\d{2,}\s*mm", query_lower)) and idx == 2:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif check_not_mentioned(query_content, question_type) and idx == 3:
                    is_correct = "Likely Correct"
                    correct_count += 1
                else:
                    is_correct = "Possibly Incorrect"
            elif question_type == "Skin Description":
                keywords = ["bump", "flat", "sunken", "thick", "thin", "wart", "crust", "scab", "weeping", "raised", "swollen", "atrophy", "keratosis", "peeling", "exudation", "hives", "urticaria", "layered", "exfoliative", "eczema", "keratotic", "myxoid cyst", "blister", "macula", "papule", "nodule", "plaque", "pustule", "vesicle", "herpes", "acne", "rash", "patch"]
                matched_keywords = [k for k in keywords if k in query_lower]
                if any(k in query_lower for k in ["hypopigmentation", "vitiligo", "macula", "eczema", "patch"]) and idx == 1:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif any(k in query_lower for k in ["bump", "raised", "swollen", "hives", "urticaria", "myxoid cyst", "blister", "papule", "nodule", "herpes", "acne"]) and idx == 0:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif any(k in query_lower for k in ["sunken", "atrophy"]) and idx == 2:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif any(k in query_lower for k in ["thick", "keratosis", "keratotic"]) and idx == 3:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif any(k in query_lower for k in ["thin"]) and idx == 4:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif any(k in query_lower for k in ["wart"]) and idx == 5:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif any(k in query_lower for k in ["crust", "peeling", "layered", "exfoliative", "herpes"]) and idx == 6:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif any(k in query_lower for k in ["scab"]) and idx == 7:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif any(k in query_lower for k in ["weeping", "exudation"]) and idx == 8:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif check_not_mentioned(query_content, question_type) and idx == 9:
                    is_correct = "Likely Correct"
                    correct_count += 1
                else:
                    is_correct = "Possibly Incorrect"
            elif question_type == "Lesion Color":
                keywords = ["red", "pink", "brown", "blue", "purple", "black", "white", "pale", "erythema", "purplish", "leukoplakia", "hematoma", "normal", "hypopigmentation", "hyperpigmentation", "yellow", "grey", "acne", "rash", "discoloration"]
                matched_keywords = [k for k in keywords if k in query_lower]
                if any(k in query_lower for k in ["normal", "same as skin"]) and idx == 0:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif any(k in query_lower for k in ["red", "pink", "erythema", "acne", "rash", "discoloration"]) and idx == 2:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif any(k in query_lower for k in ["brown"]) and idx == 3:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif any(k in query_lower for k in ["purple", "purplish", "hematoma"]) and idx == 5:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif any(k in query_lower for k in ["white", "pale", "leukoplakia"]) and idx == 7:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif any(k in query_lower for k in ["hyperpigmentation"]) and idx == 9:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif any(k in query_lower for k in ["hypopigmentation"]) and idx == 10:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif check_not_mentioned(query_content, question_type) and idx == 11:
                    is_correct = "Likely Correct"
                    correct_count += 1
                else:
                    is_correct = "Possibly Incorrect"
            elif question_type == "Lesion Count":
                keywords = ["rash", "rashes", "lesions", "spots", "multiple", "single", "hives", "many", "few", "papula", "maculopapule", "urticaria", "numerous", "several"]
                matched_keywords = [k for k in keywords if re.search(k, query_lower)]
                if (any(k in query_lower for k in ["rash", "rashes", "lesions", "spots", "multiple", "hives", "many", "papula", "maculopapule", "urticaria", "numerous", "several"]) or re.search(r"\d+\s*(piece|pieces|lesion|lesions|spot|spots)", query_lower)) and idx == 1:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif any(k in query_lower for k in ["single", "one"]) and not re.search(r"multiple|many|rashes", query_lower) and idx == 0:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif check_not_mentioned(query_content, question_type) and idx == 2:
                    is_correct = "Likely Correct"
                    correct_count += 1
                else:
                    is_correct = "Possibly Incorrect"
            elif question_type == "Texture":
                keywords = ["smooth", "rough", "scales", "scaly", "peeling", "exfoliative", "keratotic", "flaky", "granular", "eczema", "acne", "gritty", "velvety"]
                matched_keywords = [k for k in keywords if k in query_lower]
                if any(k in query_lower for k in ["smooth", "no scales", "velvety"]) and not any(k in query_lower for k in ["acne", "eczema"]) and idx == 0:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif any(k in query_lower for k in ["rough", "scales", "scaly", "peeling", "exfoliative", "keratotic", "flaky", "granular", "eczema", "acne", "gritty"]) and idx == 1:
                    is_correct = "Likely Correct"
                    correct_count += 1
                elif check_not_mentioned(query_content, question_type) and idx == 2:
                    is_correct = "Likely Correct"
                    correct_count += 1
                else:
                    is_correct = "Possibly Incorrect"
            
            print(f"Q: {question} -> Predicted: {pred_option} ({is_correct}) | Matched Keywords: {matched_keywords}")
            q_count += 1
        
        accuracy = correct_count / q_count if q_count > 0 else 0
        print(f"Estimated Accuracy for {enc_id}: {accuracy:.2f}")
        total_correct += correct_count
        total_questions += q_count
    
    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
    print(f"\nOverall Estimated Accuracy: {overall_accuracy:.2f} ({total_correct}/{total_questions})")

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
                assert 0 <= res[qid] < len(q_data["options_en"]), f"Invalid index {res[qid]} for {qid} (max: {len(q_data['options_en']) - 1})"
        print(f"Output {output_file} is valid")
    except Exception as e:
        print(f"Validation error for {output_file}: {e}")
        raise

# Hàm chính
def main():
    try:
        valid_results, valid_data, question_dict = process_dataset(VALID_FILE, "closed_qa_valid_results.json")
        validate_output(os.path.join(OUTPUT_DIR, "closed_qa_valid_results.json"), QUESTION_FILE)
        
        test_results, test_data, _ = process_dataset(TEST_FILE, "closed_qa_test_results.json")
        validate_output(os.path.join(OUTPUT_DIR, "closed_qa_test_results.json"), QUESTION_FILE)
        
        print("\nEvaluating valid set samples:")
        manual_score(valid_results, valid_data, question_dict, num_samples=15)
        
    except Exception as e:
        print(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()