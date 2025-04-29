import os
import json
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import Blip2Processor
import torch
from tqdm import tqdm
import logging
from sentence_transformers import SentenceTransformer, util
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Tải tài nguyên NLTK
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Tải mô hình scispacy
nlp = spacy.load("en_core_sci_sm")

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def infer_qid(query_content, closed_qa_dict, model, threshold=0.2):
    query_content = query_content.lower().strip()
    if not query_content:
        return [(None, None, None)]
    
    # Trích xuất từ khóa y khoa bằng scispacy
    doc = nlp(query_content)
    query_keywords = ' '.join([ent.text for ent in doc.ents])
    if not query_keywords:
        # Loại bỏ stop words nếu không tìm thấy thực thể y khoa
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(query_content)
        query_keywords = ' '.join([word for word in tokens if word not in stop_words])
    
    query_embedding = model.encode(query_keywords, convert_to_tensor=True, show_progress_bar=False)
    
    matched_qids = []
    for qa in closed_qa_dict:
        qid = qa["qid"]
        question_text = qa["question_en"].lower()
        options = qa["options_en"]
        
        # Đảm bảo options là danh sách chuỗi
        options = [str(opt) for opt in options]
        
        combined_text = question_text + " " + " ".join(options).lower()
        doc = nlp(combined_text)
        combined_keywords = ' '.join([ent.text for ent in doc.ents])
        if not combined_keywords:
            stop_words = set(stopwords.words('english'))
            tokens = word_tokenize(combined_text)
            combined_keywords = ' '.join([word for word in tokens if word not in stop_words])
        
        question_embedding = model.encode(combined_keywords, convert_to_tensor=True, show_progress_bar=False)
        
        similarity = util.cos_sim(query_embedding, question_embedding).item()
        
        if similarity > threshold:
            matched_qids.append((qid, qa["question_en"], options))
    
    if not matched_qids:
        logger.info(f"No qid inferred for query_content: {query_content}")
        return [(None, None, None)]
    
    matched_qids.sort(key=lambda x: util.cos_sim(
        model.encode(query_keywords, convert_to_tensor=True, show_progress_bar=False),
        model.encode(x[1] + " " + " ".join(x[2]).lower(), convert_to_tensor=True, show_progress_bar=False)
    ).item(), reverse=True)
    
    return matched_qids

class MediqaDataset(Dataset):
    def __init__(self, data_dir, query_file, closed_qa_file, mode='train', transform=None):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, 'images')
        self.mask_dir = os.path.join(data_dir, 'masks', mode)
        self.mode = mode
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        self.sentence_model = SentenceTransformer('all-mpnet-base-v2')

        with open(query_file, 'r') as f:
            self.queries = json.load(f)
        
        with open(closed_qa_file, 'r') as f:
            self.closed_qa_dict = json.load(f)
        
        self.image_files = []
        self.masks = []
        self.qa_data = []
        
        # Tắt tqdm trong chế độ test để chỉ hiển thị một thanh tiến trình
        disable_tqdm = (self.mode == 'test')
        with tqdm(total=len(self.queries), desc="Processing queries", unit="query", disable=disable_tqdm) as pbar:
            for query in self.queries:
                encounter_id = query['encounter_id']
                # Kết hợp query_title_en và query_content_en
                query_content = (query.get('query_title_en', '') + " " + query.get('query_content_en', '')).lower().strip()
                
                image_ids = []
                if mode == 'train':
                    for img_file in os.listdir(self.image_dir):
                        if img_file.startswith(f'IMG_{encounter_id}_') and (img_file.endswith('.png') or img_file.endswith('.jpg')):
                            img_id = img_file.replace(f'IMG_{encounter_id}_', '').rsplit('.', 1)[0]
                            image_ids.append(img_id)
                else:
                    image_ids = query.get('image_ids', query.get('image_id', []))
                    if isinstance(image_ids, str):
                        image_ids = [image_ids]
                    image_ids = [img_id.replace(f'IMG_{encounter_id}_', '').rsplit('.', 1)[0] if img_id.startswith(f'IMG_{encounter_id}_') else img_id for img_id in image_ids]
                
                matched_qids = infer_qid(query_content, self.closed_qa_dict, self.sentence_model)
                
                for img_id in image_ids:
                    img_path = os.path.join(self.image_dir, f'IMG_{encounter_id}_{img_id}.png')
                    if not os.path.exists(img_path):
                        img_path = os.path.join(self.image_dir, f'IMG_{encounter_id}_{img_id}.jpg')
                    
                    try:
                        Image.open(img_path).convert('RGB')
                        mask_path = None
                        if mode == 'train':
                            for suffix in ['_mask_ann0.tiff', '_mask_ann1.tiff', '_mask_ann2.tiff', '_mask_ann3.tiff']:
                                temp_path = os.path.join(self.mask_dir, f'IMG_{encounter_id}_{img_id}{suffix}')
                                if os.path.exists(temp_path):
                                    mask_path = temp_path
                                    break
                            if mask_path:
                                mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                                if mask_img is None:
                                    logger.info(f"Invalid mask {mask_path}")
                                    continue
                            else:
                                logger.info(f"No mask found for {img_path}")
                                continue
                    except Exception as e:
                        logger.info(f"Failed to load image {img_path}: {e}")
                        continue
                    
                    if mode == 'train':
                        if os.path.exists(img_path) and mask_path:
                            self.image_files.append(img_path)
                            self.masks.append(mask_path)
                    else:
                        if os.path.exists(img_path):
                            self.image_files.append(img_path)
                    
                    for qid, question_text, options in matched_qids:
                        if qid is None:
                            logger.info(f"No qid inferred for encounter_id: {encounter_id}, image_id: {img_id}")
                            qid = ''
                            question_text = ''
                            options = []  # Đảm bảo options là danh sách rỗng
                        else:
                            # Đảm bảo options là danh sách chuỗi
                            options = [str(opt) for opt in options]
                        
                        self.qa_data.append({
                            'encounter_id': encounter_id,
                            'image_id': img_id,
                            'query': query_content,
                            'qid': qid,
                            'options': options,
                            'question_text': question_text
                        })
                
                pbar.update(1)
        
        logger.info(f"Total images in dataset: {len(self.image_files)}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.info(f"Error loading image {img_path}: {e}")
            raise ValueError(f"Invalid image at index {idx}")

        qa_info = self.qa_data[idx]
        query = qa_info['query']
        qid = qa_info['qid']
        options = qa_info['options'] or []
        question_text = qa_info['question_text'] or ""

        transformed_image = self.transform(image)
        
        try:
            inputs = self.processor(images=image, text=f"Question: {question_text}\nContext: {query}\nOptions: {', '.join(options)}", return_tensors="pt", do_rescale=False)
        except Exception as e:
            logger.info(f"Error processing image {img_path} with Blip2Processor: {e}")
            raise ValueError(f"Invalid Blip2 processing at index {idx}")

        mask = torch.zeros((1, 256, 256))  # Giá trị mặc định
        if self.mode == 'train':
            try:
                mask_img = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
                if mask_img is None:
                    logger.info(f"Invalid mask {self.masks[idx]}")
                    raise ValueError(f"Invalid mask at index {idx}")
                mask = mask_img / 255.0
                mask = Image.fromarray(mask)
                if self.transform:
                    mask = self.transform(mask)
                    mask = mask.squeeze()
                    mask = mask.unsqueeze(0)
            except Exception as e:
                logger.info(f"Error loading mask {self.masks[idx]}: {e}")
                raise ValueError(f"Invalid mask at index {idx}")

        return {
            'image': transformed_image,
            'prompt': f"Question: {question_text}\nContext: {query}\nOptions: {', '.join(options)}",
            'qid': qid,
            'options': options if options else [],  # Đảm bảo options không rỗng
            'mask': mask,
            'encounter_id': qa_info['encounter_id'],
            'image_id': qa_info['image_id']
        }