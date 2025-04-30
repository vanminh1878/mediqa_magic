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
from keybert import KeyBERT
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util

# Tải tài nguyên NLTK
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Tải mô hình spacy
nlp = spacy.load("en_core_sci_sm")

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def infer_qid(query_content, closed_qa_dict, keybert_model, sentence_model, threshold=0.4):
    query_content = query_content.lower().strip()
    if not query_content:
        logger.info("Empty query_content, returning all questions")
        return [(qa["qid"], qa["question_en"], qa["options_en"]) for qa in closed_qa_dict]
    
    # Trích xuất từ khóa bằng KeyBERT
    keywords = keybert_model.extract_keywords(
        query_content,
        keyphrase_ngram_range=(1, 2),
        stop_words='english',
        top_n=5,
        use_mmr=True,
        diversity=0.5
    )
    query_keywords = ' '.join([kw[0] for kw in keywords])
    if not query_keywords:
        # Fallback: Dùng spacy
        doc = nlp(query_content)
        query_keywords = ' '.join([ent.text for ent in doc.ents])
        if not query_keywords:
            stop_words = set(stopwords.words('english'))
            tokens = word_tokenize(query_content)
            query_keywords = ' '.join([word for word in tokens if word not in stop_words])
    
    matched_qids = []
    for qa in closed_qa_dict:
        qid = qa["qid"]
        question_text = qa["question_en"].lower()
        options = qa["options_en"]
        options = [str(opt) for opt in options if opt]
        
        # Trích xuất từ khóa từ câu hỏi và options
        combined_text = question_text + " " + " ".join(options).lower()
        keywords = keybert_model.extract_keywords(
            combined_text,
            keyphrase_ngram_range=(1, 2),
            stop_words='english',
            top_n=5,
            use_mmr=True,
            diversity=0.5
        )
        combined_keywords = ' '.join([kw[0] for kw in keywords])
        if not combined_keywords:
            doc = nlp(combined_text)
            combined_keywords = ' '.join([ent.text for ent in doc.ents])
            if not combined_keywords:
                stop_words = set(stopwords.words('english'))
                tokens = word_tokenize(combined_text)
                combined_keywords = ' '.join([word for word in tokens if word not in stop_words])
        
        # So sánh từ khóa bằng độ tương đồng ngữ nghĩa
        query_embedding = sentence_model.encode(query_keywords, convert_to_tensor=True, show_progress_bar=False)
        question_embedding = sentence_model.encode(combined_keywords, convert_to_tensor=True, show_progress_bar=False)
        similarity = util.cos_sim(query_embedding, question_embedding).item()
        
        if similarity > threshold:
            matched_qids.append((qid, qa["question_en"], options))
    
    if not matched_qids:
        logger.info(f"No qid inferred for query_content: {query_content}, returning all questions")
        return [(qa["qid"], qa["question_en"], qa["options_en"]) for qa in closed_qa_dict]
    
    # Sắp xếp theo độ tương đồng
    matched_qids.sort(key=lambda x: util.cos_sim(
        sentence_model.encode(query_keywords, convert_to_tensor=True, show_progress_bar=False),
        sentence_model.encode(x[1] + " " + " ".join(x[2]).lower(), convert_to_tensor=True, show_progress_bar=False)
    ).item(), reverse=True)
    
    return matched_qids[:5]  # Giới hạn tối đa 5 qid để tránh dư thừa

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
        self.keybert_model = KeyBERT(model='all-mpnet-base-v2')
        self.sentence_model = SentenceTransformer('all-mpnet-base-v2')

        with open(query_file, 'r') as f:
            self.queries = json.load(f)
        
        with open(closed_qa_file, 'r') as f:
            self.closed_qa_dict = json.load(f)
        
        self.image_files = []
        self.masks = []
        self.qa_data = []
        self.skipped_samples = []
        
        disable_tqdm = (self.mode == 'test')
        with tqdm(total=len(self.queries), desc="Processing queries", unit="query", disable=disable_tqdm) as pbar:
            for query in self.queries:
                encounter_id = query.get('encounter_id', '')
                if not encounter_id:
                    logger.warning(f"Missing encounter_id in query: {query}")
                    self.skipped_samples.append((encounter_id, "Missing encounter_id"))
                    pbar.update(1)
                    continue
                
                query_content = (query.get('query_title_en', '') + " " + query.get('query_content_en', '')).lower().strip()
                if not query_content:
                    query_content = "skin issue"
                
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
                    image_ids = [
                        img_id.replace(f'IMG_{encounter_id}_', '').rsplit('.', 1)[0] 
                        if img_id.startswith(f'IMG_{encounter_id}_') else img_id 
                        for img_id in image_ids if img_id
                    ]
                
                matched_qids = infer_qid(query_content, self.closed_qa_dict, self.keybert_model, self.sentence_model)
                
                if not image_ids:
                    logger.warning(f"No valid image_ids for encounter_id: {encounter_id}")
                    self.skipped_samples.append((encounter_id, "No valid image_ids"))
                    for qid, question_text, options in matched_qids:
                        if qid is None:
                            continue
                        options = [str(opt) for opt in options if opt]
                        self.qa_data.append({
                            'encounter_id': encounter_id,
                            'image_id': '',
                            'query': query_content,
                            'qid': qid,
                            'options': options,
                            'question_text': question_text or ''
                        })
                    pbar.update(1)
                    continue
                
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
                                    logger.warning(f"Invalid mask {mask_path}")
                                    self.skipped_samples.append((encounter_id, f"Invalid mask for image_id: {img_id}"))
                                    continue
                            else:
                                logger.warning(f"No mask found for {img_path}")
                                self.skipped_samples.append((encounter_id, f"No mask for image_id: {img_id}"))
                                continue
                    except Exception as e:
                        logger.warning(f"Failed to load image {img_path}: {e}")
                        self.skipped_samples.append((encounter_id, f"Failed to load image_id: {img_id}, error: {e}"))
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
                            continue
                        options = [str(opt) for opt in options if opt]
                        self.qa_data.append({
                            'encounter_id': encounter_id,
                            'image_id': img_id,
                            'query': query_content,
                            'qid': qid,
                            'options': options,
                            'question_text': question_text or ''
                        })
                
                pbar.update(1)
        
        logger.info(f"Total images in dataset: {len(self.image_files)}")
        logger.info(f"Total QA entries: {len(self.qa_data)}")
        if self.skipped_samples:
            logger.warning(f"Skipped samples: {len(self.skipped_samples)}")
            for enc_id, reason in self.skipped_samples:
                logger.warning(f"Skipped encounter_id: {enc_id}, reason: {reason}")

    def __len__(self):
        return len(self.image_files) if self.image_files else len(self.qa_data)

    def __getitem__(self, idx):
        if not self.image_files:
            qa_info = self.qa_data[idx]
            return {
                'image': None,
                'prompt': f"Question: {qa_info['question_text']}\nContext: {qa_info['query']}\nOptions: {', '.join(qa_info['options'])}",
                'qid': qa_info['qid'],
                'options': qa_info['options'],
                'mask': torch.zeros((1, 256, 256)),
                'encounter_id': qa_info['encounter_id'],
                'image_id': qa_info['image_id']
            }
        
        img_path = self.image_files[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
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
            logger.error(f"Error processing image {img_path} with Blip2Processor: {e}")
            raise ValueError(f"Invalid Blip2 processing at index {idx}")

        mask = torch.zeros((1, 256, 256))
        if self.mode == 'train':
            try:
                mask_img = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
                if mask_img is None:
                    logger.error(f"Invalid mask {self.masks[idx]}")
                    raise ValueError(f"Invalid mask at index {idx}")
                mask = mask_img / 255.0
                mask = Image.fromarray(mask)
                if self.transform:
                    mask = self.transform(mask)
                    mask = mask.squeeze()
                    mask = mask.unsqueeze(0)
            except Exception as e:
                logger.error(f"Error loading mask {self.masks[idx]}: {e}")
                raise ValueError(f"Invalid mask at index {idx}")

        return {
            'image': transformed_image,
            'prompt': f"Question: {question_text}\nContext: {query}\nOptions: {', '.join(options)}",
            'qid': qid,
            'options': options,
            'mask': mask,
            'encounter_id': qa_info['encounter_id'],
            'image_id': qa_info['image_id']
        }