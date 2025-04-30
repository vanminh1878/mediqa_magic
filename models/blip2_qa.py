import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration, T5ForConditionalGeneration, T5Tokenizer
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util
import spacy
import logging
import re

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BLIP2QA:
    def __init__(self, model_name="Salesforce/blip2-flan-t5-xl", t5_model_name="t5-small", device="cuda"):
        self.device = device
        # Khởi tạo BLIP2
        self.blip_processor = Blip2Processor.from_pretrained(model_name)
        self.blip_model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
        # Khởi tạo T5
        self.t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
        self.t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name).to(device)
        # Khởi tạo KeyBERT và SentenceTransformer
        self.keybert_model = KeyBERT(model='all-mpnet-base-v2')
        self.sentence_model = SentenceTransformer('all-mpnet-base-v2')
        self.nlp = spacy.load("en_core_sci_sm")
        torch.cuda.empty_cache()
        logger.info("BLIP2QA and T5 initialized successfully")

    def answer_question(self, image, prompt, options):
        if not options:
            logger.warning("No options provided")
            return -1
        
        options = [str(opt) for opt in options if opt]
        
        # Cách 1: Sử dụng BLIP2 để sinh câu trả lời
        if image is not None:
            inputs = self.blip_processor(images=image, text=prompt, return_tensors="pt", do_rescale=False).to(self.device, torch.float16)
        else:
            inputs = self.blip_processor(text=prompt, return_tensors="pt").to(self.device, torch.float16)
        
        try:
            outputs = self.blip_model.generate(**inputs, max_length=100)
            answer = self.blip_processor.decode(outputs[0], skip_special_tokens=True).strip()
            logger.info(f"BLIP2 Prompt: {prompt}")
            logger.info(f"BLIP2 Generated answer: {answer}")
        except Exception as e:
            logger.warning(f"Error generating BLIP2 answer: {e}")
            answer = ""

        # Nếu BLIP2 trả lời hợp lệ, ánh xạ sang options
        if answer:
            # Trích xuất từ khóa bằng KeyBERT
            keywords = self.keybert_model.extract_keywords(
                answer,
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                top_n=5,
                use_mmr=True,
                diversity=0.5
            )
            answer_keywords = ' '.join([kw[0] for kw in keywords])
            if not answer_keywords:
                doc = self.nlp(answer)
                answer_keywords = ' '.join([ent.text for ent in doc.ents])
                if not answer_keywords:
                    answer_keywords = answer
            
            # Tính độ tương đồng và luật từ khóa
            similarities = []
            for opt in options:
                opt_lower = opt.lower()
                # Trích xuất từ khóa từ option
                opt_keywords = self.keybert_model.extract_keywords(
                    opt,
                    keyphrase_ngram_range=(1, 2),
                    stop_words='english',
                    top_n=3,
                    use_mmr=True,
                    diversity=0.5
                )
                opt_keywords_text = ' '.join([kw[0] for kw in opt_keywords]) or opt_lower
                
                # Tính độ tương đồng ngữ nghĩa
                answer_emb = self.sentence_model.encode(answer_keywords, convert_to_tensor=True, show_progress_bar=False)
                opt_emb = self.sentence_model.encode(opt_keywords_text, convert_to_tensor=True, show_progress_bar=False)
                sim = util.cos_sim(answer_emb, opt_emb).item()
                
                # Luật từ khóa
                if opt_lower in answer.lower():
                    sim += 0.7  # Trọng số cao cho khớp trực tiếp
                opt_doc = self.nlp(opt)
                opt_ents = ' '.join([ent.text for ent in opt_doc.ents])
                if opt_ents and opt_ents in answer_keywords:
                    sim += 0.5  # Trọng số cho từ khóa y khoa
                
                similarities.append(sim)
            
            # Nếu độ tương đồng thấp, chọn "Not mentioned"
            if max(similarities) < 0.3:
                logger.info("Low similarities, checking for 'Not mentioned'")
                for i, opt in enumerate(options):
                    if "not mentioned" in opt.lower():
                        logger.info(f"Selected option (BLIP2): {opt} (index: {i})")
                        return i
            
            option_idx = similarities.index(max(similarities))
            logger.info(f"Options: {options}")
            logger.info(f"Similarities: {similarities}")
            logger.info(f"Selected option (BLIP2): {options[option_idx]} (index: {option_idx})")
            return option_idx

        # Cách 2: Dự phòng bằng T5 few-shot
        logger.info("Falling back to T5 few-shot due to empty or invalid BLIP2 answer")
        
        # Tạo prompt few-shot (dựa trên dữ liệu train mẫu)
        question_text = prompt.split('Question: ')[1].split('Context: ')[0].strip()
        context_text = prompt.split('Context: ')[1].split('Options: ')[0].strip()
        options_text = ', '.join([f'{i+1}. {opt}' for i, opt in enumerate(options)])
        few_shot_prompt = (
            "Example 1: Question: Where is the affected area? Context: A patient with pleural effusion is accompanied by a systemic rash, as seen in the picture (currently only the back picture is available). Options: 1. head, 2. neck, 3. upper extremities, 4. lower extremities, 5. chest/abdomen, 6. back, 7. other, 8. Not mentioned. Answer: 6.\n"
            "Example 2: Question: How long ago did the lesion appear? Context: It started 2 weeks ago. Options: 1. Less than 1 week, 2. 1-4 weeks, 3. More than 4 weeks, 4. Not mentioned. Answer: 2.\n"
            "Example 3: Question: How long ago did the lesion appear? Context: On the outside of the thigh, there is a small circle of lump. Approximately 2 months. Options: 1. Less than 1 week, 2. 1-4 weeks, 3. More than 4 weeks, 4. Not mentioned. Answer: 3.\n"
            f"Question: {question_text} Context: {context_text} Options: {options_text}. Answer: ?"
        )
        
        inputs = self.t5_tokenizer(few_shot_prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        try:
            outputs = self.t5_model.generate(**inputs, max_length=10)
            t5_answer = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            logger.info(f"T5 Few-shot prompt: {few_shot_prompt}")
            logger.info(f"T5 Answer: {t5_answer}")
            
            # Trích xuất số thứ tự từ câu trả lời
            match = re.search(r'\b([1-9])\b', t5_answer)
            if match:
                option_idx = int(match.group(1)) - 1
                if 0 <= option_idx < len(options):
                    logger.info(f"Selected option (T5): {options[option_idx]} (index: {option_idx})")
                    return option_idx
            
            # Nếu không trích xuất được số, chọn "Not mentioned"
            for i, opt in enumerate(options):
                if "not mentioned" in opt.lower():
                    logger.info(f"Selected option (T5 fallback): {opt} (index: {i})")
                    return i
            logger.warning("No valid option number or 'Not mentioned' found")
            return -1
        except Exception as e:
            logger.warning(f"Error in T5 few-shot: {e}")
            # Fallback cuối cùng: chọn "Not mentioned"
            for i, opt in enumerate(options):
                if "not mentioned" in opt.lower():
                    logger.info(f"Selected option (final fallback): {opt} (index: {i})")
                    return i
            return -1