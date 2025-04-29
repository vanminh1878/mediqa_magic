import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import spacy
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class BLIP2QA:
    def __init__(self, model_name="Salesforce/blip2-flan-t5-xl", device="cuda"):
        self.device = device
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
        self.sentence_model = SentenceTransformer('all-mpnet-base-v2')
        self.nlp = spacy.load("en_core_sci_sm")
        torch.cuda.empty_cache()  # Clear GPU memory
    
    def answer_question(self, image, prompt, options):
        if not options:
            logger.info("No options provided, returning -1")
            return -1
        
        # Đảm bảo options là danh sách chuỗi
        options = [str(opt) for opt in options if opt]
        
        # Xử lý hình ảnh và prompt
        inputs = self.processor(images=image, text=prompt, return_tensors="pt", do_rescale=False).to(self.device, torch.float16)
        try:
            outputs = self.model.generate(**inputs, max_length=50)
            answer = self.processor.decode(outputs[0], skip_special_tokens=True).strip()
        except Exception as e:
            logger.info(f"Error generating answer: {e}")
            return -1
        
        if not answer:
            logger.info("Empty answer generated, returning -1")
            return -1
        
        logger.info(f"Generated answer: {answer}")
        
        # Trích xuất từ khóa y khoa từ câu trả lời
        doc = self.nlp(answer)
        answer_keywords = ' '.join([ent.text for ent in doc.ents])
        if not answer_keywords:
            answer_keywords = answer
        
        # Tính độ tương đồng ngữ nghĩa
        answer_embedding = self.sentence_model.encode(answer_keywords, convert_to_tensor=True, show_progress_bar=False)
        option_embeddings = self.sentence_model.encode(options, convert_to_tensor=True, show_progress_bar=False)
        similarities = util.cos_sim(answer_embedding, option_embeddings)[0]
        
        # Thêm trọng số cho từ khóa khớp trực tiếp
        for i, opt in enumerate(options):
            opt_doc = self.nlp(opt)
            opt_keywords = ' '.join([ent.text for ent in opt_doc.ents])
            if opt_keywords and opt_keywords in answer_keywords:
                similarities[i] += 0.2  # Tăng điểm nếu từ khóa y khoa khớp
            elif opt.lower() in answer.lower():
                similarities[i] += 0.1  # Tăng điểm nếu option xuất hiện trực tiếp trong câu trả lời
        
        # Chọn option có độ tương đồng cao nhất
        option_idx = similarities.argmax().item()
        logger.info(f"Similarities: {similarities.tolist()}, Selected option: {options[option_idx]} (index: {option_idx})")
        
        return option_idx