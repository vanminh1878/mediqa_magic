import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import spacy
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class BLIP2QA:
    def __init__(self, model_name="Salesforce/blip2-flan-t5-xl", device="cuda"):
        self.device = device
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
        self.sentence_model = SentenceTransformer('all-mpnet-base-v2')
        self.nlp = spacy.load("en_core_sci_sm")
        torch.cuda.empty_cache()
    
    def answer_question(self, image, prompt, options):
        if not options:
            logger.info("No options provided")
            return -1
        
        options = [str(opt) for opt in options if opt]
        
        # Nếu không có hình ảnh, trả lời dựa trên văn bản
        if image is None:
            inputs = self.processor(text=prompt, return_tensors="pt").to(self.device, torch.float16)
        else:
            inputs = self.processor(images=image, text=prompt, return_tensors="pt", do_rescale=False).to(self.device, torch.float16)
        
        try:
            outputs = self.model.generate(**inputs, max_length=100)
            answer = self.processor.decode(outputs[0], skip_special_tokens=True).strip()
        except Exception as e:
            logger.warning(f"Error generating answer: {e}")
            return -1
        
        if not answer:
            logger.info("Empty answer generated")
            # Chọn "Not mentioned" nếu có
            for i, opt in enumerate(options):
                if "not mentioned" in opt.lower():
                    return i
            return -1
        
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Generated answer: {answer}")
        
        # Trích xuất từ khóa y khoa
        doc = self.nlp(answer)
        answer_keywords = ' '.join([ent.text for ent in doc.ents])
        if not answer_keywords:
            answer_keywords = answer
        
        # Tính độ tương đồng ngữ nghĩa
        answer_embedding = self.sentence_model.encode(answer_keywords, convert_to_tensor=True, show_progress_bar=False)
        option_embeddings = self.sentence_model.encode(options, convert_to_tensor=True, show_progress_bar=False)
        similarities = util.cos_sim(answer_embedding, option_embeddings)[0]
        
        # Luật từ khóa: ưu tiên option khớp trực tiếp
        for i, opt in enumerate(options):
            opt_lower = opt.lower()
            if opt_lower in answer.lower():
                similarities[i] += 0.5  # Trọng số cao cho khớp trực tiếp
            opt_doc = self.nlp(opt)
            opt_keywords = ' '.join([ent.text for ent in opt_doc.ents])
            if opt_keywords and opt_keywords in answer_keywords:
                similarities[i] += 0.3  # Trọng số cho từ khóa y khoa
        
        # Nếu tất cả độ tương đồng thấp, chọn "Not mentioned"
        if max(similarities) < 0.2:
            logger.info("Low similarities, selecting 'Not mentioned' if available")
            for i, opt in enumerate(options):
                if "not mentioned" in opt.lower():
                    return i
        
        option_idx = similarities.argmax().item()
        logger.info(f"Options: {options}")
        logger.info(f"Similarities: {similarities.tolist()}")
        logger.info(f"Selected option: {options[option_idx]} (index: {option_idx})")
        
        return option_idx