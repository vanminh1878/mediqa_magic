import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
from sentence_transformers import SentenceTransformer, util
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BertVQA:
    def __init__(self, bert_model_name="sentence-transformers/all-MiniLM-L6-v2", 
                 vqa_model_name="Salesforce/blip-vqa-base", device="cuda"):
        self.device = device
        
        # Initialize Sentence-Transformers for question mapping
        try:
            self.bert_model = SentenceTransformer(bert_model_name).to(device)
            logger.info("Sentence-Transformers model initialized")
        except Exception as e:
            logger.error(f"Error loading Sentence-Transformers: {e}")
            raise
        
        # Initialize BLIP for VQA
        try:
            logger.info(f"Loading BLIP processor for {vqa_model_name}")
            self.vqa_processor = BlipProcessor.from_pretrained(vqa_model_name)
            logger.info("BLIP processor loaded successfully")
        except Exception as e:
            logger.error(f"Error loading BLIP processor: {e}")
            raise
        
        try:
            logger.info(f"Loading BLIP model for {vqa_model_name}")
            self.vqa_model = BlipForQuestionAnswering.from_pretrained(vqa_model_name, torch_dtype=torch.float16).to(device)
            self.vqa_model.eval()
            logger.info("BLIP model initialized")
        except Exception as e:
            logger.error(f"Error loading BLIP model: {e}")
            raise

    def map_query_to_question(self, query, closed_qa_dict):
        """Map query to the most relevant question using cosine similarity."""
        query_embedding = self.bert_model.encode(query, convert_to_tensor=True, device=self.device)
        question_texts = [qa['question_en'] for qa in closed_qa_dict]
        question_embeddings = self.bert_model.encode(question_texts, convert_to_tensor=True, device=self.device)
        similarities = util.cos_sim(query_embedding, question_embeddings)[0]
        best_idx = similarities.argmax().item()
        mapped_qa = closed_qa_dict[best_idx]
        logger.info(f"Mapped query: {query[:50]}... to question: {mapped_qa['question_en']}")
        return mapped_qa

    def answer_question(self, image, query, closed_qa_dict, question_index):
        """Answer the question by mapping query to question and using VQA."""
        if not closed_qa_dict:
            logger.warning("No closed QA definitions provided")
            return -1
        
        # Map query to the most relevant question
        mapped_qa = self.map_query_to_question(query, closed_qa_dict)
        qid = mapped_qa['qid']
        question_text = mapped_qa['question_en']
        options = [str(opt) for opt in mapped_qa['options_en'] if opt]
        
        if not options:
            logger.warning(f"No valid options for mapped question: {qid}")
            return -1
        
        # Prepare prompt for VQA
        prompt = f"Question: {question_text}\nOptions: {', '.join([f'{i+1}. {opt}' for i, opt in enumerate(options)])}"
        
        # Process inputs for BLIP
        try:
            inputs = self.vqa_processor(images=image, text=prompt, return_tensors="pt").to(self.device, torch.float16)
        except Exception as e:
            logger.error(f"Error processing inputs for BLIP: {e}")
            return -1
        
        # Generate answer
        try:
            with torch.no_grad():
                outputs = self.vqa_model.generate(**inputs, max_length=50, num_beams=5)
        except Exception as e:
            logger.error(f"Error generating answer with BLIP: {e}")
            return -1
        
        # Decode generated text
        try:
            generated_text = self.vqa_processor.decode(outputs[0], skip_special_tokens=True).lower().strip()
        except Exception as e:
            logger.error(f"Error decoding BLIP output: {e}")
            return -1
        
        # Match generated text to options
        option_idx = -1
        for i, opt in enumerate(options):
            opt_lower = opt.lower().strip()
            # Kiểm tra cả option và số thứ tự
            if opt_lower in generated_text or str(i+1) in generated_text:
                option_idx = i
                break
            # Kiểm tra khớp gần đúng (cho các trường hợp BLIP trả lời không chính xác)
            for word in opt_lower.split():
                if word in generated_text and len(word) > 3:  # Chỉ tính từ dài hơn 3 ký tự
                    option_idx = i
                    break
        
        # Fallback for repeated questions
        if question_index > 1 and any(qid.startswith(prefix) for prefix in ['CQID011', 'CQID012', 'CQID020']):
            not_mentioned_idx = next((i for i, opt in enumerate(options) if "not mentioned" in opt.lower()), -1)
            if not_mentioned_idx != -1 and option_idx == -1:
                logger.info(f"Low confidence for repeated question, selecting 'Not mentioned' (index: {not_mentioned_idx})")
                return not_mentioned_idx
        
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Options: {options}")
        logger.info(f"Generated answer: {generated_text}")
        logger.info(f"Selected option: {options[option_idx] if option_idx >= 0 else 'None'} (index: {option_idx})")
        return option_idx