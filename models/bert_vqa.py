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
        
        # Initialize Sentence-Transformers
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
        logger.info(f"Mapped query: {query[:50]}... to question: {mapped_qa['question_en']}, qid: {mapped_qa['qid']}")
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
            logger.error(f"Error processing inputs for BLIP: qid={qid}, error={e}")
            return -1
        
        # Generate answer
        try:
            with torch.no_grad():
                outputs = self.vqa_model.generate(**inputs, max_length=50, num_beams=5, early_stopping=True)
        except Exception as e:
            logger.error(f"Error generating answer with BLIP: qid={qid}, error={e}")
            return -1
        
        # Decode generated text
        try:
            generated_text = self.vqa_processor.decode(outputs[0], skip_special_tokens=True).lower().strip()
        except Exception as e:
            logger.error(f"Error decoding BLIP output: qid={qid}, error={e}")
            return -1
        
        # Match generated text to options
        option_idx = -1
        for i, opt in enumerate(options):
            opt_lower = opt.lower().strip()
            # Kiểm tra khớp chính xác
            if opt_lower in generated_text or str(i+1) in generated_text:
                option_idx = i
                break
            # Kiểm tra khớp gần đúng dựa trên từ khóa
            opt_words = [w for w in opt_lower.split() if len(w) > 3]
            for word in opt_words:
                if word in generated_text:
                    option_idx = i
                    break
            if option_idx >= 0:
                break
        
        # Fallback for specific QIDs
        if option_idx == -1:
            # QID-specific keyword matching
            qid_keywords = {
                'CQID010-001': {
                    'single': 0, 'spot': 0, 'limited': 1, 'area': 1, 'widespread': 2, 'not mentioned': 3
                },
                'CQID011': {
                    'head': 0, 'neck': 1, 'upper': 2, 'lower': 3, 'chest': 4, 'abdomen': 4, 'back': 5, 'other': 6, 'not mentioned': 7
                },
                'CQID012': {
                    'thumb': 0, 'nail': 0, 'palm': 1, 'larger': 2, 'not mentioned': 3
                },
                'CQID015-001': {
                    'hour': 0, 'day': 1, 'week': 2, 'month': 3, 'year': 4, 'multiple': 5, 'not mentioned': 6
                },
                'CQID020': {
                    'raised': 0, 'bumpy': 0, 'flat': 1, 'sunken': 2, 'thick': 3, 'thin': 4, 'warty': 5, 'crust': 6, 'scab': 7, 'weeping': 8, 'not mentioned': 9
                },
                'CQID025-001': {
                    'yes': 0, 'no': 1, 'not mentioned': 2
                },
                'CQID034-001': {
                    'normal': 0, 'pink': 1, 'red': 2, 'brown': 3, 'blue': 4, 'purple': 5, 'black': 6, 'white': 7, 'combination': 8, 'hyperpigmentation': 9, 'hypopigmentation': 10, 'not mentioned': 11
                },
                'CQID035-001': {
                    'single': 0, 'multiple': 1, 'not mentioned': 2
                },
                'CQID036-001': {
                    'smooth': 0, 'rough': 1, 'not mentioned': 2
                }
            }
            for qid_prefix, keywords in qid_keywords.items():
                if qid.startswith(qid_prefix):
                    for keyword, idx in keywords.items():
                        if keyword in generated_text:
                            option_idx = idx
                            break
                    break
        
        # Fallback for repeated questions
        if question_index > 1 and any(qid.startswith(prefix) for prefix in ['CQID011', 'CQID012', 'CQID020']):
            not_mentioned_idx = next((i for i, opt in enumerate(options) if "not mentioned" in opt.lower()), -1)
            if not_mentioned_idx != -1 and option_idx == -1:
                logger.info(f"Low confidence for repeated question, selecting 'Not mentioned' (index={not_mentioned_idx}, qid={qid})")
                return not_mentioned_idx
        
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Options: {options}")
        logger.info(f"Generated answer: {generated_text}")
        logger.info(f"Selected option: {options[option_idx] if option_idx >= 0 else 'None'} (index={option_idx}, qid={qid})")
        return option_idx