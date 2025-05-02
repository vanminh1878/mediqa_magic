import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BertVQA:
    def __init__(self, bert_model_name="sentence-transformers/all-MiniLM-L6-v2", 
                 vqa_model_name="Salesforce/blip2-flan-t5-xl", device="cuda"):
        self.device = device
        
        # Initialize Sentence-Transformers for question mapping
        self.bert_model = SentenceTransformer(bert_model_name).to(device)
        logger.info("Sentence-Transformers model initialized")
        
        # Initialize BLIP-2 for VQA
        self.vqa_processor = AutoProcessor.from_pretrained(vqa_model_name)
        self.vqa_model = Blip2ForConditionalGeneration.from_pretrained(vqa_model_name, torch_dtype=torch.float16).to(device)
        self.vqa_model.eval()
        logger.info("BLIP-2 VQA model initialized")

    def map_query_to_question(self, query, closed_qa_dict):
        """Map query to the most relevant question using cosine similarity."""
        # Encode query
        query_embedding = self.bert_model.encode(query, convert_to_tensor=True, device=self.device)
        
        # Encode all questions
        question_texts = [qa['question_en'] for qa in closed_qa_dict]
        question_embeddings = self.bert_model.encode(question_texts, convert_to_tensor=True, device=self.device)
        
        # Compute cosine similarity
        similarities = util.cos_sim(query_embedding, question_embeddings)[0]
        best_idx = similarities.argmax().item()
        
        # Get the mapped question
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
            logger.warning("No valid options for mapped question")
            return -1
        
        # Prepare prompt for VQA
        prompt = f"Question: {question_text}\nOptions: {', '.join([f'{i+1}. {opt}' for i, opt in enumerate(options)])}"
        
        # Process inputs for BLIP-2
        inputs = self.vqa_processor(images=image, text=prompt, return_tensors="pt").to(self.device, torch.float16)
        
        # Generate answer
        with torch.no_grad():
            outputs = self.vqa_model.generate(**inputs)
        
        # Decode generated text
        generated_text = self.vqa_processor.decode(outputs[0], skip_special_tokens=True).lower()
        
        # Match generated text to options
        option_idx = -1
        for i, opt in enumerate(options):
            if opt.lower() in generated_text or str(i+1) in generated_text:
                option_idx = i
                break
        
        # Fallback for repeated questions (e.g., CQID011-002)
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