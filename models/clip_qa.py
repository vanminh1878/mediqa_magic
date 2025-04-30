import torch
from transformers import CLIPProcessor, CLIPModel
import logging
import spacy

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CLIPQA:
    def __init__(self, model_name="openai/clip-vit-large-patch14", device="cuda"):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.nlp = spacy.load("en_core_sci_sm")
        logger.info("CLIPQA initialized successfully")

    def answer_question(self, image, prompt, options, question_index):
        if not options:
            logger.warning("No options provided")
            return -1
        
        options = [str(opt) for opt in options if opt]
        
        # Chuẩn bị inputs
        text_inputs = [prompt + f"\nAnswer: {opt}" for opt in options]
        inputs = self.processor(
            text=text_inputs,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Dự đoán
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image  # [1, num_options]
        
        # Tính xác suất
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
        
        # Xử lý câu hỏi lặp
        if question_index > 1 and any(qid in prompt for qid in ['CQID011', 'CQID012', 'CQID020']):
            not_mentioned_idx = next((i for i, opt in enumerate(options) if "not mentioned" in opt.lower()), -1)
            if not_mentioned_idx != -1 and max(probs) < 0.4:
                logger.info(f"Low confidence for repeated question, selecting 'Not mentioned' (index: {not_mentioned_idx})")
                return not_mentioned_idx
        
        # Chọn option có xác suất cao nhất
        option_idx = probs.argmax()
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Options: {options}")
        logger.info(f"Probabilities: {probs}")
        logger.info(f"Selected option: {options[option_idx]} (index: {option_idx})")
        return option_idx