import torch
from transformers import Blip2ForConditionalGeneration, Blip2Processor
from difflib import SequenceMatcher

class BLIP2QA:
    def __init__(self, model_name="Salesforce/blip2-flan-t5-xl", device="cuda"):
        self.device = device
        self.processor = Blip2Processor.from_pretrained(model_name)
        torch.cuda.empty_cache()  # Clear GPU memory
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    
    def answer_question(self, image, prompt, options):
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device, torch.float16)
        outputs = self.model.generate(**inputs, max_length=50)
        answer = self.processor.decode(outputs[0], skip_special_tokens=True).strip()
        
        if options:
            similarities = [SequenceMatcher(None, answer.lower(), opt.lower()).ratio() for opt in options]
            option_idx = similarities.index(max(similarities))
            return option_idx
        return -1