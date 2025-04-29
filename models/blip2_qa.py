import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util

class BLIP2QA:
    def __init__(self, model_name="Salesforce/blip2-flan-t5-xl", device="cuda"):
        self.device = device
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
        self.sentence_model = SentenceTransformer('all-mpnet-base-v2')
        torch.cuda.empty_cache()  # Clear GPU memory
    
    def answer_question(self, image, prompt, options):
        inputs = self.processor(images=image, text=prompt, return_tensors="pt", do_rescale=False).to(self.device, torch.float16)
        outputs = self.model.generate(**inputs, max_length=50)
        answer = self.processor.decode(outputs[0], skip_special_tokens=True).strip()
        
        if options:
            # Đảm bảo options là danh sách chuỗi
            options = [str(opt) for opt in options]
            # Tính độ tương đồng ngữ nghĩa
            answer_embedding = self.sentence_model.encode(answer, convert_to_tensor=True, show_progress_bar=False)
            option_embeddings = self.sentence_model.encode(options, convert_to_tensor=True, show_progress_bar=False)
            similarities = util.cos_sim(answer_embedding, option_embeddings)[0]
            option_idx = similarities.argmax().item()
            return option_idx
        return -1