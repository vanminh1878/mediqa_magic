import json
import os
import random

def generate_synthetic_data(output_file, num_samples=100, image_dir='/kaggle/input/mediqa-data/mediqa-data/images/'):
    diseases = ['folliculitis', 'pompholyx', 'psoriasis', 'vitiligo', 'syphilitic rash']
    locations = ['thigh', 'palm', 'back', 'hand', 'face']
    
    # Lấy danh sách các tệp ảnh thực tế
    valid_images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    if not valid_images:
        raise ValueError(f"No valid images found in {image_dir}")
    
    synthetic_data = []
    for i in range(num_samples):
        disease = random.choice(diseases)
        location = random.choice(locations)
        encounter_id = f"SYN{i:05d}"
        query_title = f"Is this {disease}?"
        query_content = f"I have small {random.choice(['red spots', 'lumps', 'rashes'])} on my {location} for {random.randint(1, 12)} weeks."
        response = f"Based on the description, it could be {disease}. Consult a dermatologist for {random.choice(['topical treatment', 'biopsy', 'medication'])}."
        
        closed_qa = {'CQID011-001': locations.index(location) + 1} if location != 'face' else {'CQID011-001': 7}
        
        # Chọn ngẫu nhiên một ảnh thực tế
        image_id = random.choice(valid_images)
        
        synthetic_data.append({
            'encounter_id': encounter_id,
            'query_title_en': query_title,
            'query_content_en': query_content,
            'image_ids': [image_id],  # Sử dụng ảnh thực tế
            'responses': [{'content_en': response}],
            'closed_qa': closed_qa
        })
    
    with open(output_file, 'w') as f:
        json.dump(synthetic_data, f, indent=2)
    
    print(f"Synthetic data saved to {output_file}")

if __name__ == "__main__":
    generate_synthetic_data("/kaggle/working/synthetic_train.json")