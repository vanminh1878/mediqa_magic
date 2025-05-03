import json
import os
import random
import re

def generate_synthetic_data(output_file, num_samples=200, image_dir='/kaggle/input/mediqa-data/mediqa-data/images/'):
    diseases = ['folliculitis', 'pompholyx', 'psoriasis', 'vitiligo', 'syphilitic rash']
    locations = ['head', 'neck', 'arm', 'thigh', 'chest', 'back', 'palm']
    time_units = ['hour', 'day', 'week', 'month', 'year']
    sizes = ['thumb nail', 'palm', 'large area']
    descriptions = ['raised', 'flat', 'sunken', 'thick', 'thin', 'warty', 'crust', 'scab', 'weeping', 'blister']
    colors = ['pink', 'red', 'brown', 'blue', 'purple', 'black', 'white']
    textures = ['smooth', 'rough']
    extents = ['single spot', 'limited area', 'widespread']
    
    valid_images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    if not valid_images:
        raise ValueError(f"No valid images found in {image_dir}")
    
    synthetic_data = []
    for i in range(num_samples):
        disease = random.choice(diseases)
        location = random.choice(locations)
        time_unit = random.choice(time_units)
        size = random.choice(sizes)
        description = random.choice(descriptions)
        color = random.choice(colors)
        texture = random.choice(textures)
        extent = random.choice(extents)
        is_itchy = random.choice(['itchy', 'not itchy'])
        lesion_count = random.choice(['single', 'multiple'])
        
        encounter_id = f"SYN{i:05d}"
        query_title = f"Is this {disease}?"
        duration = random.randint(1, 12) if time_unit != 'year' else random.randint(1, 5)
        query_content = (
            f"I have {extent} {description} {color} lesions on my {location}, "
            f"about {size}, for {duration} {time_unit}{'s' if duration > 1 else ''}. "
            f"It is {texture} and {is_itchy}. "
            f"There {'is' if lesion_count == 'single' else 'are'} {lesion_count} lesion{'s' if lesion_count == 'multiple' else ''}."
        )
        response = f"Based on the description, it could be {disease}. Consult a dermatologist for {random.choice(['topical treatment', 'biopsy', 'medication'])}."
        
        closed_qa = {
            'CQID010-001': ['single spot', 'limited area', 'widespread', 'Not mentioned'].index(extent),
            'CQID011-001': ['head', 'neck', 'arm', 'thigh', 'chest', 'back', 'palm', 'Not mentioned'].index(location),
            'CQID011-002': 7, 'CQID011-003': 7, 'CQID011-004': 7, 'CQID011-005': 7, 'CQID011-006': 7,
            'CQID012-001': ['thumb nail', 'palm', 'large area', 'Not mentioned'].index(size),
            'CQID012-002': 3, 'CQID012-003': 3, 'CQID012-004': 3, 'CQID012-005': 3, 'CQID012-006': 3,
            'CQID015-001': ['hour', 'day', 'week', 'month', 'year', 'years', 'Not mentioned'].index(time_unit if time_unit != 'year' else 'years'),
            'CQID020-001': ['raised', 'flat', 'sunken', 'thick', 'thin', 'warty', 'crust', 'scab', 'weeping', 'blister', 'Not mentioned'].index(description),
            'CQID020-002': 9, 'CQID020-003': 9, 'CQID020-004': 9, 'CQID020-005': 9, 'CQID020-006': 9,
            'CQID020-007': 9, 'CQID020-008': 9, 'CQID020-009': 9,
            'CQID025-001': 0 if is_itchy == 'itchy' else 1,
            'CQID034-001': ['normal', 'pink', 'red', 'brown', 'blue', 'purple', 'black', 'white', 'combination', 'hyperpigmentation', 'hypopigmentation', 'Not mentioned'].index(color) + 1,
            'CQID035-001': ['single', 'multiple', 'Not mentioned'].index(lesion_count),
            'CQID036-001': ['smooth', 'rough', 'Not mentioned'].index(texture)
        }
        
        image_id = random.choice(valid_images)
        
        synthetic_data.append({
            'encounter_id': encounter_id,
            'query_title_en': query_title,
            'query_content_en': query_content,
            'image_ids': [image_id],
            'responses': [{'content_en': response}],
            'closed_qa': closed_qa
        })
    
    with open(output_file, 'w') as f:
        json.dump(synthetic_data, f, indent=2)
    
    print(f"Synthetic data saved to {output_file}")

if __name__ == "__main__":
    generate_synthetic_data("/kaggle/working/synthetic_train.json")