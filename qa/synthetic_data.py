import json
import os
import random

def generate_synthetic_data(output_file, num_samples=100, image_dir='/kaggle/input/mediqa-data/mediqa-data/images/'):
    diseases = ['folliculitis', 'pompholyx', 'psoriasis', 'vitiligo', 'syphilitic rash']
    locations = ['head', 'neck', 'arm', 'thigh', 'chest', 'back', 'palm']
    time_units = ['hour', 'day', 'week', 'month', 'year']
    sizes = ['thumb', 'palm', 'large']
    descriptions = ['bumpy', 'flat', 'sunken', 'thick', 'thin', 'warty', 'crust', 'scab', 'weeping']
    colors = ['pink', 'red', 'brown', 'blue', 'purple', 'black', 'white']
    textures = ['smooth', 'rough']
    
    valid_images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
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
        encounter_id = f"SYN{i:05d}"
        query_title = f"Is this {disease}?"
        query_content = (f"I have {description} {color} {size} lesions on my {location} for {random.randint(1, 12)} {time_unit}s. "
                         f"It is {texture} and {'itchy' if random.random() > 0.5 else 'not itchy'}.")
        response = f"Based on the description, it could be {disease}. Consult a dermatologist for {random.choice(['topical treatment', 'biopsy', 'medication'])}."
        
        closed_qa = {
            'CQID010-001': random.randint(0, 2) if random.random() > 0.2 else 3,
            'CQID011-001': locations.index(location) if location != 'palm' else 6,
            'CQID011-002': 7, 'CQID011-003': 7, 'CQID011-004': 7, 'CQID011-005': 7, 'CQID011-006': 7,
            'CQID012-001': sizes.index(size) if size != 'large' else 2,
            'CQID012-002': 3, 'CQID012-003': 3, 'CQID012-004': 3, 'CQID012-005': 3, 'CQID012-006': 3,
            'CQID015-001': time_units.index(time_unit),
            'CQID020-001': descriptions.index(description),
            'CQID020-002': 9, 'CQID020-003': 9, 'CQID020-004': 9, 'CQID020-005': 9, 'CQID020-006': 9,
            'CQID020-007': 9, 'CQID020-008': 9, 'CQID020-009': 9,
            'CQID025-001': 0 if 'itchy' in query_content else 1,
            'CQID034-001': colors.index(color) + 1,  # +1 vì 'normal' là 0
            'CQID035-001': random.randint(0, 1),
            'CQID036-001': textures.index(texture)
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