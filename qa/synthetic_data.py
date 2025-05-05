import json
import os
import random

def generate_synthetic_data(output_file, num_samples=2500, image_dir='/kaggle/input/mediqa-data/mediqa-data/images/'):
    diseases = ['folliculitis', 'pompholyx', 'psoriasis', 'vitiligo', 'syphilitic rash', 'eczema', 'acne', 'melanoma', 'urticaria', 'parapsoriasis']
    locations = ['head', 'neck', 'upper extremities', 'lower extremities', 'chest', 'abdomen', 'back', 'palm']
    time_units = ['hour', 'day', 'week', 'month', 'year']
    sizes = ['thumb nail', 'palm', 'large area']
    descriptions = ['raised', 'flat', 'sunken', 'thick', 'thin', 'warty', 'crust', 'scab', 'weeping', 'bumpy']
    colors = ['normal', 'pink', 'red', 'brown', 'blue', 'purple', 'black', 'white', 'combination', 'hyperpigmentation', 'hypopigmentation']
    textures = ['smooth', 'rough']
    extents = ['single spot', 'limited area', 'widespread']
    
    valid_images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    if not valid_images:
        raise ValueError(f"No valid images found in {image_dir}")
    
    synthetic_data = []
    for i in range(num_samples):
        disease = random.choice(diseases)
        num_locations = random.randint(1, 4)
        selected_locations = random.sample(locations, num_locations)
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
            f"I have {extent} {description} {color} lesions on my {', '.join(selected_locations)}, "
            f"about {size}, for {duration} {time_unit}{'s' if duration > 1 else ''}. "
            f"It is {texture} and {is_itchy}. "
            f"There {'is' if lesion_count == 'single' else 'are'} {lesion_count} lesion{'s' if lesion_count == 'multiple' else ''}. "
            f"Symptoms include {random.choice(['small red spots', 'stubborn lesions', 'slight numbness', 'cracking nails'])}."
        )
        response = f"Based on the description, it could be {disease}. Consult a dermatologist for diagnosis."
        
        closed_qa = {
            'CQID010-001': ['single spot', 'limited area', 'widespread', 'Not mentioned'].index(extent),
            'CQID011-001': ['head', 'neck', 'upper extremities', 'lower extremities', 'chest', 'abdomen', 'back', 'palm', 'Not mentioned'].index(selected_locations[0]),
            'CQID011-002': ['head', 'neck', 'upper extremities', 'lower extremities', 'chest', 'abdomen', 'back', 'palm', 'Not mentioned'].index(selected_locations[1]) if len(selected_locations) > 1 else 7,
            'CQID011-003': ['head', 'neck', 'upper extremities', 'lower extremities', 'chest', 'abdomen', 'back', 'palm', 'Not mentioned'].index(selected_locations[2]) if len(selected_locations) > 2 else 7,
            'CQID011-004': ['head', 'neck', 'upper extremities', 'lower extremities', 'chest', 'abdomen', 'back', 'palm', 'Not mentioned'].index(selected_locations[3]) if len(selected_locations) > 3 else 7,
            'CQID011-005': random.randint(0, 7) if random.random() > 0.5 else 7,
            'CQID011-006': random.randint(0, 7) if random.random() > 0.7 else 7,
            'CQID012-001': ['thumb nail', 'palm', 'large area', 'Not mentioned'].index(size),
            'CQID012-002': ['thumb nail', 'palm', 'large area', 'Not mentioned'].index(random.choice(sizes)) if random.random() > 0.1 else 3,
            'CQID012-003': ['thumb nail', 'palm', 'large area', 'Not mentioned'].index(random.choice(sizes)) if random.random() > 0.2 else 3,
            'CQID012-004': ['thumb nail', 'palm', 'large area', 'Not mentioned'].index(random.choice(sizes)) if random.random() > 0.3 else 3,
            'CQID012-005': ['thumb nail', 'palm', 'large area', 'Not mentioned'].index(random.choice(sizes)) if random.random() > 0.4 else 3,
            'CQID012-006': ['thumb nail', 'palm', 'large area', 'Not mentioned'].index(random.choice(sizes)) if random.random() > 0.2 else 3,
            'CQID015-001': ['within hours', 'within days', 'within weeks', 'within months', 'over a year', 'multiple years', 'Not mentioned'].index(
                {'hour': 'within hours', 'day': 'within days', 'week': 'within weeks', 'month': 'within months', 'year': 'multiple years'}[time_unit]
            ),
            'CQID020-001': ['raised', 'flat', 'sunken', 'thick', 'thin', 'warty', 'crust', 'scab', 'weeping', 'bumpy', 'Not mentioned'].index(description),
            'CQID020-002': ['raised', 'flat', 'sunken', 'thick', 'thin', 'warty', 'crust', 'scab', 'weeping', 'bumpy', 'Not mentioned'].index(random.choice(descriptions)) if random.random() > 0.1 else 9,
            'CQID020-003': ['raised', 'flat', 'sunken', 'thick', 'thin', 'warty', 'crust', 'scab', 'weeping', 'bumpy', 'Not mentioned'].index(random.choice(descriptions)) if random.random() > 0.2 else 9,
            'CQID020-004': ['raised', 'flat', 'sunken', 'thick', 'thin', 'warty', 'crust', 'scab', 'weeping', 'bumpy', 'Not mentioned'].index(random.choice(descriptions)) if random.random() > 0.3 else 9,
            'CQID020-005': ['raised', 'flat', 'sunken', 'thick', 'thin', 'warty', 'crust', 'scab', 'weeping', 'bumpy', 'Not mentioned'].index(random.choice(descriptions)) if random.random() > 0.4 else 9,
            'CQID020-006': ['raised', 'flat', 'sunken', 'thick', 'thin', 'warty', 'crust', 'scab', 'weeping', 'bumpy', 'Not mentioned'].index(random.choice(descriptions)) if random.random() > 0.5 else 9,
            'CQID020-007': ['raised', 'flat', 'sunken', 'thick', 'thin', 'warty', 'crust', 'scab', 'weeping', 'bumpy', 'Not mentioned'].index(random.choice(descriptions)) if random.random() > 0.6 else 9,
            'CQID020-008': ['raised', 'flat', 'sunken', 'thick', 'thin', 'warty', 'crust', 'scab', 'weeping', 'bumpy', 'Not mentioned'].index(random.choice(descriptions)) if random.random() > 0.7 else 9,
            'CQID020-009': ['raised', 'flat', 'sunken', 'thick', 'thin', 'warty', 'crust', 'scab', 'weeping', 'bumpy', 'Not mentioned'].index(random.choice(descriptions)) if random.random() > 0.8 else 9,
            'CQID025-001': 0 if is_itchy == 'itchy' else 1,
            'CQID034-001': ['normal', 'pink', 'red', 'brown', 'blue', 'purple', 'black', 'white', 'combination', 'hyperpigmentation', 'hypopigmentation', 'Not mentioned'].index(color),
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