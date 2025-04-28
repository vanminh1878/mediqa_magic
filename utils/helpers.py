import os
import cv2
import json
import numpy as np

def save_mask(mask, encounter_id, image_id, output_dir):
    mask_path = os.path.join(output_dir, f'IMG_{encounter_id}_{image_id}_mask_sys.tiff')
    cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))

def save_qa_results(results, output_file):
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)