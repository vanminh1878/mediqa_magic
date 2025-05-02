import os
import json
import torch
from torch.utils.data import DataLoader
from data.process_data_qa import MediqaQADataset
from models.bert_vqa import BertVQA
from utils.helpers import save_qa_results
from torchvision import transforms
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Disable Hugging Face progress bars
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
# Disable sentence-transformers logging
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

def post_process_qa_results(qa_results_dict, closed_qa_dict):
    """Check consistency and handle repeated questions."""
    for enc_id, qid_dict in qa_results_dict.items():
        location_qids = sorted([qid for qid in qid_dict if qid.startswith('CQID011-')])
        if location_qids:
            # Find 'Not mentioned' index for CQID011
            not_mentioned_idx = len(closed_qa_dict[[qa['qid'] for qa in closed_qa_dict].index(location_qids[0])]['options_en']) - 1
            selected_locations = [qid for qid in location_qids if qid_dict[qid] != not_mentioned_idx]
            for i, qid in enumerate(location_qids):
                if i >= len(selected_locations):
                    qid_dict[qid] = not_mentioned_idx
                    logger.info(f"Set {qid} to 'Not mentioned' for encounter_id {enc_id}")
            
            # Update CQID010-001 if necessary
            if 'CQID010-001' in qid_dict and selected_locations:
                not_mentioned_idx_010 = len(closed_qa_dict[[qa['qid'] for qa in closed_qa_dict].index('CQID010-001')]['options_en']) - 1
                if qid_dict['CQID010-001'] == not_mentioned_idx_010:
                    qid_dict['CQID010-001'] = 1  # "limited area"
                    logger.info(f"Updated CQID010-001 to 'limited area' for encounter_id {enc_id}")
    
    return qa_results_dict

def run_qa_inference(data_dir, query_file, closed_qa_file, output_dir, mode='test'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize BertVQA
    try:
        bert_vqa = BertVQA(device=device)
        logger.info("BertVQA model initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing BertVQA: {e}")
        raise

    # Synchronize transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    try:
        dataset = MediqaQADataset(data_dir, query_file, closed_qa_file, mode=mode, transform=transform)
        logger.info(f"Dataset loaded with {len(dataset)} samples")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Load query file to get encounter_ids
    try:
        with open(query_file, 'r') as f:
            queries = json.load(f)
        all_encounter_ids = set(query.get('encounter_id', '') for query in queries if query.get('encounter_id'))
        logger.info(f"Total encounter_ids in {mode}.json: {len(all_encounter_ids)}")
    except Exception as e:
        logger.error(f"Error loading {mode}.json: {e}")
        raise

    # Load closed_qa_dict
    try:
        with open(closed_qa_file, 'r') as f:
            closed_qa_dict = json.load(f)
    except Exception as e:
        logger.error(f"Error loading closed_qa_file: {e}")
        raise

    qa_results_dict = {}
    processed_samples = 0
    skipped_samples = []

    # Iterate through dataloader
    with tqdm(total=len(dataloader), desc=f"Running QA inference ({mode})", unit="sample") as pbar:
        for batch in dataloader:
            try:
                encounter_id = batch['encounter_id'][0]
                image_id = batch['image_id'][0]
                qid = batch['qid'][0]
                query = batch['query'][0]
                options = batch['options'][0]
                image = batch['image'].to(device)
                question_index = batch['question_index'].item()

                # Answer question
                if qid and options:
                    try:
                        option_idx = bert_vqa.answer_question(image, query, closed_qa_dict, question_index)
                        if option_idx >= 0:
                            if encounter_id not in qa_results_dict:
                                qa_results_dict[encounter_id] = {}
                            qa_results_dict[encounter_id][qid] = option_idx
                            logger.debug(f"QA result for encounter_id: {encounter_id}, qid: {qid}, option_idx: {option_idx}")
                        else:
                            logger.warning(f"Invalid option_idx for encounter_id: {encounter_id}, qid: {qid}")
                            skipped_samples.append((encounter_id, image_id, f"Invalid option_idx for qid: {qid}"))
                    except Exception as e:
                        logger.warning(f"Error answering question for encounter_id: {encounter_id}, qid: {qid}, error: {e}")
                        skipped_samples.append((encounter_id, image_id, f"QA error: {e}"))
                else:
                    logger.warning(f"Skipping QA for encounter_id: {encounter_id}, qid: {qid}, query: {query}")
                    skipped_samples.append((encounter_id, image_id, f"Missing qid or options: qid={qid}, options={options}"))

                processed_samples += 1
                pbar.update(1)

                # Free memory
                del image
                torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Error processing encounter_id: {encounter_id}, error: {e}")
                skipped_samples.append((encounter_id, image_id, f"Processing error: {e}"))
                pbar.update(1)
                continue

    # Handle missing encounter_ids
    missing_encounter_ids = all_encounter_ids - set(qa_results_dict.keys())
    for enc_id in missing_encounter_ids:
        logger.warning(f"Missing encounter_id in results: {enc_id}")
        qa_results_dict[enc_id] = {}
        skipped_samples.append((enc_id, "", "Missing in dataset"))

    # Post-processing
    qa_results_dict = post_process_qa_results(qa_results_dict, closed_qa_dict)

    # Convert to list
    qa_results = [
        {"encounter_id": enc_id, **qid_dict}
        for enc_id, qid_dict in qa_results_dict.items()
    ]

    # Save results
    output_file = os.path.join(output_dir, f'data_cvqa_sys_{mode}.json')
    try:
        logger.info(f"QA results before saving: {qa_results[:5]}...")  # Log 5 results đầu tiên
        save_qa_results(qa_results, output_file)
        logger.info(f"QA results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving QA results: {e}")
        raise

    # Report
    logger.info(f"Processed samples: {processed_samples}")
    if skipped_samples:
        logger.warning(f"Skipped samples: {len(skipped_samples)}")
        for enc_id, img_id, err in skipped_samples:
            logger.warning(f"Skipped encounter_id: {enc_id}, image_id: {img_id}, error: {err}")

if __name__ == "__main__":
    data_dir = "/kaggle/input/mediqa-data/mediqa-data/"
    closed_qa_file = "/kaggle/input/mediqa-data/mediqa-data/closedquestions_definitions_imageclef2025.json"
    output_dir = "/kaggle/working/output/"

    # Run for test
    query_file_test = "/kaggle/input/mediqa-data/mediqa-data/test.json"
    run_qa_inference(data_dir, query_file_test, closed_qa_file, output_dir, mode='test')

    # Run for valid
    query_file_valid = "/kaggle/input/mediqa-data/mediqa-data/valid.json"
    run_qa_inference(data_dir, query_file_valid, closed_qa_file, output_dir, mode='valid')