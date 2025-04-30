import json
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_dummy_labels(input_file, output_file, closed_qa_file):
    with open(input_file, 'r') as f:
        queries = json.load(f)
    
    with open(closed_qa_file, 'r') as f:
        closed_qa = json.load(f)
    
    for query in queries:
        query_content = (query.get('query_title_en', '') + " " + query.get('query_content_en', '')).lower().strip()
        encounter_id = query['encounter_id']
        
        for qa in closed_qa:
            qid = qa['qid']
            options = qa['options_en']
            
            if qid == "CQID010-001":  # How much of the body is affected?
                if "systemic" in query_content or "widespread" in query_content:
                    query[qid] = options.index("widespread")  # 2
                elif "single" in query_content or "spot" in query_content:
                    query[qid] = options.index("single spot")  # 0
                else:
                    query[qid] = options.index("limited area")  # 1
            
            elif qid.startswith("CQID011-"):  # Where is the affected area?
                question_number = int(qid.split('-')[1])  # 001 -> 1, 002 -> 2, ...
                
                if question_number == 1:  # CQID011-001
                    if "back" in query_content:
                        query[qid] = options.index("back")  # 5
                    elif "foot" in query_content or "sole" in query_content:
                        query[qid] = options.index("lower extremities")  # 3
                    elif "hand" in query_content:
                        query[qid] = options.index("upper extremities")  # 2
                    elif "head" in query_content or "face" in query_content:
                        query[qid] = options.index("head")  # 0
                    else:
                        query[qid] = options.index("Not mentioned")  # 7
                else:  # CQID011-002, CQID011-003, ...
                    query[qid] = options.index("Not mentioned")  # 7
            
            # Thêm quy tắc cho các qid khác nếu cần (ví dụ: CQID012-001, ...)
            
    with open(output_file, 'w') as f:
        json.dump(queries, f, indent=4)
    
    logger.info(f"Generated labels for {input_file} and saved to {output_file}")

if __name__ == "__main__":
    # Tạo nhãn cho train
    generate_dummy_labels(
        "/kaggle/input/mediqa-data/mediqa-data/train_cvqa.json",
        "/kaggle/working/train_cvqa_labeled.json",
        "/kaggle/input/mediqa-data/mediqa-data/closedquestions_definitions_imageclef2025.json"
    )
    # Tạo nhãn cho valid
    generate_dummy_labels(
        "/kaggle/input/mediqa-data/mediqa-data/valid.json",
        "/kaggle/working/valid_cvqa_labeled.json",
        "/kaggle/input/mediqa-data/mediqa-data/closedquestions_definitions_imageclef2025.json"
    )