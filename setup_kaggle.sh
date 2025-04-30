#!/bin/bash
pip install -r requirements.txt
# Tải mô hình en_core_sci_sm
python -m spacy download en_core_sci_sm
mkdir -p /kaggle/working/output/