#!/bin/bash

# Cài đặt các công cụ biên dịch cần thiết
apt-get update
apt-get install -y build-essential libomp-dev

# Cài đặt các thư viện từ requirements.txt
pip install -r requirements.txt

# Cài đặt nmslib riêng để xử lý lỗi biên dịch
pip install nmslib==2.1.1 --no-binary nmslib

# Tải mô hình en_core_sci_sm từ scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz
mkdir -p /kaggle/working/output/