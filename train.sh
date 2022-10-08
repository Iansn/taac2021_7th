#!/usr/bin/env bash
python ./tagging/pre/train_test_split.py
python ./tagging/pre_train.py
python ./tagging/pre/extract_feat.py --video_dir ./data/algo-2021/dataset/videos/video_5k/train_5k --model_path ./tagging/model/model_pre/last.pth --write_path ./temp_data/video_feat_train
python ./tagging/pre/extract_ocr_asr_feat.py --text_dir ./data/algo-2021/dataset/tagging/tagging_dataset_train_5k/text_txt/tagging --asr_feat_path ./temp_data/asr_feat_train --ocr_feat_path ./temp_data/ocr_feat_train
python ./structuring/pre/train_test_split.py
python ./structuring/train.py
python ./tagging/train_mutimod.py