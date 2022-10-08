#!/usr/bin/env bash
python ./tagging/extract_feat.py --video_dir ./data/algo-2021/dataset/videos/test_5k_2nd --model_path ./tagging/model/model_pre/last.pth --write_path ./temp_data/video_feat_test
python ./tagging/extract_ocr_asr_feat.py --text_dir ./data/algo-2021/dataset/tagging/tagging_dataset_test_5k_2nd/text_txt/tagging --asr_feat_path ./temp_data/asr_feat_test --ocr_feat_path ./temp_data/ocr_feat_test
python ./structuring/inference.py
python ./tagging/inference_result.py