import torch
import json
import os
import argparse
from transformers import BertModel, BertConfig
from transformers import BertTokenizer
import time
bert_pretrain_path=r'./tagging/pretrain_models/chinese-roberta-wwm-ext'

bert_config = BertConfig.from_pretrained(bert_pretrain_path)

bert = BertModel.from_pretrained(bert_pretrain_path, config=bert_config)
bert=bert.cuda()
bert.eval()
tokenizer=BertTokenizer.from_pretrained(bert_pretrain_path)
def extract_feat(args):
    cnt=0
    for name in os.listdir(args.text_dir):
        if not name.endswith('.txt'):
            continue

        cnt+=1

        s=time.time()
        text_info=json.load(open(os.path.join(args.text_dir,name),'r',encoding='utf-8'))
        ocr_text=text_info['video_ocr']
        asr_text=text_info['video_asr']


        if len(ocr_text)==0:
            state_dict={'feat':torch.zeros(768)}
            torch.save(state_dict, os.path.join(args.ocr_feat_path, name[:-4] + '.pth'))
        else:


            ocr_data = tokenizer(ocr_text, return_tensors='pt', max_length=args.max_words_len, truncation=True,
                                 padding='longest')
            ocr_ids = ocr_data['input_ids'].cuda()

            ocr_mask = ocr_data['attention_mask'].cuda()
            with torch.no_grad():
                ocr_feat = bert(ocr_ids, attention_mask=ocr_mask)[0].to('cpu')
                l = ocr_feat.size(1)
                ocr_feat = ocr_feat[0]

                ocr_feat = ocr_feat[1:l - 1]
                ocr_feat = torch.sum(ocr_feat, dim=0) / (l - 2)

                state_dict = {'feat': ocr_feat}
                torch.save(state_dict, os.path.join(args.ocr_feat_path, name[:-4] + '.pth'))
        if len(asr_text)==0:
            state_dict = {'feat': torch.zeros(768)}
            torch.save(state_dict, os.path.join(args.asr_feat_path, name[:-4] + '.pth'))
        else:
            asr_data = tokenizer(asr_text, return_tensors='pt', max_length=args.max_words_len, truncation=True,
                                 padding='longest')
            asr_ids = asr_data['input_ids'].cuda()

            # print(token_ocn_ids.size())
            asr_mask = asr_data['attention_mask'].cuda()
            with torch.no_grad():
                asr_feat = bert(asr_ids, attention_mask=asr_mask)[0].to('cpu')
                l = asr_feat.size(1)
                asr_feat = asr_feat[0]

                asr_feat = asr_feat[1:l - 1]
                asr_feat = torch.sum(asr_feat, dim=0) / (l - 2)
                state_dict = {'feat': asr_feat}
                torch.save(state_dict, os.path.join(args.asr_feat_path, name[:-4] + '.pth'))


        e=time.time()
        print(name,cnt,e-s)


if __name__=='__main__':

    cnt = 0
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_dir', default="../dataset/videos/train_5k", type=str)

    parser.add_argument('--ocr_feat_path', default='', type=str)
    parser.add_argument('--asr_feat_path', default='', type=str)

    parser.add_argument('--max_words_len',default=512,type=int)
    args = parser.parse_args()
    if not os.path.exists(args.ocr_feat_path):
        os.makedirs(args.ocr_feat_path)
    if not os.path.exists(args.asr_feat_path):
        os.makedirs(args.asr_feat_path)
    extract_feat(args)