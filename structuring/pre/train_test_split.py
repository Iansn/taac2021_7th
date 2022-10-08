import random as rnd
import os
import json
import argparse
rnd.seed(1000)
parse=argparse.ArgumentParser()
parse.add_argument('--feat_path',default=r'./temp_data/video_feat_train',type=str)
parse.add_argument('--anno_path',default=r'./data/algo-2021/dataset/structuring/GroundTruth/train5k.txt',type=str)
parse.add_argument('--video_train_path',default=r'./temp_data/train.txt',type=str)
parse.add_argument('--video_test_path',default='./temp_data/test.txt',type=str)
parse.add_argument('--train_path',default='./temp_data/train_seg.txt',type=str)
parse.add_argument('--test_path',default='./temp_data/test_seg.txt',type=str)
args=parse.parse_args()

anno_infos=json.load(open(args.anno_path,'r',encoding='utf-8'))
filter_names=[]
for name in anno_infos:
    anno_info=anno_infos[name]['annotations']
    flag=0
    for segment in anno_info:
        start=segment['segment'][0]
        end=segment['segment'][1]
        if start>=end:
            flag=1
            filter_names.append(name[:-4])
            break
writer_train=open(args.train_path,'w')
writer_test=open(args.test_path,'w')
video_map={}
for video_name in os.listdir(args.feat_path):
    name=video_name.split('#')[0]

    video_map[name]=video_name
f=open(args.video_train_path,'r')
train_names=[]
for line in f:
    line=line.replace('\n','')
    line=line.split('#')[0]
    if line.endswith('.mp4'):
        line=line[:-4]
    if line not in train_names:
        if line not in filter_names:
            train_names.append(line)
f.close()
for name in train_names:

    writer_train.write(video_map[name]+'\n')
f=open(args.video_test_path,'r')
test_names=[]
for line in f:
    line = line.replace('\n', '')
    line=line.split('#')[0]
    if line.endswith('.mp4'):
        line=line[:-4]
    if line not in test_names:
        if line not in filter_names:
            test_names.append(line)
f.close()
for name in test_names:
    writer_test.write(video_map[name]+'\n')
writer_train.close()
writer_test.close()
