import json
import random as rnd
import argparse
rnd.seed(2000)
parse=argparse.ArgumentParser()
parse.add_argument('--anno_path',default='./data/algo-2021/dataset/structuring/GroundTruth/train5k.txt',type=str)
parse.add_argument('--train_path',default='./temp_data/train.txt')
parse.add_argument('--test_path',default='./temp_data/test.txt')
args=parse.parse_args()
train_writer=open(args.train_path,'w')
test_writer=open(args.test_path,'w')
annos=json.load(open(args.anno_path,'r',encoding='utf-8'))
train_labels={}
test_labels={}
for name in annos:
    r=rnd.random()
    if r>1.0:
        anno=annos[name]['annotations']
        for segment in anno:
            labels=segment['labels']
            for label in labels:
                if label not in test_labels:
                    test_labels[label]=1
                else:
                    test_labels[label]+=1
        test_writer.write(name+'\n')
    else:
        anno=annos[name]['annotations']
        for segment in anno:
            labels=segment['labels']
            for label in labels:
                if label not in train_labels:
                    train_labels[label]=1
                else:
                    train_labels[label]+=1
        train_writer.write(name+'\n')
for name in train_labels:
    c1=train_labels[name]
    c2=0
    if name in test_labels:
        c2=test_labels[name]
    print(name,c1,c2)
train_writer.close()
test_writer.close()
