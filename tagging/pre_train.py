
from src.models.slowonly_net import slowonly_net

import torch
import numpy as np
import src.utils.transforms as transforms
from src.data.video_dataset_split import video_dataset
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler

import os
from src.utils.warmup_scheduler import WarmUpLR
import argparse
parse=argparse.ArgumentParser()
parse.add_argument('--video_list_train',default='./temp_data/train.txt',type=str)
parse.add_argument('--video_list_test',default='./temp_data/test.txt',type=str)
parse.add_argument('--video_dir',default='./data/algo-2021/dataset/videos/video_5k/train_5k',type=str)
parse.add_argument('--label_id_path',default='./data/algo-2021/dataset/label_id.txt',type=str)
parse.add_argument('--anno_path',default='./data/algo-2021/dataset/structuring/GroundTruth/train5k.txt',type=str)
parse.add_argument('--save_model_path',default='./tagging/model/model_pre',type=str)
parse.add_argument('--pretrained_model_path',default='./tagging/pretrain_models/slowonly_nl_embedded_gaussian_r50_8x8x1_150e_kinetics400_rgb_20210308-e8dd9e82.pth')
args=parse.parse_args()


label_num=82


def test_pre(model,dataloader_test):
    model.eval()


    pred_count=np.zeros(label_num)
    correct_count=np.zeros(label_num)
    count=np.zeros(label_num)
    for i, (frames, labels) in enumerate(dataloader_test):
        frames = frames.cuda()
        with torch.no_grad():
            pred = model(frames)
            pred = pred.sigmoid()
            pred = pred.to('cpu').numpy()
            pred_mask = pred > 0.5

        labels = labels.numpy()
        for j in range(len(labels)):
            for t in range(label_num):
                if labels[j][t]>0:
                    count[t]+=1
                if pred_mask[j][t]>0 and labels[j][t]==pred_mask[j][t]:
                    correct_count[t]+=1
                if pred_mask[j][t]>0:
                    pred_count[t]+=1
        print(i,len(dataloader_test))
    precision=correct_count/(pred_count+1e-8)
    recall=correct_count/(count+1e-8)
    f1_score=2*precision*recall/(precision+recall+1e-8)
    for i in range(len(f1_score)):
        print(i,f1_score[i])
    score=np.mean(f1_score)
    print(score)
    return score
def train():

    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)
    dataset_train=video_dataset(video_list=args.video_list_train,video_dir=args.video_dir,anno_path=args.anno_path,label_id_path=args.label_id_path,clip_num=8)
    '''dataset_test = video_dataset(video_list=video_list_test, video_dir=video_dir, clip_num=8, anno_path=anno_path,label_id_path=label_id_path,
                                 test_mode=True)
    dataloader_test = DataLoader(dataset_test, batch_size=32, num_workers=8, collate_fn=transforms.collate_fn)'''

    sampler = RandomSampler(dataset_train)
    dataloader_train = DataLoader(dataset_train, batch_size=16, sampler=sampler, num_workers=8,collate_fn=transforms.collate_fn)

    model=slowonly_net(label_num=label_num,pretrained_path=args.pretrained_model_path)
    #model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model = model.cuda()
    opt = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4, momentum=0.9)

    max_epoch = 15
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[8, 12])
    last_score = 0
    best_score = 0

    warm_iter = len(dataloader_train)//3

    warm_sch = WarmUpLR(opt, warm_iter)
    total_iter=0
    for epoch in range(max_epoch):
        model.train()
        iter = 0
        for i, (frames, labels) in enumerate(dataloader_train):
            opt.zero_grad()
            frames = frames.cuda()
            labels = labels.cuda()
            logits = model(frames)
            logits = logits.sigmoid()
            loss = torch.mean(torch.sum(
                -labels * torch.log(logits + 1e-8)*(1-logits) - (1 - labels)  * torch.log(
                    1 - logits + 1e-8)*(logits**4), dim=-1))
            loss.backward()

            opt.step()
            print('epoch:[%d/%d] iter:[%d/%d] lr:%f loss:%f last_score:%f best_score:%f' % (
            epoch, max_epoch, iter, len(dataloader_train), opt.param_groups[0]['lr'], float(loss), last_score,
            best_score))
            iter += 1
            total_iter+=1
            if total_iter<=warm_iter:
                warm_sch.step()

        sch.step()
        '''gap = test_pre(model, dataloader_test)
        print(gap)
        if gap > best_score:
            best_score = gap
            torch.save(model.state_dict(), os.path.join(save_model_path, 'best.pth'))
        last_score = gap'''
        torch.save(model.state_dict(), os.path.join(args.save_model_path, 'last.pth'))
        

if __name__=='__main__':
    train()
