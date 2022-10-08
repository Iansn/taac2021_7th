from src.models.mutimod_model import mutimod_net
import torch
import numpy as np
import src.utils.transforms_mutimod as transforms_mutimod
from src.data.video_dataset_mutimod import video_dataset
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler

import os
from src.utils.warmup_scheduler import WarmUpLR
import json
import argparse
parse=argparse.ArgumentParser()
parse.add_argument('--video_list_train',default='./temp_data/train.txt',type=str)
parse.add_argument('--video_list_test',default='./temp_data/test.txt',type=str)
parse.add_argument('--video_dir',default='./data/algo-2021/dataset/videos/video_5k/train_5k',type=str)
parse.add_argument('--label_id_path',default='./data/algo-2021/dataset/label_id.txt',type=str)
parse.add_argument('--anno_path',default='./data/algo-2021/dataset/structuring/GroundTruth/train5k.txt',type=str)
parse.add_argument('--save_model_path',default='./tagging/model',type=str)
parse.add_argument('--pretrained_model_path',default='./tagging/model/model_pre/last.pth',type=str)
parse.add_argument('--asr_feat_path',default='./temp_data/asr_feat_train',type=str)
parse.add_argument('--ocr_feat_path',default='./temp_data/ocr_feat_train',type=str)
args=parse.parse_args()

label_num = 82


label_prop_thr=0.1


def get_label_weight():
    weight=np.ones([82,2])
    f = open(args.label_id_path, 'r', encoding='utf-8')
    label_map = {}
    for line in f:
        line = line.replace('\n', '').split('\t')
        label_name = line[0]
        label_id = int(line[1])
        label_map[label_name] = label_id
    f.close()
    annos = json.load(open(args.anno_path, 'r', encoding='utf-8'))
    label_count = {}
    sums=0
    for name in annos:
        anno = annos[name]['annotations']
        for seg in anno:
            labels = seg['labels']
            for label in labels:
                label_id = label_map[label]
                if label_id not in label_count:
                    label_count[label_id] = 1
                else:
                    label_count[label_id] += 1
                sums+=1
    for i in range(82):

        weight[i][0]=0.5*np.log(sums/label_count[i])
        weight[i][1]=0.5*np.log(sums/(sums-label_count[i]))
        print(i,weight[i][0],weight[i][1])

    return weight





def test_pre(model, dataloader_test):
    model.eval()

    pred_count = np.zeros(label_num)
    correct_count = np.zeros(label_num)
    count = np.zeros(label_num)
    for i, (frames, labels,asr_feat,ocr_feat) in enumerate(dataloader_test):
        frames = frames.cuda()
        asr_feat = asr_feat.cuda()
        ocr_feat = ocr_feat.cuda()
        with torch.no_grad():
            pred = model(frames,asr_feat,ocr_feat)
            pred = pred.sigmoid()

            pred = pred.to('cpu').numpy()
            pred_mask = pred > 0.5

        labels = labels.numpy()
        for j in range(len(labels)):
            for t in range(label_num):
                if labels[j][t] > 0:
                    count[t] += 1
                if pred_mask[j][t] > 0 and labels[j][t] == pred_mask[j][t]:
                    correct_count[t] += 1
                if pred_mask[j][t] > 0:
                    pred_count[t] += 1
        print(i, len(dataloader_test))
    precision = correct_count / (pred_count + 1e-8)
    recall = correct_count / (count + 1e-8)
    f1_score = 2 * precision * recall / (precision + recall + 1e-8)
    for i in range(len(f1_score)):
        print(i, f1_score[i])
    score = np.mean(f1_score)
    print(score)
    return score


def train():
    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)


    dataset_train = video_dataset(video_list=args.video_list_train, video_dir=args.video_dir, anno_path=args.anno_path,
                                  label_id_path=args.label_id_path, asr_path=args.asr_feat_path,ocr_path=args.ocr_feat_path,clip_num=8)


    sampler = RandomSampler(dataset_train)
    dataloader_train = DataLoader(dataset_train, batch_size=32, sampler=sampler, num_workers=8,
                                  collate_fn=transforms_mutimod.collate_fn)
    # model = i3d_net(label_num=label_num, pretrained_path=pretrained_model_path)

    model = mutimod_net(label_num=label_num, pretrained_path=None)
    ckpt = torch.load(args.pretrained_model_path, map_location='cpu')
    new_ckpt = {}
    for name in ckpt:
        if 'cls_head' in name:
            continue
        if name.startswith('module.'):
            new_ckpt[name[7:]] = ckpt[name]
        else:
            new_ckpt[name] = ckpt[name]
    model.load_state_dict(new_ckpt,strict=False)
    #model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model = model.cuda()
    opt = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4, momentum=0.9)
    # opt=torch.optim.AdamW(model.parameters(),lr=0.0005,weight_decay=1e-4)
    max_epoch = 15
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[8, 12])
    last_score = 0
    best_score = 0
    label_weight = get_label_weight()
    label_weight = torch.from_numpy(label_weight.astype(np.float32))
    label_weight = label_weight.cuda()
    warm_iter = len(dataloader_train) // 3
    # warm_iter=int(0.1*max_iter*max_epoch)
    warm_sch = WarmUpLR(opt, warm_iter)
    total_iter = 0

    for epoch in range(max_epoch):
        model.train()
        iter = 0
        for i, (frames, labels,asr_feat,ocr_feat) in enumerate(dataloader_train):
            opt.zero_grad()
            frames = frames.cuda()
            labels = labels.cuda()
            asr_feat=asr_feat.cuda()
            ocr_feat=ocr_feat.cuda()
            logits = model(frames,asr_feat,ocr_feat)
            logits = logits.sigmoid()

            loss = torch.mean(torch.sum(
                -labels * torch.log(logits + 1e-8)*(1-logits) - (1 - labels) * torch.log(
                    1 - logits + 1e-8) * (logits ** 4), dim=-1))


            loss.backward()
            # clip_grad.clip_grad_norm(model.parameters(), max_norm=40, norm_type=2)
            opt.step()
            print('epoch:[%d/%d] iter:[%d/%d] lr:%f loss:%f last_score:%f best_score:%f' % (
                epoch, max_epoch, iter, len(dataloader_train), opt.param_groups[0]['lr'], float(loss),last_score,
                best_score))
            iter += 1
            total_iter += 1
            if total_iter <= warm_iter:
                warm_sch.step()

        sch.step()
        '''gap = test_pre(model, dataloader_test)
        print(gap)
        if gap > best_score:
            best_score = gap
            torch.save(model.state_dict(), os.path.join(save_model_path, 'best.pth'))
        last_score = gap'''
        torch.save(model.state_dict(), os.path.join(args.save_model_path, 'last.pth'))


if __name__ == '__main__':
    train()