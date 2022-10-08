from src.data.temporal_dataset import Temporal_Dataset
from src.loss.loss_function import bmn_loss_func, get_mask
import os

import torch
import torch.nn.parallel
import torch.optim as optim
import numpy as np

from src.models.model import BMN
from src.utils.warmup_scheduler import WarmUpLR
import json
import argparse
parse=argparse.ArgumentParser()
parse.add_argument('--anno_path',default='./data/algo-2021/dataset/structuring/GroundTruth/train5k.txt',type=str)
parse.add_argument('--video_train_path',default='./temp_data/train_seg.txt',type=str)
parse.add_argument('--video_test_path',default='./temp_data/test_seg.txt',type=str)
parse.add_argument('--feat_path',default='./temp_data/video_feat_train',type=str)
parse.add_argument('--checkpoint_path',default='./structuring/model',type=str)
parse.add_argument('--cross_dir',default='./temp_data/cross',type=str)
args=parse.parse_args()
score_thr = [0.5]
cross=5

batch_size=8
temporal_scale=132
feat_dim=4096
lr=0.00025
train_epoch=15

def generate_cross():
    if not os.path.exists(args.cross_dir):
        os.makedirs(args.cross_dir)
    f=open(args.video_train_path,'r')
    names=[]
    for line in f:
        line=line.replace('\n','')
        names.append(line)
    f.close()
    l=len(names)
    split=l//cross+1
    for i in range(cross):
        start=i*split
        end=min((i+1)*split,l)
        val_split=names[start:end]

        writer_train=open(os.path.join(args.cross_dir,'train_'+str(i)+'.txt'),'w')
        writer_val=open(os.path.join(args.cross_dir,'val_'+str(i)+'.txt'),'w')
        for name in val_split:
            writer_val.write(name+'\n')
        for name in names:
            if name not in val_split:
                writer_train.write(name+'\n')
        writer_train.close()
        writer_val.close()
def train():
    generate_cross()
    for i in range(cross):
        BMN_Train(i,os.path.join(args.cross_dir,'train_'+str(i)+'.txt'),os.path.join(args.cross_dir,'val_'+str(i)+'.txt'))
    #test()
def test():
    models=[]
    for i in range(cross):
        model=BMN(temporal_scale=temporal_scale,feat_dim=feat_dim)
        checkpoint = torch.load(os.path.join(args.checkpoint_path,'best_'+str(i)+'.pth'))
        new_checkpoint = {}
        for name in checkpoint:
            if name.startswith('module.'):
                new_checkpoint[name[7:]] = checkpoint[name]
            else:
                new_checkpoint[name] = checkpoint[name]
        model.load_state_dict(new_checkpoint, strict=False)
        model=model.cuda()
        model.eval()
        models.append(model)
    test_loader = torch.utils.data.DataLoader(Temporal_Dataset(args.anno_path, args.video_test_path,
                                                               args.feat_path, temporal_scale, mode='val'),
                                              batch_size=1, shuffle=False,
                                              num_workers=16, drop_last=False)

    tscale = temporal_scale


    correct = np.zeros(len(score_thr))
    count = 0
    pre_count = 0
    with torch.no_grad():

        for i, (input_data, ori_scale, ori_duration, anno_info, gt_start) in enumerate(test_loader):

            input_data = input_data.cuda()
            input_data = input_data.permute(0, 2, 1)
            start=np.zeros(temporal_scale)
            confidence_map=np.zeros([2,tscale,tscale])
            for j in range(cross):
                confidence_map_predict, start_predict = models[j](input_data)
                start_predict=start_predict[0].cpu().numpy()
                start+=start_predict/cross
                confidence_map_predict=confidence_map_predict[0].cpu().numpy()
                confidence_map+=confidence_map_predict/cross



            start_scores = start

            clr_confidence = confidence_map[1]
            reg_confidence = confidence_map[0]
            predict_scores = np.zeros(tscale - 1)
            predict_scores[0] = start_scores[0]
            last_index = 0



            gt_start_time = []
            for j in range(len(anno_info)):
                info = anno_info[j]
                segment = info['segment']

                if j == 0:
                    continue
                gt_start_time.append(float(segment[0]))

            gt_start_time = np.array(gt_start_time)
            if len(gt_start_time) == 0:
                continue
            count += len(gt_start_time)

            predicted = np.zeros(len(gt_start_time))

            while last_index < tscale - 1:
                cur_score = reg_confidence[last_index][last_index + 1:tscale] * start_scores[last_index + 1:tscale] * \
                            clr_confidence[last_index][last_index + 1:tscale]
                max_index = np.argmax(cur_score)
                max_index = last_index + max_index + 1
                # print(last_index, max_index, tscale - 1)

                if max_index == last_index or max_index == tscale - 1:
                    break
                predict_time = max_index / (tscale - 1) * float(ori_duration[0])

                last_index = max_index
                pre_count += 1

                diff = np.abs(predict_time - gt_start_time)

                min_idx = np.argmin(diff)
                if predicted[min_idx] == 0:
                    flag = 0
                    for j in range(len(score_thr)):

                        if diff[min_idx] < score_thr[j]:
                            flag = 1
                            correct[j] += 1
                    if flag == 1:
                        predicted[min_idx] = 1

            print(i, count, pre_count, correct)
        precision = correct / pre_count
        recall = correct / count
        f1_score = 2 * precision * recall / (precision + recall + 1e-8)
    print(f1_score, precision, recall)
    avg_f1_score = np.mean(f1_score)
    print(avg_f1_score)
    return avg_f1_score


def BMN_Train(num_cross,cross_train_path,cross_val_path):
    model = BMN(temporal_scale=temporal_scale,feat_dim=feat_dim)
    #model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()
    model = model.cuda()
    optimizer = optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=1e-3)


    train_loader = torch.utils.data.DataLoader(Temporal_Dataset(args.anno_path, cross_train_path,args.feat_path, temporal_scale),batch_size, shuffle=True,
                                               num_workers=16)



    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,13])


    bm_mask = get_mask(temporal_scale)
    best_score = 0
    best_epoch = -1
    warm_iter=len(train_loader)//3
    warm_sch=WarmUpLR(optimizer,warm_iter)
    total_iter=0
    for epoch in range(train_epoch):


        model.train()


        for n_iter, (input_data, label_confidence, label_start, video_name) in enumerate(train_loader):
            input_data = input_data.cuda()
            input_data = input_data.permute(0, 2, 1)
            # print(input_data.size())
            label_start = label_start.cuda()
            # label_end = label_end.cuda()
            label_confidence = label_confidence.cuda()
            confidence_map, start1 = model(input_data)
            loss = bmn_loss_func(confidence_map, start1, label_confidence, label_start, bm_mask.cuda())
            optimizer.zero_grad()
            loss[0].backward()
            optimizer.step()
            total_iter+=1
            if total_iter<=warm_iter:
                warm_sch.step()

            print(
                'cross:[%d/%d],epoch:%d iter[%d/%d] tem_loss1:%f tem_loss2:%f,pem class_loss:%f pem reg_loss:%f loss:%f best_score:%f best_epoch:%d lr:%f' % (
                    num_cross,cross,epoch, n_iter, len(train_loader), float(loss[1]), float(loss[1]), float(loss[3]), float(loss[2]),
                    float(loss[0]), best_score, best_epoch, optimizer.param_groups[0]['lr']))


        save_state_dict={}
        for name in model.state_dict():
            if 'sample_mask' in name:
                continue
            save_state_dict[name]=model.state_dict()[name]

        torch.save(save_state_dict, os.path.join(args.checkpoint_path,'last_'+str(num_cross)+'.pth'))
        score,seg_result= BMN_inference(model,cross_val_path)
        with open(os.path.join(args.cross_dir,'val_'+str(num_cross)+'.json'),'w') as f:
            json.dump(seg_result,f)
        if score >best_score:
            best_score = score
            best_epoch = epoch
            torch.save(save_state_dict, os.path.join(args.checkpoint_path, 'best_' + str(num_cross) + '.pth'))
        scheduler.step()
        print(best_score, best_epoch)


def BMN_inference(model,cross_val_path):

    model.eval()

    test_loader = torch.utils.data.DataLoader(Temporal_Dataset(args.anno_path, cross_val_path,
                                                               args.feat_path, temporal_scale, mode='val'),
                                              batch_size=1, shuffle=False,
                                              num_workers=16, drop_last=False)
    tscale = temporal_scale

    all_score=0
    correct = np.zeros(len(score_thr))
    count = 0
    pre_count = 0
    seg_result={}
    with torch.no_grad():

        for i, (input_data, ori_scale, ori_duration, anno_info,gt_start,video_name) in enumerate(test_loader):

            name=video_name[0]
            seg_result[name]=[]
            input_data = input_data.cuda()
            input_data = input_data.permute(0, 2, 1)
            confidence_map, start1 = model(input_data)
            start = start1


            start_scores = start[0].cpu().numpy()

            clr_confidence = (confidence_map[0][1]).cpu().numpy()
            reg_confidence = (confidence_map[0][0]).cpu().numpy()
            predict_scores = np.zeros(tscale-1)
            predict_scores[0] = start_scores[0]
            last_index = 0



            gt_start_time = []
            for j in range(len(anno_info)):
                info=anno_info[j]
                segment=info['segment']

                if j==0:
                    continue
                gt_start_time.append(float(segment[0]))




            gt_start_time=np.array(gt_start_time)
            if len(gt_start_time)==0:
                continue
            count+=len(gt_start_time)

            predicted=np.zeros(len(gt_start_time))

            while last_index < tscale - 1:
                cur_score = reg_confidence[last_index][last_index + 1:tscale]*start_scores[last_index+1:tscale]*clr_confidence[last_index][last_index+1:tscale]
                max_index = np.argmax(cur_score)
                max_index = last_index + max_index + 1
                # print(last_index, max_index, tscale - 1)

                if max_index == last_index or max_index==tscale-1:
                    break
                predict_time= max_index/(tscale-1)*float(ori_duration[0])
                seg_result[name].append({'segment':[float(last_index/(tscale-1)*ori_duration[0]),float(predict_time)],'score':1.0})
                last_index = max_index
                pre_count += 1

                diff=np.abs(predict_time-gt_start_time)

                min_idx=np.argmin(diff)
                if predicted[min_idx]==0:
                    flag=0
                    for j in range(len(score_thr)):

                        if diff[min_idx] < score_thr[j]:
                            flag=1
                            correct[j] += 1
                    if flag==1:
                        predicted[min_idx]=1


            print(i, count, pre_count, correct)
            seg_result[name].append({'segment':[float(last_index/(tscale-1)*ori_duration[0]),float(ori_duration[0])],'score':1.0})
        precision=correct/pre_count
        recall = correct / count
        f1_score = 2 * precision * recall / (precision + recall + 1e-8)
    print(f1_score, precision, recall)
    avg_f1_score=np.mean(f1_score)
    print(avg_f1_score)
    return avg_f1_score,seg_result





if __name__ == '__main__':
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    train()
