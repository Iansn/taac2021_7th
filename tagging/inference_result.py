import json
import os

from src.models.mutimod_model import mutimod_net
import torch
import numpy as np
import cv2

import time
import argparse
parse=argparse.ArgumentParser()
parse.add_argument('--video_dir',default='./data/algo-2021/dataset/videos/test_5k_2nd',type=str)
parse.add_argument('--asr_feat_path',default='./temp_data/asr_feat_test',type=str)
parse.add_argument('--ocr_feat_path',default='./temp_data/ocr_feat_test',type=str)
parse.add_argument('--seg_result_path',default='./temp_data/seg_result.json',type=str)
parse.add_argument('--model_path',default='./tagging/model/last.pth',type=str)
parse.add_argument('--write_path',default='./result/result.json',type=str)
parse.add_argument('--label_id_path',default='./data/algo-2021/dataset/label_id.txt',type=str)
args=parse.parse_args()
img_size=(320,256)
mean=[123.675, 116.28, 103.53]
std=[58.395, 57.12, 57.375]

label_num = 82
score_thr = 0.0
clip_num=16
def get_label_map():
    f=open(args.label_id_path,'r',encoding='utf-8')
    label_map={}
    for line in f:
        line=line.replace('\n','')
        split_line=line.split('\t')
        label_map[int(split_line[1])]=split_line[0]
    f.close()


    return label_map

def get_frames(start_pos,end_pos,frame_count):
    if start_pos==end_pos:
        end_pos=start_pos+1

    cur_frame_count = end_pos-start_pos
    #print(cur_frame_count)
    internal = cur_frame_count // clip_num
    frame_pos=[]
    if internal == 0:
        for i in range(start_pos,end_pos):
            frame_pos.append(i)
        l=len(frame_pos)
        for i in range(0,clip_num-l):
            frame_pos.append(end_pos-1)

    else:
        indx = np.arange(0, internal * clip_num, internal)
        # print(len(indx),internal,frame_count,self.clip_num)

        for i in range(1, len(indx)):
            r = int((indx[i - 1] + indx[i]) / 2)
            frame_pos.append(min(r+start_pos,frame_count-1))
        r = int((indx[len(indx) - 1] + cur_frame_count - 1) / 2)
        frame_pos.append(min(r+start_pos,frame_count-1))

    return frame_pos
def get_decode_position(segments,frame_count,fps):
    batch_pos=[]
    decode_pos=[]
    for i in range(len(segments)):
        segment = segments[i]['segment']
        # print(segment,fps,frame_count)
        start_pos = int(min(max(0, np.floor(segment[0] * fps)), frame_count - 1))
        end_pos = int(min(max(0, np.ceil(segment[1] * fps)), frame_count))
        frame_pos=get_frames(start_pos,end_pos,frame_count)
        decode_pos.extend(frame_pos)
        batch_pos.append(frame_pos)
    return batch_pos,decode_pos
def get_ocr_feat(name):
    feat=torch.load(os.path.join(args.ocr_feat_path,name))['feat'].flatten(0)
    feat=feat.numpy()
    return feat
def get_asr_feat(name):
    feat = torch.load(os.path.join(args.asr_feat_path, name))['feat'].flatten(0)
    feat = feat.numpy()
    return feat
def inference():

    label_map= get_label_map()
    #print(label_map)
    model = mutimod_net(label_num, None)
    ckpt = torch.load(args.model_path, map_location='cpu')
    new_ckpt = {}
    for name in ckpt:
        if name.startswith('module.'):
            new_ckpt[name[7:]] = ckpt[name]
        else:
            new_ckpt[name] = ckpt[name]
    model.load_state_dict(new_ckpt)
    model.cuda()
    model.eval()
    seg_infos=json.load(open(args.seg_result_path,'r'))
    result = {}
    cnt=0

    for video_name in seg_infos:
        s=time.time()
        cnt+=1


        cap=cv2.VideoCapture()
        cap.open(os.path.join(args.video_dir,video_name))
        fps=cap.get(cv2.CAP_PROP_FPS)

        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        segments=seg_infos[video_name]
        batch_pos,decode_pos=get_decode_position(segments,frame_count,fps)
        asr_feat=get_asr_feat(video_name[:-4]+'.pth')
        ocr_feat=get_ocr_feat(video_name[:-4]+'.pth')



        ptr=0
        batch_frames = []
        decode_frames={}
        while ptr<frame_count:
            ret=cap.grab()
            if ptr in decode_pos:
                ret, frame = cap.retrieve()
                frame = cv2.resize(frame, img_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # frame = frame.astype(np.float32)
                frame = (frame * 1.0 - mean) / std
                frame = frame.transpose(2, 0, 1)
                decode_frames[ptr]=frame
            ptr+=1
        cap.release()

        for i in range(len(batch_pos)):
            pos=batch_pos[i]
            frames=[]
            #print(pos)
            for j in range(len(pos)):
                frames.append(decode_frames[pos[j]])
            batch_frames.append(frames)










        batch_frames=np.array(batch_frames).astype(np.float32)
        batch_frames=torch.from_numpy(batch_frames)
        batch_frames=batch_frames.permute(0,2,1,3,4).cuda()
        asr_feat=torch.from_numpy(asr_feat.astype(np.float32)).cuda().unsqueeze(0).repeat([batch_frames.size(0),1])
        ocr_feat=torch.from_numpy(ocr_feat.astype(np.float32)).cuda().unsqueeze(0).repeat([batch_frames.size(0),1])
        with torch.no_grad():
            pred=model(batch_frames,asr_feat,ocr_feat)
            pred=pred.sigmoid()
            pred=pred.to('cpu').numpy()
        if video_name not in result:
            result[video_name] = {}
            result[video_name]['result'] = []
        for j in range(len(pred)):

            cur_pred=pred[j]
            segment=segments[j]['segment']

            segment_res = [float(np.round(segment[0], 2)), float(np.round(segment[1], 2))]
            labels = []
            scores = []
            cur_pred_sorted=np.argsort(-cur_pred)

            for t in range(20):
                idx=cur_pred_sorted[t]


                score=float(cur_pred[idx])
                if score>score_thr:
                    labels.append(label_map[idx])
                    scores.append(float(np.round(score, 4)))


                #推广页不可能有多个label
                if t==0 and idx==23:
                    break





            result_info = {'segment': segment_res, 'labels': labels, 'scores': scores}

            result[video_name]['result'].append(result_info)



        #print(result[video_name])
        e=time.time()
        print(cnt,e-s)
    with open(args.write_path,'w',encoding='utf-8') as f:

        json.dump(result,f,ensure_ascii=False)

if __name__=='__main__':
    inference()
