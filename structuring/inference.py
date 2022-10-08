import json
from src.models.model import BMN

import numpy as np
import torch

import os
import time
import cv2
import argparse
parse=argparse.ArgumentParser()
parse.add_argument('--test_feat_dir',default='./temp_data/video_feat_test',type=str)
parse.add_argument('--model_path',default='./structuring/model',type=str)
parse.add_argument('--result_path',default='./temp_data/seg_result.json',type=str)
args=parse.parse_args()
temporal_scale=132
feat_dim=4096
cross=5

def rescale_feat(feat,scale):
    feat=cv2.resize(feat,(feat.shape[1],scale))
    return feat
def seg_test():

    feats=os.listdir(args.test_feat_dir)
    models=[]
    for i in range(cross):
        model = BMN(temporal_scale=temporal_scale,feat_dim=feat_dim)
        checkpoint = torch.load(os.path.join(args.model_path,'last_'+str(i)+'.pth'), map_location='cpu')
        new_checkpoint = {}
        for name in checkpoint:
            if name.startswith('module.'):
                new_checkpoint[name[7:]] = checkpoint[name]
            else:
                new_checkpoint[name] = checkpoint[name]
        model.load_state_dict(new_checkpoint, strict=False)
        model.cuda()
        model.eval()
        models.append(model)


    tscale = temporal_scale
    result_dict={}
    cnt=0
    #filter_name=get_filter_name()
    for name in feats:
        s = time.time()
        segment_dict = []
        split_ = name.split('#')

        ori_duration = float(split_[1])
        new_duration = float(split_[2][:-4])
        feat = torch.load(os.path.join(args.test_feat_dir, name))['feat'].flatten(1)
        ori_scale = feat.size(0)

        feat = feat.numpy()
        diff_feat = np.zeros([ori_scale, feat.shape[1]])
        diff_feat[0] = feat[0]
        diff_feat[1:ori_scale] = feat[1:ori_scale] - feat[0:ori_scale - 1]

        feat = np.concatenate([feat, diff_feat], axis=-1)
        feat = rescale_feat(feat, temporal_scale)
        feat = torch.from_numpy(feat.astype(np.float32)).cuda()
        feat = feat.unsqueeze(0).permute(0, 2, 1)
        with torch.no_grad():
            start = np.zeros(temporal_scale)
            confidence_map = np.zeros([2, tscale, tscale])
            for j in range(cross):
                confidence_map_predict, start_predict = models[j](feat)
                start_predict = start_predict[0].cpu().numpy()
                start += start_predict / cross
                confidence_map_predict = confidence_map_predict[0].cpu().numpy()
                confidence_map += confidence_map_predict / cross


            start_scores = start

            clr_confidence = confidence_map[1]
            reg_confidence = confidence_map[0]

        
        new_proposals = []
        last_index = 0
        max_index = 0
        while last_index < tscale - 1:
            max_index = np.argmax(
                reg_confidence[last_index][last_index + 1:tscale] * clr_confidence[last_index][last_index + 1:tscale]*start_scores[last_index+1:tscale])
            max_index = last_index + max_index + 1
            # print(last_index, max_index, tscale - 1)
            score = reg_confidence[last_index][max_index]

            if max_index == last_index:
                break
            new_proposals.append({'score': float(score), 'segment': [last_index / (tscale - 1) * ori_duration,
                                                                     max_index / (tscale - 1) * ori_duration]})
            last_index = max_index
        if max_index != tscale - 1:
            new_proposals.append({'score': float(reg_confidence[last_index, tscale - 1]),
                                  'segment': [last_index / (tscale - 1) * ori_duration, ori_duration]})

        flag=0
        for proposals in new_proposals:
            segment=proposals['segment']
            score = proposals['score']

            segment_dict.append({'segment':[float(segment[0]),float(segment[1])],'score':score})
            flag+=1


        result_dict[name.split('#')[0]+'.mp4']=segment_dict
        cnt+=1
        e=time.time()
        print(name,e-s,cnt,flag)
    with open(args.result_path,'w') as f:
        json.dump(result_dict,f)




if __name__=='__main__':
    seg_test()
