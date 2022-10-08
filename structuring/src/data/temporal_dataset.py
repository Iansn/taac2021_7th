from torch.utils.data import Dataset
import numpy as np
import json
import torch
import os
from src.utils.utils import iou_with_anchors
import cv2

class Temporal_Dataset(Dataset):
    def __init__(self,anno_path,file_list,feat_path,time_rescale_size,mode='train'):
        super(Temporal_Dataset,self).__init__()
        self.anno_path=anno_path
        self.video_infos=[]
        f=open(file_list,'r')
        for line in f:
            self.video_infos.append(line.replace('\n',''))
        f.close()
        self.feat_path=feat_path
        self.temporal_scale=time_rescale_size
        self.mode=mode
        self.annos=json.load(open(self.anno_path,'r',encoding='utf-8'))
        self.temporal_gap = 1. / (self.temporal_scale-1)





    def __len__(self):
        return len(self.video_infos)
    def load_feat(self,video_name):
        split_=video_name.split('#')
        ori_duration=float(split_[1])
        new_duration=float(split_[2][:-4])
        ckpt=torch.load(os.path.join(os.path.join(self.feat_path,video_name)),map_location='cpu')
        
        feat=ckpt['feat']

        ori_scale=feat.size(0)

        feat = feat.numpy()
        #print(feat.shape)
        diff_feat = np.zeros([ori_scale, feat.shape[1]])
        diff_feat[0] = feat[0]
        diff_feat[1:ori_scale] = feat[1:ori_scale] - feat[0:ori_scale - 1]

        feat = np.concatenate([feat, diff_feat], axis=-1)

        feat=self.rescale_feat(feat,self.temporal_scale)

        feat = torch.from_numpy(feat.astype(np.float32))
        return feat,ori_scale,ori_duration,new_duration
    def __getitem__(self, idx):
        video_name=self.video_infos[idx]

        anno_info=self.annos[video_name.split('#')[0]+'.mp4']['annotations']
        feat,ori_scale,ori_duration,new_duration=self.load_feat(video_name)

        match_score_start, confidence_score = self._get_train_label(anno_info, ori_duration)

        if self.mode == "train":

            return feat, confidence_score, match_score_start,video_name
        else:
            return feat,ori_scale,ori_duration,anno_info,match_score_start,video_name


    def rescale_feat(self,feat,scale):

        feat=cv2.resize(feat,(feat.shape[1],scale))
        return feat

    def _get_train_label(self, anno_info,ori_duration):


        video_labels = anno_info  # the measurement is second, not frame


        gt_bbox = []
        #gt_iou_map = []
        gt_xmins=[]
        gt_xmaxs=[]


        for j in range(len(video_labels)):
            tmp_info = video_labels[j]

            tmp_start=tmp_info['segment'][0]
            tmp_end=tmp_info['segment'][1]
            gt_bbox.append([min(tmp_start,tmp_end), max(tmp_start,tmp_end)])
            gt_xmins.append(min(tmp_start,tmp_end))

            gt_xmaxs.append(max(tmp_start,tmp_end))

            if j==len(video_labels)-1:
                gt_xmins.append(max(tmp_start,tmp_end))



        gt_xmins = np.array(gt_xmins)
        match_score_start = []
        for j in range(self.temporal_scale):
            cur_time=j/(self.temporal_scale-1)*ori_duration
            diff=np.abs(cur_time-gt_xmins)
            min_indx=np.argmin(diff)
            match_score_start.append(max(0.5-diff[min_indx],-1))

        gt_xmaxs = np.array(gt_xmaxs)

        gt_xmins=gt_xmins[:len(gt_xmins)-1]

        gt_iou_map = np.zeros([self.temporal_scale, self.temporal_scale])
        for i in range(self.temporal_scale):
            for j in range(i+1, self.temporal_scale):
                gt_iou_map[i, j] = np.max(
                    iou_with_anchors(i/(self.temporal_scale-1)*ori_duration, j/(self.temporal_scale-1)*ori_duration, gt_xmins, gt_xmaxs))
        gt_iou_map=gt_iou_map.astype(np.float32)
        gt_iou_map = torch.from_numpy(gt_iou_map)


        match_score_start = torch.from_numpy(np.array(match_score_start).astype(np.float32))


        return match_score_start, gt_iou_map

