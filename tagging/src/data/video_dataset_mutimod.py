from torch.utils.data import Dataset
import cv2
import numpy as np
import random as rnd
import os
import copy
from decord import VideoReader
from decord import cpu, gpu
import json
import torch
img_size = (320, 256)
mean = [123.675, 116.28, 103.53]
std = [58.395, 57.12, 57.375]


class video_dataset(Dataset):
    def __init__(self, video_list, anno_path, video_dir, label_id_path,asr_path,ocr_path, clip_num, test_mode=False, seed=2000):
        super(video_dataset, self).__init__()
        self.video_list = video_list
        self.video_dir = video_dir
        self.clip_num = clip_num
        self.test_mode = test_mode
        self.video_infos = []
        label_id_map, label_num = self.get_label_id_map(label_id_path)
        self.label_num = label_num
        self.anno_path = anno_path
        self.ocr_feat_path=ocr_path
        self.asr_feat_path=asr_path
        self.anno_infos = json.load(open(anno_path, 'r', encoding='utf-8'))
        f = open(video_list, 'r')

        for line in f:
            line = line.replace('\n', '')

            anno_info = self.anno_infos[line]['annotations']

            for i in range(len(anno_info)):
                segment = anno_info[i]
                video_info = {}
                segment_info = segment['segment']
                labels = segment['labels']
                start = segment_info[0]
                end = segment_info[1]
                if start >= end:
                    continue
                if not test_mode and end - start < 1.0 and i < len(anno_info) - 1:
                    continue

                video_info['name'] = line
                video_info['segment'] = segment_info
                video_label = np.zeros(label_num).astype(np.float32)

                for label in labels:
                    video_label[label_id_map[label]] = 1

                video_info['labels'] = video_label
                self.video_infos.append(video_info)
        rnd.seed(seed)
        f.close()

    def __len__(self):
        return len(self.video_infos)
    def get_ocr_feat(self,name):
        feat=torch.load(os.path.join(self.ocr_feat_path,name))['feat'].flatten(0)
        feat=feat.numpy()
        return feat
    def get_asr_feat(self,name):
        feat = torch.load(os.path.join(self.asr_feat_path, name))['feat'].flatten(0)
        feat = feat.numpy()
        return feat
    def get_label_id_map(self, label_id_path):
        label_num = 0
        f = open(label_id_path, 'r', encoding='utf-8')
        label_map = {}
        for line in f:
            label_num += 1
            line = line.replace('\n', '')
            split_line = line.split('\t')
            label_map[split_line[0]] = int(split_line[1])
        f.close()
        return label_map, label_num

    def get_sample_pos(self, clip_num, frame_count, segment, fps, is_time_shift=False, time_shift=1.0):

        if is_time_shift == True and segment[1] - segment[0] > 2 * time_shift:
            start_seg = max(0, segment[0] + rnd.uniform(-time_shift, time_shift))
            end_seg = max(0, segment[1] + rnd.uniform(-time_shift, time_shift))
            start_pos = int(min(np.floor(start_seg * fps), frame_count - 1))
            end_pos = int(min(np.ceil(end_seg * fps), frame_count))
        else:
            start_seg = segment[0]
            end_seg = segment[1]
            start_pos = int(min(max(0, np.floor(start_seg * fps)), frame_count - 1))
            end_pos = int(min(max(0, np.ceil(end_seg * fps)), frame_count))
        # print(start_pos,end_pos,segment,start_seg,end_seg,fps,frame_count)
        return self.get_frames(start_pos, end_pos, clip_num)

    def get_frames(self, start_pos, end_pos, clip_num):
        if start_pos == end_pos:
            end_pos = start_pos + 1

        cur_frame_count = end_pos - start_pos
        # print(cur_frame_count)
        internal = cur_frame_count // clip_num
        frame_pos = []
        if internal == 0:
            for i in range(start_pos, end_pos):
                frame_pos.append(i)


        else:
            indx = np.arange(0, internal * clip_num, internal)
            # print(len(indx),internal,frame_count,self.clip_num)

            for i in range(1, len(indx)):
                if self.test_mode:
                    r = int((indx[i - 1] + indx[i]) / 2)
                else:
                    r = rnd.randint(indx[i - 1], indx[i] - 1)
                frame_pos.append(min(r + start_pos, end_pos - 1))
            if self.test_mode:
                r = int((indx[len(indx) - 1] + cur_frame_count - 1) / 2)
            else:
                r = rnd.randint(indx[len(indx) - 1], cur_frame_count - 1)
            frame_pos.append(min(r + start_pos, end_pos - 1))

        return frame_pos

    def get_train_data(self, idx):
        # cap=cv2.VideoCapture()
        video_info = self.video_infos[idx]
        labels = video_info['labels']

        name = video_info['name']
        segment = video_info['segment']
        asr_feat = self.get_asr_feat(name[:-4] + '.pth')
        ocr_feat = self.get_ocr_feat(name[:-4] + '.pth')
        # print(labels,segment)
        try:

            # print(os.path.join(self.video_dir,name))

            av = VideoReader(os.path.join(self.video_dir, name), ctx=cpu(0), width=img_size[0], height=img_size[1])
            frame_count = len(av)
            fps = av.get_avg_fps()


        except Exception as e:
            print(e)
            print('1111111')
            return None, None,None,None


        flip = rnd.random()
        is_time_shift = rnd.random() > 0.5

        frame_pos = self.get_sample_pos(self.clip_num, frame_count, segment, fps, is_time_shift)

        try:
            batch_frames = av.get_batch(frame_pos)
            batch_frames = batch_frames.asnumpy()

        except Exception as e:
            print(e)
            print(frame_pos, fps, frame_count, name)
            return None, None,None,None

        frames = []
        last_frame = None
        for frame in batch_frames:
            if flip > 0.5:
                frame = cv2.flip(frame, 1)
            frame = frame.astype(np.float32)
            frame = (frame - mean) / std
            frame = frame.transpose(2, 0, 1)
            frames.append(frame)
            last_frame = frame
        l = len(frames)
        # print(l)
        for i in range(0, self.clip_num - l):
            frames.append(copy.deepcopy(last_frame))

        return frames, labels,asr_feat,ocr_feat

    def get_test_data(self, idx):
        # cap=cv2.VideoCapture()
        video_info = self.video_infos[idx]
        labels = video_info['labels']

        name = video_info['name']
        segment = video_info['segment']
        asr_feat = self.get_asr_feat(name[:-4] + '.pth')
        ocr_feat = self.get_ocr_feat(name[:-4] + '.pth')
        # print(labels,segment)
        try:

            av = VideoReader(os.path.join(self.video_dir, name), ctx=cpu(0), width=img_size[0], height=img_size[1])
            frame_count = len(av)
            fps = av.get_avg_fps()
        except Exception as e:
            print(e)
            print('1111111')
            return None, None,None,None


        frame_pos = self.get_sample_pos(self.clip_num, frame_count, segment, fps)

        # print(frame_pos)'''
        try:
            batch_frames = av.get_batch(frame_pos)
            batch_frames = batch_frames.asnumpy()

        except Exception as e:
            print(e)
            print(frame_pos, fps, frame_count, name)
            return None, None,None,None
        frames = []
        last_frame = None
        for frame in batch_frames:
            frame = frame.astype(np.float32)
            frame = (frame - mean) / std
            frame = frame.transpose(2, 0, 1)
            frames.append(frame)
            last_frame = frame
        l = len(frames)
        # print(l)
        for i in range(0, self.clip_num - l):
            frames.append(copy.deepcopy(last_frame))

        return frames, labels,asr_feat,ocr_feat

    def __getitem__(self, idx):
        if not self.test_mode:
            return self.get_train_data(idx)
        else:
            return self.get_test_data(idx)