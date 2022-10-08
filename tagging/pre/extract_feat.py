import os
from slowonly_net import slowonly_net
import torch
import cv2
import numpy as np
import time
import argparse


from concurrent.futures import ThreadPoolExecutor
label_num = 82
img_size = (320, 256)
mean = [123.675, 116.28, 103.53]
std = [58.395, 57.12, 57.375]


def get_frames(start_pos,end_pos,clip_num):
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
            frame_pos.append(min(r+start_pos,end_pos-1))
        r = int((indx[len(indx) - 1] + cur_frame_count - 1) / 2)
        frame_pos.append(min(r+start_pos,end_pos-1))

    return frame_pos
def get_decode_position(segments,clip_num):
    batch_pos=[]
    decode_pos=[]
    for i in range(len(segments)):
        segment = segments[i]
        # print(segment,fps,frame_count)
        start_pos = segment[0]
        end_pos = segment[1]
        frame_pos=get_frames(start_pos,end_pos,clip_num)
        decode_pos.extend(frame_pos)
        batch_pos.append(frame_pos)
    return batch_pos,decode_pos

def feature_extract(model,video_path, name, feature_dir,num):
    s = time.time()
    cap=cv2.VideoCapture()
    cap.open(os.path.join(video_path))
    frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps=cap.get(cv2.CAP_PROP_FPS)
    duration=frame_count/fps
    fps=int(np.ceil(fps*0.5))
    clip_num=8
    new_duration=float(frame_count*1.0/fps)
    internal=frame_count//fps
    segment=np.arange(0,internal*fps,fps)
    segments=[]
    for i in range(1,len(segment)):
        segments.append([segment[i-1],segment[i]])

    if segment[-1]<frame_count:
        segments.append([segment[-1],frame_count])
    #print(len(segments))
    batch_pos,decode_pos=get_decode_position(segments,clip_num)
    ptr = 0
    #print(len(batch_pos),new_duration)
    batch_frames = []
    decode_frames = {}
    while ptr < frame_count:
        ret = cap.grab()
        if ptr in decode_pos:
            ret, frame = cap.retrieve()
            frame = cv2.resize(frame, img_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # frame = frame.astype(np.float32)
            frame = (frame * 1.0 - mean) / std
            frame = frame.transpose(2, 0, 1)
            decode_frames[ptr] = frame
        ptr += 1
    cap.release()

    for i in range(len(batch_pos)):
        pos = batch_pos[i]
        frames = []
        #print(len(pos))
        for j in range(len(pos)):
            frames.append(decode_frames[pos[j]])
        batch_frames.append(frames)
    batch=16
    p=0
    video_feat=None
    batch_frames = np.array(batch_frames).astype(np.float32)
    while p<len(batch_frames):
        start=p
        end=min(p+batch,len(batch_frames))
        frame_tensor=torch.from_numpy(batch_frames[start:end])
        frame_tensor=frame_tensor.permute(0,2,1,3,4).cuda()

        with torch.no_grad():
            cur_feat = model.extract_feat(frame_tensor)

            if video_feat is None:
                video_feat=cur_feat
            else:
                video_feat=torch.cat([video_feat,cur_feat],dim=0)
        p+=batch


    #print(video_feat.size())
    
    video_feat=video_feat.to('cpu')
    
    state_dict={'feat':video_feat}




    torch.save(state_dict, os.path.join(feature_dir, name[:-4] +'#'+str(float(np.round(duration,3)))+'#'+str(float(np.round(new_duration,3)))+ '.pth'))

    e = time.time()
    print(e - s, fps, frame_count,num)
    return frame_count

if __name__ == '__main__':

    cnt = 0
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', default="../dataset/videos/train_5k", type=str)

    parser.add_argument('--write_path',default='',type=str)

    parser.add_argument('--model_path',default='',type=str)
    args = parser.parse_args()
    if not os.path.exists(args.write_path):
        os.makedirs(args.write_path)
    model_path=args.model_path
    model = slowonly_net(label_num=label_num)
    checkpoint = torch.load(model_path, map_location='cpu')

    new_checkpoint={}
    for name in checkpoint:
        if name.startswith('module.'):
            new_checkpoint[name[7:]]=checkpoint[name]
        else:
            new_checkpoint[name]=checkpoint[name]
    model.load_state_dict(new_checkpoint)
    model = model.cuda()
    model.eval()
    file_list=os.listdir(args.video_dir)
    #filter_video=json.load(open(args.anno,'r',encoding='utf-8'))
    feature_dir=args.write_path
    max_frame_count=0

    with ThreadPoolExecutor(max_workers=8) as executor:
        for i in range(len(file_list)):
            cnt += 1
            name = file_list[i]
            executor.submit(feature_extract, model, os.path.join(args.video_dir, name), name, feature_dir, i)

