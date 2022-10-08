import torch


import numpy as np




def collate_fn(batch):
    labels=[item[1] for item in batch if item is not None]
    ocr_feats=[item[3] for item in batch if item is not None]
    asr_feats=[item[2] for item in batch if item is not None]
    frames=[item[0] for item in batch if item is not None]
    '''for item in frames:
        for ii in item:
            print(ii.shape)'''


    labels=np.array(labels).astype(np.float32)
    frames=np.array(frames).astype(np.float32)
    asr_feats=np.array(asr_feats).astype(np.float32)
    ocr_feats=np.array(ocr_feats).astype(np.float32)
    frames=torch.from_numpy(frames)
    frames=frames.permute(0,2,1,3,4)

    labels=torch.from_numpy(labels)
    
    asr_feats=torch.from_numpy(asr_feats)
    ocr_feats=torch.from_numpy(ocr_feats)



    return (frames,labels,asr_feats,ocr_feats)
