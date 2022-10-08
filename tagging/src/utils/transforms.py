import torch

import numpy as np



def collate_fn(batch):
    labels=[item[1] for item in batch if item is not None]
    frames=[item[0] for item in batch if item is not None]



    labels=np.array(labels).astype(np.float32)
    frames=np.array(frames).astype(np.float32)

    frames=torch.from_numpy(frames)
    frames=frames.permute(0,2,1,3,4)

    labels=torch.from_numpy(labels)



    return (frames,labels)
