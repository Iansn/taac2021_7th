import torch.nn as nn
import torch.nn.functional as F
import torch
from resnet3d_slowonly import ResNet3dSlowOnly
class slowonly_net(nn.Module):
    def __init__(self,label_num,pretrained_path=None):
        super(slowonly_net,self).__init__()
        self.resnet3d = ResNet3dSlowOnly(pretrained=None, depth=50, lateral=False, conv1_kernel=(1, 7, 7),
                                         conv1_stride_t=1, pool1_stride_t=1, inflate=(0, 0, 1, 1), norm_eval=False,
                                         non_local=((0, 0, 0), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 0, 0)),
                                         non_local_cfg=dict(sub_sample=True, use_scale=True,
                                                            norm_cfg=dict(type='BN3d', requires_grad=True),
                                                            mode='embedded_gaussian'))

        if pretrained_path is not None:
            ckpt = torch.load(pretrained_path)
            new_state_dict = {}
            for name in ckpt['state_dict']:
                new_state_dict[name.replace('backbone.', '')] = ckpt['state_dict'][name]
            self.resnet3d.load_state_dict(new_state_dict, strict=False)
        #self.drop=nn.Dropout(0.5)
        self.cls_head=nn.Linear(2048,label_num)
    def forward(self,x):
        x=self.resnet3d(x)

        x=F.adaptive_avg_pool3d(x,(1,1,1))

        x=x.view(-1,2048)
        return self.cls_head(x)
    def extract_feat(self,x):
        x=self.resnet3d.extract_feat(x)

        x=F.adaptive_avg_pool3d(x,(1,1,1)).flatten(1)
        #x=x.permute(0,2,1).reshape([-1,2048])
        #x=x.view(-1,2048)
        return x