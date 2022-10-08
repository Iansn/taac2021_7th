# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn as nn

class BMN(nn.Module):
    def __init__(self, temporal_scale,feat_dim,prop_boundary_ratio=0.5,num_sample=32,num_sample_perbin=3):
        super(BMN, self).__init__()
        self.tscale = temporal_scale
        self.prop_boundary_ratio = prop_boundary_ratio
        self.num_sample = num_sample
        self.num_sample_perbin = num_sample_perbin
        self.feat_dim=feat_dim

        self.hidden_dim_1d = 256
        self.hidden_dim_2d = 256
        self.hidden_dim_3d = 256

        self._get_interp1d_mask()



        self.x_1d_b = nn.Sequential(
            nn.Conv1d(self.feat_dim, self.hidden_dim_1d, kernel_size=3,padding=1,groups=4),
            nn.ReLU(),

        )


        encoder_layer=nn.TransformerEncoderLayer(d_model=self.hidden_dim_1d,nhead=8,dim_feedforward=1024)
        self.transformer=nn.TransformerEncoder(encoder_layer,num_layers=6,norm=nn.LayerNorm(self.hidden_dim_1d))
        encoder_layer_local=nn.TransformerEncoderLayer(d_model=self.hidden_dim_1d,nhead=8,dim_feedforward=1024)
        self.transformer_local=nn.TransformerEncoder(encoder_layer_local,num_layers=3,norm=nn.LayerNorm(self.hidden_dim_1d))
        # Temporal Evaluation Module
        self.x_1d_s = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1,groups=4),
            nn.ReLU(),
            nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1),
            nn.Sigmoid()
        )




        self.x_3d_p = nn.Sequential(
            nn.Conv3d(self.hidden_dim_1d, self.hidden_dim_3d, kernel_size=(self.num_sample, 1, 1),stride=(self.num_sample, 1, 1)),
            nn.ReLU()
        )


        self.x_2d_p = nn.Sequential(

            nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim_2d, 2, kernel_size=1),
            nn.Sigmoid()
        )

        self.local_step=33

    def get_pos_ebb(self,x,scale,size):
        pe = torch.zeros(scale, size,device=x.device)
        position = torch.arange(0, scale).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, size, 2) * (-math.log(10000.0) / size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(1)
    def get_local_feat(self,x):
        feat=x
        b,c,t=feat.size()
        group=t//self.local_step
        feat=feat.view(b,c,self.local_step,group).permute(2,3,0,1).reshape([self.local_step,b*group,c])
        posion_ebb_local=self.get_pos_ebb(x,self.local_step,self.hidden_dim_1d)
        feat = feat + posion_ebb_local.expand_as(feat)

        feat = self.transformer_local(feat)

        feat=feat.view(self.local_step,group,b,c).view(-1,b,c).permute(1,2,0)
        return feat
    def get_global_feat(self,x):
        feat=x.permute(2,0,1)
        posion_ebb=self.get_pos_ebb(x,self.tscale,self.hidden_dim_1d)
        feat1=feat+posion_ebb.expand_as(feat)
        feat2=feat.flip(0)+posion_ebb.expand_as(feat)
        feat1=self.transformer(feat1)
        feat2=self.transformer(feat2)
        feat=feat1+feat2.flip(0)
        feat=feat.permute(1,2,0)
        return feat


    def forward(self, x):

        base_feature = self.x_1d_b(x)


        local_feat=self.get_local_feat(base_feature)
        gb_feat=self.get_global_feat(base_feature)
        base_feature=local_feat+gb_feat

        start1 = self.x_1d_s(base_feature).squeeze(1)


        confidence_map=base_feature
        confidence_map = self._boundary_matching_layer(confidence_map)
        confidence_map = self.x_3d_p(confidence_map).squeeze(2)




        confidence_map = self.x_2d_p(confidence_map)



        return confidence_map, start1

    def _boundary_matching_layer(self, x):
        input_size = x.size()

        out = torch.matmul(x, self.sample_mask).reshape(input_size[0],input_size[1],self.num_sample,self.tscale,self.tscale)
        return out

    def _get_interp1d_bin_mask(self, seg_xmin, seg_xmax, tscale, num_sample, num_sample_perbin):
        # generate sample mask for a boundary-matching pair
        plen = float(seg_xmax - seg_xmin)
        plen_sample = plen / (num_sample * num_sample_perbin - 1.0)
        total_samples = [
            seg_xmin + plen_sample * ii
            for ii in range(num_sample * num_sample_perbin)
        ]
        p_mask = []
        for idx in range(num_sample):
            bin_samples = total_samples[idx * num_sample_perbin:(idx + 1) * num_sample_perbin]
            bin_vector = np.zeros([tscale])
            for sample in bin_samples:
                sample_upper = math.ceil(sample)
                sample_decimal, sample_down = math.modf(sample)
                if int(sample_down) <= (tscale - 1) and int(sample_down) >= 0:
                    bin_vector[int(sample_down)] += 1 - sample_decimal
                if int(sample_upper) <= (tscale - 1) and int(sample_upper) >= 0:
                    bin_vector[int(sample_upper)] += sample_decimal
            bin_vector = 1.0 / num_sample_perbin * bin_vector
            p_mask.append(bin_vector)
        p_mask = np.stack(p_mask, axis=1)
        return p_mask

    def _get_interp1d_mask(self):
        # generate sample mask for each point in Boundary-Matching Map
        mask_mat = []
        for end_index in range(self.tscale):
            mask_mat_vector = []
            for start_index in range(self.tscale):
                if start_index <end_index:
                    p_xmin = start_index
                    p_xmax = end_index
                    center_len = float(p_xmax - p_xmin) + 1
                    sample_xmin = p_xmin - center_len * self.prop_boundary_ratio
                    sample_xmax = p_xmax + center_len * self.prop_boundary_ratio
                    p_mask = self._get_interp1d_bin_mask(
                        sample_xmin, sample_xmax, self.tscale, self.num_sample,
                        self.num_sample_perbin)
                else:
                    p_mask = np.zeros([self.tscale, self.num_sample])
                mask_mat_vector.append(p_mask)
            mask_mat_vector = np.stack(mask_mat_vector, axis=2)
            mask_mat.append(mask_mat_vector)
        mask_mat = np.stack(mask_mat, axis=3)
        mask_mat = mask_mat.astype(np.float32)

        self.sample_mask = nn.Parameter(torch.Tensor(mask_mat).view(self.tscale, -1), requires_grad=False)


if __name__ == '__main__':
    import opts
    opt = opts.parse_opt()
    opt = vars(opt)
    model=BMN(opt)
    input=torch.randn(2,400,100)
    a,b,c=model(input)
    print(a.shape,b.shape,c.shape)