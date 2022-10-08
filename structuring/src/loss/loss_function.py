# -*- coding: utf-8 -*-
import torch
import numpy as np



def get_mask(tscale):
    mask = np.zeros([tscale, tscale], np.float32)
    for i in range(tscale):
        for j in range(i+1, tscale):
            mask[i, j] = 1
    return torch.Tensor(mask)


def bmn_loss_func(pred_bm, pred_start, gt_iou_map, gt_start, bm_mask):
    pred_bm_reg = pred_bm[:, 0].contiguous()
    pred_bm_cls = pred_bm[:, 1].contiguous()

    gt_iou_map = gt_iou_map * bm_mask

    pem_reg_loss = pem_reg_loss_func(pred_bm_reg, gt_iou_map, bm_mask)
    pem_cls_loss = pem_cls_loss_func(pred_bm_cls, gt_iou_map, bm_mask)
    tem_loss = tem_loss_func(pred_start, gt_start)

    loss = 10*pem_reg_loss+tem_loss+pem_cls_loss
    return loss, tem_loss,pem_reg_loss, pem_cls_loss


def tem_loss_func(pred_start, gt_start):
    def bi_loss(pred_score, gt_label):

        pmask = (gt_label >=0.0).float()

        epsilon = 1e-8
        num_positive = torch.sum(pmask,dim=-1)
        num_entries = pmask.size(1)

        ratio = num_entries / (num_positive+epsilon)

        coef_0 = 0.5 * ratio / (ratio - 1)
        coef_1 = 0.5 * ratio

        loss_pos = coef_1.view(-1,1) *torch.log(pred_score + epsilon) * pmask
        loss_neg = coef_0.view(-1,1) *torch.log(1.0 - pred_score + epsilon) * (1-pmask)
        loss = -1 * torch.mean(torch.mean(loss_pos + loss_neg,dim=-1))
        return loss

    loss_start = bi_loss(pred_start, gt_start)

    loss = loss_start
    return loss


def pem_reg_loss_func(pred_score, gt_iou_map, mask):

    pred_score=pred_score*mask
    gt_iou_map=gt_iou_map*mask
    num_entries=torch.sum(mask)
    pred_score=pred_score.flatten(1)
    gt_iou_map=gt_iou_map.flatten(1)
    loss=torch.mean(torch.sum(torch.abs(pred_score-gt_iou_map),dim=-1)/num_entries)


    return loss


def pem_cls_loss_func(pred_score, gt_iou_map, mask):
    pmask = (gt_iou_map > 0.9).float()
    nmask = (gt_iou_map <= 0.9).float()
    nmask = nmask * mask
    epsilon = 1e-8
    nmask=nmask.flatten(1)
    pmask=pmask.flatten(1)
    num_positive = torch.sum(pmask,dim=-1)
    num_entries = num_positive + torch.sum(nmask,dim=-1)
    ratio = num_entries / (num_positive+epsilon)
    coef_0 = 0.5 * ratio / (ratio - 1)
    coef_1 = 0.5 * ratio
    pred_score=pred_score.flatten(1)
    #print(pred_score.size(),num_entries.size(),coef_0.size(),coef_1.size())
    loss_pos = coef_1.view(-1,1) * torch.log(pred_score + epsilon) * pmask
    loss_neg = coef_0.view(-1,1) * torch.log(1.0 - pred_score + epsilon) * nmask
    loss = torch.mean(-1 * torch.sum(loss_pos + loss_neg,dim=-1) / num_entries)
    return loss
