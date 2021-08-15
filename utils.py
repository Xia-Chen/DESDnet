import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from sklearn.metrics import roc_auc_score, roc_curve


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def psnr(mse):
    psnr = []
    for i in mse:
        psnr.append(10 * math.log10(1 / i))
    return psnr


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def normalize_img(img):

    img_re = copy.copy(img)
    
    img_re = (img_re - np.min(img_re)) / (np.max(img_re) - np.min(img_re))
    
    return img_re


def point_score(outputs, imgs):
    loss_func_mse = nn.MSELoss(reduction='none')
    error = loss_func_mse((outputs[0]+1)/2,(imgs[0]+1)/2)
    normal = (1-torch.exp(-error))
    score = (torch.sum(normal*loss_func_mse((outputs[0]+1)/2,(imgs[0]+1)/2)) / torch.sum(normal)).item()
    return score


def anomaly_score(psnr, max_psnr, min_psnr):
    return ((psnr - min_psnr) / (max_psnr-min_psnr))


def normalize_score_inv_clip(score, max_score, min_score):
    return (1.0 - ((score - min_score) / (max_score-min_score)))


def normalize_score_clip(score, max_score, min_score):
    return ((score - min_score) / (max_score-min_score))


def normalize_score_list_gel(score):           # normalize in each video and save in list form
    anomaly_score_list = list()
    for i in range(len(score)):
        anomaly_score_list.append(normalize_score_clip(score[i], np.max(score), np.min(score)))

    return anomaly_score_list


def anomaly_score_list(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(normalize_score_clip(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))
        
    return anomaly_score_list


def normalize_score_list_inv(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(normalize_score_inv_clip(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))

    return anomaly_score_list


def AUC(anomal_scores, labels):
    frame_auc = roc_auc_score(y_true=np.squeeze(labels, axis=0), y_score=np.squeeze(anomal_scores))
    return frame_auc


def score_sum(list1, list2, list3, alpha):
    list_result = []
    for i in range(len(list1)):
        # list_result.append((alpha*list1[i]+(1-alpha)*list2[i]))
        list_result.append((alpha * list1[i] + (1 - alpha)/2 * list2[i] + (1 - alpha)/2 * list3[i]))
        
    return list_result


def eer(label, score):
    fpr_1, tpr_1, _ = roc_curve(label, score)
    fnr_1 = 1 - tpr_1
    eer = fpr_1[np.nanargmin(np.absolute((fnr_1 - fpr_1)))]
    return eer


def get_test_frame_labels(ground_truth, sequence_n_frame):
    assert len(ground_truth) == len(sequence_n_frame)
    labels_exclude_last = np.zeros(0, dtype=int)
    labels_exclude_first = np.zeros(0, dtype=int)
    labels_full = np.zeros(0, dtype=int)
    for i in range(len(sequence_n_frame)):
        seg = ground_truth[i]
        # 先把第i个视频的所有帧的标记令为0
        tmp_labels = np.zeros(sequence_n_frame[i])
        for j in range(0, len(seg), 2):
            # 根据gt将对应下标位置的标记置1，表示对应帧异常
            tmp_labels[(seg[j] - 1):seg[j + 1]] = 1
        # labels_exclude_last = np.append(labels_exclude_last, tmp_labels[:-1])
        # labels_exclude_first = np.append(labels_exclude_first, tmp_labels[1:])
        labels_full = np.append(labels_full, tmp_labels)
    return labels_full


# # consider the consecutive future frames to compute the regularity scores
# def multi_future_frames_to_scores(old_scores):
#     n_frames = len(old_scores)
#     new_scores = np.zeros(old_scores.shape)
#     flag_preserve_head_tail = False
#
#     if flag_preserve_head_tail:
#         # preserve the first frames
#         new_scores[0] = old_scores[0]
#         new_scores[1] = old_scores[1]
#         # preserve the last frames
#         new_scores[n_frames - 2] = old_scores[n_frames - 2]
#         new_scores[n_frames - 1] = old_scores[n_frames - 1]
#     else:
#         # process the first frame
#         new_scores[0] = (1 / (3 ** 2)) * (old_scores[0] * 3 + old_scores[1] * 2 + old_scores[2] * 1)
#         new_scores[1] = (1 / (4 ** 2)) * (old_scores[0] * 2 + old_scores[1] * 3 + old_scores[2] * 2 + old_scores[3] * 1)
#         # process the last frames
#         new_scores[n_frames - 2] = (1 / (4 ** 2)) * (
#                 old_scores[n_frames - 4] * 1 + old_scores[n_frames - 3] * 2 + old_scores[n_frames - 2] * 3 + old_scores[
#             n_frames - 1] * 2)
#         new_scores[n_frames - 1] = (1 / (3 ** 2)) * (
#                 old_scores[n_frames - 3] * 1 + old_scores[n_frames - 2] * 2 + old_scores[n_frames - 1] * 3)
#
#     for i in range(2, n_frames - 2):
#         new_scores[i] = 1 / (5 ** 2) * (
#                 old_scores[i - 2] * 1 + old_scores[i - 1] * 2 + old_scores[i] * 3 + old_scores[i + 1] * 2 +
#                 old_scores[i + 2] * 1)
#
#
#     # for i in range(n_frames - 4 + 1):
#     #         new_scores[i] = 1 / (4 ** 2) * (old_scores[i] * 4 + old_scores[i+1] * 3 + old_scores[i+2] * 2 + old_scores[i+3] * 1)
#     #
#     # new_scores[n_frames - 3] = (1 / (3 ** 2)) * (old_scores[n_frames - 3] * 4 + old_scores[n_frames - 2] * 3 + old_scores[n_frames - 1] * 2)
#     # new_scores[n_frames - 2] = (1 / (2 ** 2)) * (old_scores[n_frames - 2] * 4 + old_scores[n_frames -1] * 3)
#     # new_scores[n_frames - 1] = (1 / (1 ** 2)) * (old_scores[n_frames - 1] * 4)
#
#     return new_scores

def loss_rgb(img, img_last):
    diff = img - img_last
    motion_diff = torch.mean(diff, dim=[1, 2, 3])

    return motion_diff


def loss_func_motion_test(real_img, fake_img, img_last):
    diff_real = real_img - img_last
    diff_fake = fake_img - img_last
    diff_abs = torch.abs(diff_real - diff_fake)
    motion_diff = torch.mean(diff_abs, dim=[1, 2, 3])

    return motion_diff


def loss_func_motion(real_img, fake_img, img_last):
    diff_real = real_img - img_last
    diff_fake = fake_img - img_last
    motion_diff = torch.mean(torch.abs(diff_real - diff_fake))

    return motion_diff


# compute the image gradient loss
def loss_func_grad(x, y):

    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]

    h_tv_x = torch.abs(x[:, :, 1:, :] - x[:, :, :h_x - 1, :])
    w_tv_x = torch.abs(x[:, :, :, 1:] - x[:, :, :, :w_x - 1])

    h_tv_y = torch.abs(y[:, :, 1:, :] - y[:, :, :h_x - 1, :])
    w_tv_y = torch.abs(y[:, :, :, 1:] - y[:, :, :, :w_x - 1])

    x_diff = torch.abs(h_tv_x - h_tv_y).mean()
    y_diff = torch.abs(w_tv_x - w_tv_y).mean()

    grad_loss = (x_diff + y_diff)/2

    return grad_loss


def save_image_tensor2cv2(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为cv2格式
    :param input_tensor: 要保存的tensor
    :param filename: 保存文件名
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份到cpu
    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 去掉批次维度
    input_tensor = input_tensor.squeeze()
    # 从[0,1]转换为[0,255]再从CHW转换为HWC，最后保存为cv2
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    # RGB转BRG
    input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, input_tensor)


def multi_future_frames_to_scores(input):
    output = cv2.GaussianBlur(input, (5, 0), 10)
    return output
