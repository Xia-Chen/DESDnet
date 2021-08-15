import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.nn.init as init
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import time
from model.utils import DataLoader
from model.model import *
from sklearn.metrics import roc_auc_score
from utils import *
from Score_Cal import *
import random
import glob
import scipy.io as scio
from scipy import misc
import seaborn as sns

import argparse


parser = argparse.ArgumentParser(description="DESDnet")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=8, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=2, help='batch size for test')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
parser.add_argument('--loss_f', type=float, default=0.2, help='weight of the feature loss')
parser.add_argument('--alpha', type=float, default=0.8, help='weight for the anomality score')
parser.add_argument('--th', type=float, default=0.01, help='threshold for test updating')
parser.add_argument('--idx', type=int, default=10, help='model idx for testing')
parser.add_argument('--num_workers', type=int, default=1, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, Avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='/home/zlab-4/Chenxia/dataset/UCSD', help='directory of data')
parser.add_argument('--chkt_dir', type=str, default='exp/', help='directory of model')
parser.add_argument('--m_items_dir', type=str, default='exp/', help='directory of model')

args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if args.gpus is None:
    gpus = "0"
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus
else:
    gpus = ""
    for i in range(len(args.gpus)):
        gpus = gpus + args.gpus[i] + ","
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus[:-1]

torch.backends.cudnn.enabled = True    # make sure to use cudnn for computational performance

test_folder = args.dataset_path+args.dataset_type+"/Test"

# Loading dataset
test_dataset = DataLoader(test_folder, transforms.Compose([ transforms.ToTensor(),
             ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

test_size = len(test_dataset)

test_batch = data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                             shuffle=False, num_workers=args.num_workers_test, drop_last=False)

# load label
labels = np.load('/home/zlab-4/Chenxia/MNAD-master/data/frame_labels_'+args.dataset_type+'.npy')

videos = OrderedDict()
videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
sequence_n_frame = np.zeros(len(videos_list), dtype=int)
i = 0
for video in videos_list:
    video_name = video.split('/')[-1]
    videos[video_name] = {}
    videos[video_name]['path'] = video
    if args.dataset_type == 'Avenue' or args.dataset_type == 'shanghai':
        videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
    else:
        videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.tif'))
    videos[video_name]['frame'].sort()
    videos[video_name]['length'] = len(videos[video_name]['frame'])
    sequence_n_frame[i] = videos[video_name]['length']
    i += 1

labels_list = []
label_length = 0
psnr_dict = {}
patches_mse_dict = {}

# Setting for video anomaly detection
for video in sorted(videos_list):
    video_name = video.split('/')[-1]
    labels_list = np.append(labels_list,
                            labels[0, label_length + 4:videos[video_name]['length'] + label_length])  
    label_length += videos[video_name]['length']
    psnr_dict[video_name] = []
    patches_mse_dict[video_name] = []

# build model
model = convAE(args.c, args.t_length, args.fdim)
model.cuda()

model_dir = os.path.join(args.chkt_dir, args.dataset_type, 'checkpoint/model_{}.pth'.format(args.idx))
# load model
if os.path.isfile(model_dir):
    model_pth = torch.load(model_dir)
    model.load_state_dict(model_pth['model_state_dict'])
    print('Loading model sucessfully!')
else:
    print('The {} is not exist.'.format(model_dir))

print('Evaluation of', args.dataset_type, 'on ', model_dir)

loss_func_mse = nn.MSELoss(reduction='none')

psnr_dir = os.path.join(args.chkt_dir, args.dataset_type, 'checkpoint/psnr_{}.npy'.format(args.idx))
mse_patch_dir = os.path.join(args.chkt_dir, args.dataset_type, 'checkpoint/mse_patch_{}.npy'.format(args.idx))
if os.path.isfile(psnr_dir) and os.path.isfile(mse_patch_dir):
    psnr_dict = np.load(psnr_dir, allow_pickle=True).item()
    patches_mse_dict = np.load(mse_patch_dir, allow_pickle=True).item()

else:
    label_length = 0
    video_num = 0
    label_length += videos[videos_list[video_num].split('/')[-1]]['length']
    model.eval()
    for k, (imgs) in enumerate(test_batch):

        if k * args.test_batch_size == label_length - 4 * (video_num + 1):
            video_num += 1
            label_length += videos[videos_list[video_num].split('/')[-1]]['length']

        imgs = Variable(imgs).cuda()
        outputs, fea_r_r, _ = model.forward(imgs, imgs, args.c, False)

        frame_mse_imgs_t = torch.mean(loss_func_mse((outputs + 1) / 2, (imgs[:, 4 * args.c:, :, :] + 1) / 2),dim=[1, 2, 3])
        frame_mse_imgs = [item.item() for item in frame_mse_imgs_t]
        mse_pixel = (((outputs + 1) / 2) - ((imgs[:, 4 * args.c:, :, :] + 1) / 2)) ** 2
        patch_mse_imgs, _ = find_max_patch(mse_pixel.cpu().detach().numpy(), patches=1)

        psnr_dict[videos_list[video_num].split('/')[-1]].extend(psnr(np.array(frame_mse_imgs)))
        patches_mse_dict[videos_list[video_num].split('/')[-1]].extend(patch_mse_imgs)

    np.save(psnr_dir, psnr_dict)
    np.save(mse_patch_dir, patches_mse_dict)

# ------compute the abnormality score------------
is_norm = True
is_each = True                   # smooth
if is_norm:
    patches_mse_list = []
    psnr_list = []
    for video in sorted(videos_list):
        video_name = video.split('/')[-1]
        if is_each:
            each_mse = np.array(normalize_score_list_gel(patches_mse_dict[video_name]))
            if len(each_mse) == 0:
                break
            else:
                patches_mse_list.extend(multi_future_frames_to_scores(each_mse))
                psnr_list.extend(multi_future_frames_to_scores(np.array(normalize_score_list_gel(psnr_dict[video_name]))))
        else:
            patches_mse_list.extend(normalize_score_list_gel(patches_mse_dict[video_name]))
            psnr_list.extend(normalize_score_list_gel(psnr_dict[video_name]))

patches_mse_list = np.asarray(patches_mse_list)
psnr_list = np.asarray(psnr_list)

accuracy_psnr = AUC(psnr_list, np.expand_dims(1-labels_list, 0))
accuracy_patch = AUC(patches_mse_list, np.expand_dims(labels_list, 0))
patch_eer = eer(labels_list, np.squeeze(patches_mse_list))
print('---------------- The result of -----------------', args.dataset_type)
print('AUC based on psnr_norm:{}%'.format(accuracy_psnr*100))
print('AUC based on patch_mse_maxmean_norm:{}%, EER:{}% '.format(accuracy_patch*100, patch_eer*100))




