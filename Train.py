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
import copy
import time
from model.utils import DataLoader
from model.model import *
from sklearn.metrics import roc_auc_score
from utils import *
import random
import scipy.io as scio
from scipy import misc
from PIL import Image
import seaborn as sns

import argparse


parser = argparse.ArgumentParser(description="DESDnet")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--epochs', type=int, default=61, help='number of epochs for training')
parser.add_argument('--start_epoch', type=int, default=0, help='the start idx of epoch for training')
parser.add_argument('--savepth_interval', type=int, default=10, help='the training interval of epochs to save model')
parser.add_argument('--loss_m', type=float, default=0.2, help='weight of the motion loss')
parser.add_argument('--loss_f', type=float, default=0.01, help='weight of the feature loss')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
parser.add_argument('--num_workers', type=int, default=1, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, Avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='/home/zlab-4/Chenxia/dataset/UCSD', help='directory of data')
parser.add_argument('--exp_dir', type=str, default='checkpoint', help='directory of log')

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

torch.backends.cudnn.enabled = True                           # make sure to use cudnn for computational performance

train_folder = args.dataset_path+args.dataset_type+"/Train"

# Loading dataset
train_dataset = DataLoader(train_folder, transforms.Compose([
             transforms.ToTensor(),
             ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

train_size = len(train_dataset)


train_batch = data.DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, drop_last=True)

# Model setting
model = convAE(args.c, args.t_length, args.fdim)
params_encoder_p = list(model.encoder_p.parameters())
params_encoder_r = list(model.encoder_r.parameters())
params_decoder = list(model.decoder.parameters())
params_fusion = list(model.fusion.parameters())
params_encoder_e = list(model.encoder_e.parameters())
params = params_encoder_p + params_encoder_r + params_decoder + params_fusion + params_encoder_e
optimizer = torch.optim.Adam(params, lr=args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
model.cuda()
del params_encoder_p, params_encoder_r, params_decoder, params_fusion

# Report the training process
log_dir = os.path.join('./exp', args.dataset_type, args.exp_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

radius = []
if args.start_epoch > 0:
    model_dir = os.path.join(log_dir, 'model_{}.pth'.format(args.start_epoch))
    mask_dir = os.path.join(log_dir, 'mask_{}.pt'.format(args.start_epoch))
    radius_dir = os.path.join(log_dir, 'radius_{}.pt'.format(args.start_epoch))
    center_dir = os.path.join(log_dir, 'center.pt')
    if os.path.isfile(model_dir) and os.path.isfile(mask_dir):
        model_pth = torch.load(model_dir)
        model.load_state_dict(model_pth['model_state_dict'])
        optimizer.load_state_dict(model_pth['optimizer_state_dict'])
        scheduler.last_epoch = args.start_epoch
        print('Loading model sucessfully!')
        mask_copy = torch.load(mask_dir)['mask']
        print('Loading mask sucessfully!')

        radius = torch.load(radius_dir)['r']
        updated_R = radius[-1]
        # updated_R = torch.zeros(1, device='cuda')
        print('Loading radius sucessfully!')
        center = torch.load(center_dir)['c']
        print('Loading center sucessfully!')
    else:
        args.start_epoch = 0
        print('The {} or {} is not exist, please training model from the start.'.format(model_dir, mask_dir))

    del model_dir, mask_dir

# loss function for intensity
loss_func_mse = nn.MSELoss(reduction='none')

# Training
nu = 0.1
feas = np.zeros((2, 512, 32, 32), dtype=np.float32)
for epoch in range(args.start_epoch+1, args.epochs):
    mask_ori = []
    model.train()

    for j, (ori_imgs) in enumerate(train_batch):
        if epoch > 1:
            updated_R = 0.0              # radius initialization
            # conduct erasure
            imgs = ori_imgs              # without erasure
            for n in range(args.t_length):
                imgs[:,  n * args.c: (n+1) * args.c, ...] = imgs[:,  n * args.c: (n+1) * args.c, ...].mul(torch.Tensor(mask_copy))
        else:
            imgs = ori_imgs

        imgs = Variable(imgs).cuda()                # batch ,shape = b x d x h x w, 4x15x256x256
        ori_imgs=Variable(ori_imgs).cuda()
        # continous 4 frames predict the next frame
        outputs, fea, _ = model.forward(ori_imgs, imgs, args.c, True)

        # loss intensity
        loss_mse_r = loss_func_mse(outputs, ori_imgs[:, 4*args.c:, ...])
        loss_rec = torch.mean(loss_mse_r)

        # weighted RGB loss
        loss_motion = 0
        for i in range(4):
            diff = loss_func_mse(outputs, ori_imgs[:, i * args.c:(i + 1) * args.c, ...])
            loss_motion += ((i + 1) / ((args.t_length-1)**2)) * torch.mean(diff)

        if (j*args.batch_size) > (train_size-50):   # choose the last 50 frame for generating mask
            mask_ori.extend((loss_mse_r.detach().cpu().numpy()) > (loss_rec.item()))

        optimizer.zero_grad()

        if epoch < 2:    # for center in SVDD
            loss_svdd = torch.zeros(1, device='cuda')
            if epoch == 1:
                feas[0, ...] = np.mean(fea.detach().cpu().numpy(), 0)   #Cyclic mean calculation reduces space complexity
                if feas[1, ...].all() == 0:
                    feas[1, ...] = feas[0, ...]
                else:
                    feas[1, ...] = np.mean(feas, 0)

                center = torch.from_numpy(feas[1, ...]).cuda()

        else:
            # feature compact loss
            dist = torch.mean((fea - center) ** 2)
            scores = dist - updated_R ** 2
            loss_svdd = updated_R ** 2 + (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))

            #  update the radius of SVDD
            updated_R = np.quantile(np.sqrt(dist.detach().cpu().numpy()), 1 - nu)
            radius = np.append(radius, updated_R)

        loss = loss_rec + args.loss_m * loss_motion + args.loss_f *loss_svdd
        loss.backward()
        optimizer.step()

        if j % 1000 == 0:
            print('idx: {}, Loss: Reconstruction {:.6f}/ Motion {:.6f}/Svdd {:.6f}'.format(j, loss_rec.item(),
                                                                                loss_motion.item(),
                                                                                loss_svdd.item()))
    # generate the mask
    mask_ori = np.array(mask_ori)
    mask_copy = np.zeros([args.c, args.h, args.w])
    for c in range(args.c):
        for h in range(args.h):
            for w in range(args.w):
                mask_copy[c, h, w] = mask_ori[:, c, h, w].any()
    scheduler.step()

    print('Epoch:', epoch)
    print('idx: {}, Loss: Reconstruction {:.6f}/ Motion {:.6f}/Svdd {:.6f}'.format(j, loss_rec.item(),
                                                                        loss_motion.item(),
                                                                        loss_svdd.item()))

    if epoch == 1:
        print('Training is finished')
        # Save the model and the related item
        state_c = {"c": center}
        torch.save(state_c, os.path.join(log_dir, 'center.pt'))
        state_model = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch}
        torch.save(state_model, os.path.join(log_dir, 'model_{}.pth'.format(epoch)))
        state_mask = {"mask": mask_copy, "epoch": epoch}
        torch.save(state_mask, os.path.join(log_dir, 'mask_{}.pt'.format(epoch)))
        state_r = {"r": radius, "epoch": epoch}
        torch.save(state_r, os.path.join(log_dir, 'radius_{}.pt'.format(epoch)))

    if epoch % args.savepth_interval == 0:
        state_model = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch}
        torch.save(state_model, os.path.join(log_dir, 'model_{}.pth'.format(epoch)))
        state_mask = {"mask": mask_copy, "epoch": epoch}
        torch.save(state_mask, os.path.join(log_dir, 'mask_{}.pt'.format(epoch)))
        state_r = {"r": radius, "epoch": epoch}
        torch.save(state_r, os.path.join(log_dir, 'radius_{}.pt'.format(epoch)))

print('Training is finished')
# Save the model and the related items
state_model = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch}
torch.save(state_model, os.path.join(log_dir, 'model_{}.pth'.format(epoch)))
state_mask = {"mask": mask_copy, "epoch": epoch}
torch.save(state_mask, os.path.join(log_dir, 'mask_{}.pt'.format(epoch)))
state_r = {"r": radius, "epoch": epoch}
torch.save(state_r, os.path.join(log_dir, 'radius_{}.pt'.format(epoch)))


