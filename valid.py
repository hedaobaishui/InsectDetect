# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import argparse
import os
import pprint
import shutil
# import sys

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.models import DenseNet,ShuffleNetV2,MobileNetV2
import torch.optim as optim
from CORE.function import train,validate,save_checkpoint
from CONFIG import config,update_config
from pathlib import Path
import time
import logging


def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')

    # parser.add_argument('--cfg',
    #                     help='experiment configure file name',
    #                     required=False,
    #                     type=str,
    #                     default='../experiments/cls_hrnet_w18_small_v1_sgd_lr5e-2_wd1e-4_bs32_x100.yaml')
    # default='../experiments/cls_hrnet_w30_sgd_lr5e-2_wd1e-4_bs32_x100.yaml')

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--traindataname',
                        help='traindata directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--testModel',
                        help='testModel',
                        type=str,
                        default='')
    parser.add_argument('--preModel',
                        help='pretrained Model',
                        type=str,
                        # default='./output/pretrainedmodel/hrnetv2_w30_imagenet_pretrained.pth')
                        default='./output/pretrainedmodel/hrnet_w18_small_model_v1.pth')

    args = parser.parse_args()
    update_config(config, args)
    return args


def main():

    model = ShuffleNetV2([4, 8, 4], [24, 48, 96, 192, 1024],num_classes=3)
    # model = MobileNetV2(num_classes=3, width_mult=1.0).cuda()
    gpus = list([0])
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()
    valdir = '/home/magic/Project/jinyankeji/testdata/test_difficult/board1/board1_b_0.56/'
    # valdir = '/home/magic/Project/jinyankeji/testdata/test2/true/true/board2/'
    # valdir_cl１ = '/home/magic/Project/jinyankeji/testdata/test2/true/true/board2/test1/'
    valdir = '/home/magic/Project/jinyankeji/testdata/test2/true/true/board7/test1/'
    # valdir = '/home/magic/Project/jinyankeji/testdata/test2/wrong/wrong/board2/test/'
    normalize = transforms.Normalize(mean=[0.35307208, 0.43874484, 0.53854634],
                                     std=[0.28877657, 0.25837516, 0.22828328])
    modelfile = './log/model_best.pth.tar'
    model.load_state_dict(torch.load(modelfile))
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    valid_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(int(config.MODEL.IMAGE_SIZE[0] / 0.875)),
            transforms.CenterCrop(config.MODEL.IMAGE_SIZE[0]),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )
    top1 = validate(config, valid_loader, model, criterion,
                                  './savemodel/')
    print(top1)

if __name__ == '__main__':
    main()
