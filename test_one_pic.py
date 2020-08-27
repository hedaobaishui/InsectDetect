# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import shutil
import pprint
from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision.models import DenseNet,ShuffleNetV2,MobileNetV2
import torchvision.datasets as datasets
from CONFIG import config,update_config

import torchvision.transforms as transforms
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=False,
                        type=str,
                        default='../experiments/cls_hrnet_w18_small_v1_sgd_lr5e-2_wd1e-4_bs32_x100.yaml')
                                #'../experiments/cls_hrnet_w18_small_v1_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--picloc',
                        help='picture location',
                        type=str,
                        default='/home/magic/Project/jinyankeji/train_data/test_difficult/board1/board1_b_0.46/2/')
                        # default='G:/Project/yanjingkeji/test/1/251.jpg')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--testModel',
                        help='testModel',
                        type=str,
                        default='./output/final_state.pth.tar')

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    time1 = time.time()
    args = parse_args()

    # logger, final_output_dir, tb_log_dir = create_logger(
    #     config, args.cfg, 'valid')

    # logger.info(pprint.pformat(args))
    # logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    # model = DenseNet(growth_rate=12, block_config=(4, 8, 16, 12),
    #              num_init_features=64, bn_size=4, drop_rate=0, num_classes=3)
    model = ShuffleNetV2([4, 8, 4], [24, 48, 96, 192, 1024], num_classes=3)
    model = MobileNetV2(num_classes=3, width_mult=1.0)
    dump_input = torch.rand(
        (1, 3, config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0])
    )
    # logger.info(get_model_summary(model, dump_input))

    # if config.TEST.MODEL_FILE:
    #     logger.info('=> loading model from {}'.format(config.TEST.MODEL_FILE))
    #     model.load_state_dict(torch.load(config.TEST.MODEL_FILE))
    modelfile = './log/model_best.pth.tar'
    model.load_state_dict(torch.load(modelfile))
    # else:
    #     model_state_file = os.path.join(final_output_dir,
    #                                     'final_state.pth.tar')
    #     logger.info('=> loading model from {}'.format(model_state_file))
    #     model.load_state_dict(torch.load(model_state_file))

    # Data loading code
    trans = transforms.Compose([
        transforms.Scale(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.35307208,0.43874484,0.53854634],
                                     std=[0.28877657,0.25837516,0.22828328])
    ])
    n=0
    time2 = time.time()
    for _,_,allfile in os.walk(args.picloc):
        for file in allfile:
            filepath = args.picloc + file
            img = Image.open(filepath)
            input = trans(img)  # 这里经过转换后输出的input格式是[C,H,W],网络输入还需要增加一维批量大小B
            input = input.unsqueeze(0)  # 增加一维，输出的img格式为[1,C,H,W]
            score = model(input)  # 将图片输入网络得到输出
            probability = torch.nn.Softmax(dim=1)(score) # 计算softmax，即该图片属于各类的概率
            max_value, index = torch.max(probability, 1)  # 找到最大概率对应的索引号，该图片即为该索引号对应的类别
            if index!=2:
                #print(index,filepath)
                n= n+1
        time3 = time.time()
        print(1-n/len(allfile))
        print("begintime={0},testtime={1},testavg={2}".format((time2 - time1),(time3-time2),(time3-time2)/len(allfile)))

if __name__ == '__main__':
    main()

