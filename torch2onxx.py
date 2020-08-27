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
import torch.onnx as tonnx
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision.models import DenseNet,ShuffleNetV2,MobileNetV2,mobilenet_v2
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
                        default='G:/Project/yanjingkeji/test_difficult/board1/board1_b_0.46/1/')
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
    model = ShuffleNetV2([4, 8, 4], [24, 48, 96, 192, 1024], num_classes=3).cuda()
    # model = MobileNetV2(num_classes=3, width_mult=1.0).cuda()
    # model = mobilenet_v2(pretrained=True).cuda()
    # model = ShuffleNetV2([4, 8, 4], [24, 116, 232, 464, 1024],num_classes=3).cuda()

    # model = DenseNet(growth_rate=12, block_config=(4, 8, 16, 12),
    #              num_init_features=64, bn_size=4, drop_rate=0, num_classes=3).cuda()
    dummy_input = torch.randn(1,3, 256, 256).cuda()
    # logger.info(get_model_summary(model, dump_input))

    # if config.TEST.MODEL_FILE:
    #     logger.info('=> loading model from {}'.format(config.TEST.MODEL_FILE))
    #     model.load_state_dict(torch.load(config.TEST.MODEL_FILE))
    modelfile = './log/model_best.pth.tar'
    # modelfile = 'G:/Project/jingyankeji/log/pre/model2/model_best.pth.tar'
    model.load_state_dict(torch.load(modelfile))
    model.eval()
    input_names = ["actual_input_1"]
    output_names = ["output1"]
    torch.onnx.export(model, dummy_input, '/home/magic/Downloads/ncnn/build/examples/ShuffleNetV2.onnx', verbose=False, input_names=input_names,output_names=output_names)
if __name__ == '__main__':
    main()

