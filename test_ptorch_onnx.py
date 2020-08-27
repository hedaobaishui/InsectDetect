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
# import sys
import shutil
import pprint
import onnx
import onnxruntime
from PIL import Image
from torch.autograd import Variable
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
                        default='G:/Project/yanjingkeji/test_difficult/board1/board1_b_0.46/2/')
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
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    # model = MobileNetV2(num_classes=3, width_mult=1.0)
    model = ShuffleNetV2([4, 8, 4], [24, 48, 96, 192, 1024],num_classes=3)
    modelfile = './log/model_best.pth.tar'
    model.load_state_dict(torch.load(modelfile))
    model.eval()
    trans = transforms.Compose([
        transforms.Scale(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.35307208,0.43874484,0.53854634],
                                     std=[0.28877657,0.25837516,0.22828328])
    ])
    time2 = time.time()
    pic = '/home/magic/Downloads/ncnn/build/examples/1.bmp'
    pic2 = '/home/magic/Downloads/ncnn/build/examples/2.bmp'
    pic3 ='/home/magic/Project/jinyankeji/train_data/train/1/1.jpg'
    img = Image.open(pic).convert('RGB')



    input = trans(img)  # 这里经过转换后输出的input格式是[C,H,W],网络输入还需要增加一维批量大小B
    input = input.unsqueeze(0)  # 增加一维，输出的img格式为[1,C,H,W]
    score = model(input)  # 将图片输入网络得到输出
    probability = torch.nn.Softmax(dim=1)(score) # 计算softmax，即该图片属于各类的概率
    # probability = torch.argmax(score,dim=1) # 计算softmax，即该图片属于各类的概率
    # max_value, index = torch.max(probability, 1)  # 找到最大概率对应的索引号，该图片即为该索引号对应的类别
    time3 = time.time()
    # print('pytorch pre = ',probability)
    print(probability)

    # img = Image.open(pic2)
    # input = trans(img)  # 这里经过转换后输出的input格式是[C,H,W],网络输入还需要增加一维批量大小B
    # input = input.unsqueeze(0)  # 增加一维，输出的img格式为[1,C,H,W]
    input = Variable(input)
    onnx_model = onnx.load("/home/magic/Downloads/ncnn/build/examples/ShuffleNetV2-sim.onnx")
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession("/home/magic/Downloads/ncnn/build/examples/ShuffleNetV2.onnx")

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input)}
    ort_outs = ort_session.run(None, ort_inputs)

    # softmax
    tensor_ort_out = torch.from_numpy(ort_outs[0])
    # onnx_test_out = torch.argmax(tensor_ort_out, dim=1)
    onnx_test_out = torch.softmax(tensor_ort_out, dim=1)

    # print("the onnx result is {}".format(onnx_test_out))
    print(onnx_test_out)
    # print(max(onnx_test_out[0]))

if __name__ == '__main__':
    main()

