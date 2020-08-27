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

def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            #model.parameters(),
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            #model.parameters(),
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR
        )
    elif cfg.TRAIN.OPTIMIZER == 'rmsprop':
        optimizer = optim.RMSprop(
            #model.parameters(),
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            alpha=cfg.TRAIN.RMSPROP_ALPHA,
            centered=cfg.TRAIN.RMSPROP_CENTERED
        )

    return optimizer


def create_logger(cfg, traindata_name, phase='train'):
    root_output_dir = Path(cfg.LOG_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()


    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(traindata_name).split('/')[-1]

    final_output_dir = root_output_dir  / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / model / \
            (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)




def main():
    args = parse_args()
    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.traindataname, 'train')
    # copy model file
    this_dir = os.path.dirname(__file__)
    # models_dst_dir = os.path.join('./savemodel/', 'models')
    # if os.path.exists(models_dst_dir):
    #     shutil.rmtree(models_dst_dir)
    # shutil.copytree(os.path.join(this_dir, '../lib/models'), models_dst_dir)

    # model = models.densenet121(pretrained=True)
    # model = densenet._densenet('densenet121', 32, (6, 12, 24, 16), 64,'False', 'False')
    # model = DenseNet(growth_rate=12, block_config=(6, 12, 24, 16),
    #              num_init_features=64, bn_size=4, drop_rate=0, num_classes=3)
    # model = DenseNet(growth_rate=12, block_config=(4, 8, 16, 12),
    #              num_init_features=64, bn_size=4, drop_rate=0, num_classes=3)
    model = ShuffleNetV2([4, 8, 4], [24, 48, 96, 192, 1024],num_classes=3)
    # model = MobileNetV2(num_classes=3, width_mult=1.0)
    gpus = list([0])
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()

    optimizer = get_optimizer(config, model)

    best_perf = 0.0
    best_model = False
    last_epoch = config.TRAIN.BEGIN_EPOCH

    # 使用预训练模型
    usePreModle = False
    if usePreModle:
        logger.info('=> loading pretrained model from {}'.format(args.preModel))
        checkpoint = torch.load(args.preModel)
        model.load_state_dict(checkpoint, False)

    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file)
            last_epoch = checkpoint['epoch']
            best_perf = checkpoint['perf']
            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))
            best_model = True

    if isinstance(config.TRAIN.LR_STEP, list):  # [50,70]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR,
            last_epoch - 1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR,
            last_epoch - 1
        )

    # Data loading code
    # traindir = os.path.join(config.DATASET.ROOT, config.DATASET.TRAIN_SET)
    # traindir = 'G:/Project/yanjingkeji/trainGray/'
    traindir = '/home/magic/Project/jinyankeji/train_data/train_aug/'
    # valdir = os.path.join(config.DATASET.ROOT, config.DATASET.TEST_SET)
    # valdir = 'C:/Users/Administrator/Desktop/train/test/'
    valdir = '/home/magic/Project/jinyankeji/train_data/test_difficult/board1/board1_b_0.16/'
    valdir = '/home/magic/Project/jinyankeji/testdata/test2/true/true/board7/'
    # valdir = 'C:/Users/Administrator/Desktop/train/testGRAY/'

    # imagenet mean std
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    # rgb data mean std
    normalize = transforms.Normalize(mean=[0.35307208, 0.43874484, 0.53854634],
                                     std=[0.28877657, 0.25837516, 0.22828328])
    # normalize = transforms.Normalize(mean=[0.54390562,0.42947787,0.13272157],
    #                                  std=[0.23150272,0.27040822,0.15742092])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(config.MODEL.IMAGE_SIZE[0]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=True,
        num_workers=config.WORKERS,
        pin_memory=True
    )

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

    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        lr_scheduler.step()
        # train for one epoch
        train(config, train_loader, model, criterion, optimizer, epoch,
              './savemodel/')
        # evaluate on validation set
        perf_indicator = validate(config, valid_loader, model, criterion,
                                  './savemodel/')

        if perf_indicator > best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': config.MODEL.NAME,
            'state_dict': model.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir, filename='checkpoint.pth.tar')

    final_model_state_file = os.path.join('./savemodel/',
                                          'final_state.pth.tar')
    # logger.info('saving final model state to {}'.format(
    #     final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    # writer_dict['writer'].close()


if __name__ == '__main__':
    main()
