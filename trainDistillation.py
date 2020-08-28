# ------------------------------------------------------------------------------
# Written by hedaobaishui (taisanai@163.com)
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
import torchvision.models as Tmodels
import torch.optim as optim
from CORE.function import train_Distillation,validate,save_checkpoint
from CONFIG import config,update_config
from pathlib import Path
import time
import logging


def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')

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
    # 蒸馏温度系数
    T = 2
    pretainded_teacher_model = './savemodel/model_Resnet50.pth.tar'
    args = parse_args()
    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.traindataname, 'train')
    # Teacher model
    model_resnet50 = Tmodels.resnet50(num_classes=3)
    # TODO:加载teacher模型
    model_resnet50.load_state_dict(torch.load(pretainded_teacher_model))
    model_resnet50 = torch.nn.DataParallel(model_resnet50, device_ids=[0]).cuda()
    # 固定网络参数
    for param in model_resnet50.parameters():
        param.requires_grad = False
    # Student model
    model_mobilenetv2 = Tmodels.MobileNetV2(num_classes=3)
    model_mobilenetv2 = torch.nn.DataParallel(model_mobilenetv2, device_ids=[0]).cuda()
    # gpus = list([0])
    # model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()
    criterion_KL =  torch.nn.KLDivLoss().cuda()

    optimizer = get_optimizer(config, model_mobilenetv2)

    best_perf = 0.0
    best_model = False
    last_epoch = config.TRAIN.BEGIN_EPOCH
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file)
            last_epoch = checkpoint['epoch']
            best_perf = checkpoint['perf']
            model_mobilenetv2.module.load_state_dict(checkpoint['state_dict'])
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
    traindir = '/home/magic/Data/8_19/train_data_aug/'
    traindir = '/home/magic/Data/8_19/small_set84/'
    valdir = '/home/magic/Data/8_19/vaild_data/'

    # imagenet mean std
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    # rgb data mean std
    normalize = transforms.Normalize(mean=[0.2703114097967692, 0.31799275002263866, 0.3975207719944205],
                                     std=[0.2534873463261856, 0.23769423511732185, 0.24343107915013384])
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
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
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
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        lr_scheduler.step()
        # train for one epoch
        train_Distillation(config, train_loader, model_resnet50, model_mobilenetv2, T,
                           criterion, criterion_KL, optimizer, epoch, './savemodel/')
        # evaluate on validation set
        perf_indicator = validate(config, valid_loader, model_mobilenetv2, criterion,
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
            'state_dict': model_mobilenetv2.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir, filename='checkpoint.pth.tar')

    final_model_state_file = os.path.join('./savemodel/',
                                          'model_mobilenetv2.pth.tar')
    torch.save(model_mobilenetv2.module.state_dict(), final_model_state_file)

if __name__ == '__main__':
    main()
