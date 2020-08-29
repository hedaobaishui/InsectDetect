import numpy as np
import os
import shutil
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dataloader


class MyImageFolder(datasets.DatasetFolder):
    '''
    override __getitem__() functions to get pics' path
    '''
    def __init__(self, root, transform=None, target_transform=None,
                 loader=datasets.folder.default_loader, is_valid_file=None):
        super(MyImageFolder,self).__init__(root, loader, datasets.folder.IMG_EXTENSIONS if is_valid_file is None else None,
                                           transform=transform,
                                           target_transform=target_transform,
                                           is_valid_file=is_valid_file)
        self.imgs = self.samples
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, path) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path


def classifypictures(model, pictures_path, pics_mean_var, picture_out_path):
    '''
    Classify all pictures and copy them to their class folder.
    Args:
        model: model which is loaded using load_state_dict function.
        pictures_path: All pictures' folder.
            like '/root/0/1.jpg
                 '/root/0/2.jpg'
                  ....
            '0' or '1' or other numbers(lower clss num)is needed in path
            then pictures_path = '/root/'
        pics_mean_var: pics datasets' mean and std in a dict
            like ['means':[0.52, 0.62, 0.43], 'vars':[0.32, 0.52, 56]]
        picture_out_path: path to saving the out like '/root/out_path/'
    '''
    pics_transform = transforms.Compose([transforms.Resize(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean = pics_mean_var['means'],
                                                             std = pics_mean_var['vars'])])
    pics_datasets = MyImageFolder(pictures_path, pics_transform)
    pics_loader = dataloader.DataLoader(pics_datasets,
                                        batch_size=100,
                                        num_workers=4,

                                        pin_memory=True)
    model.eval()
    with torch.no_grad():
        for i,(input, target, pics_path) in enumerate(pics_loader):
            out = model(input)
            pre_class = torch.argmax(out,dim=1).numpy()

            for i in range(len(pre_class)):
                pic_src = pics_path[i]
                pic_name = pic_src.split('/')[-1]
                pic_dst = os.path.join(picture_out_path, str(pre_class[i]))
                if not os.path.exists(pic_dst):
                    os.mkdir(pic_dst)
                pic_dst = pic_dst + '/' + pic_name
                shutil.copy2(pic_src, pic_dst)


if __name__ == '__main__':
    pictures_path = '/home/magic/Data/8_19/original_TrainData2/justphoto2020_08_19/up/justphoto08-13-18-b_0.0/youchong/'
    pictures_path = '/home/magic/Data/8_19/original_TrainData2/justphoto2020_08_19/up/justphoto08-25-37-b_0.0/fengmi/'
    pictures_path = '/home/magic/Data/8_19/original_TrainData2/justphoto2020_08_19/up/justphoto08-25-37-b_0.0/youchong/'
    # pictures_path = '/home/magic/Data/8_19/small_set84/'
    picture_out_path = '/home/magic/Data/8_19/pre_outdir/'
    pics_mean_var = {'means': [0.2703114097967692, 0.31799275002263866, 0.3975207719944205],
                     'vars': [0.2534873463261856, 0.23769423511732185, 0.24343107915013384]}

    model_path = './savemodel/MobileNetv2_nodistil.pth.tar'
    model = torchvision.models.MobileNetV2(num_classes=3)

    # model_path = './savemodel/model_mobilenetv2.pth.tar'
    # model = torchvision.models.MobileNetV2(num_classes=3)
    #
    # model_path = './savemodel/model_Resnet50.pth.tar'
    # model = torchvision.models.resnet50(num_classes=3)

    model.load_state_dict(torch.load(model_path))
    classifypictures(model, pictures_path, pics_mean_var, picture_out_path)
