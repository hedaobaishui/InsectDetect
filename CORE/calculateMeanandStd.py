import cv2
import numpy
import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data
def calculateMeanandStd(picpath):
    '''
    Calcuate train data folder.But it's slowly.
    Args:
        picpath :train data folder!
    '''
    mean = [0.0,0.0,0.0]
    std = [0.0,0.0,0.0]
    pic_num = 0
    for rootpath, dir, pics in os.walk(picpath):
        for pic in pics:
            pic_path = rootpath + '/' +pic
            # G B R
            picdata = cv2.imread(pic_path)
            picdata = cv2.resize(picdata,(224,224)) / 255.0
            mean[0] += numpy.mean(picdata[:,:,2])
            mean[1] += numpy.mean(picdata[:,:,1])
            mean[2] += numpy.mean(picdata[:,:,0])
            pic_num += 1
    mean[0] = mean[0] / pic_num
    mean[1] = mean[1] / pic_num
    mean[2] = mean[2] / pic_num
    print(mean)
    for rootpath, dir, pics in os.walk(picpath):
        for pic in pics:
            pic_path = rootpath + '/' +pic
            # G B R
            picdata = cv2.imread(pic_path)
            picdata = cv2.resize(picdata,(224,224)) / 255.0
            std[0] += numpy.sum((picdata[:,:,2] - mean[0]) ** 2)
            std[1] += numpy.sum((picdata[:,:,1] - mean[1]) ** 2)
            std[2] += numpy.sum((picdata[:,:,0] - mean[2]) ** 2)
    std[0] = numpy.sqrt(std[0]/pic_num/224/224)
    std[1] = numpy.sqrt(std[1]/pic_num/224/224)
    std[2] = numpy.sqrt(std[2]/pic_num/224/224)
    print(std)
    print(pic_num)

#TODO
def calculateMeanandStd_useTorch(pic_path):
    '''
    Calculate train data mean and std using torch!
    Aragram:
        pic_path : train data folder
    '''
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor()])
    traindataset = datasets.ImageFolder(pic_path,transform)
    # dataLoder = torch.utils.data.DataLoader(traindataset,batch_size = 128,num_workers = 4)
    mean = [0.0,0.0,0.0]
    std = [0.0,0.0,0.0]
    pic_num = 0
    for i , (data,target) in enumerate(traindataset):
        for i in range(len(data)):
            mean[0] += data[0,:,:].mean()
            mean[1] += data[1,:,:].mean()
            mean[2] += data[2,:,:].mean()
            std[0] += data[0,:,:].std()
            std[1] += data[1,:,:].std()
            std[2] += data[2,:,:].std()
            pic_num += 1
    mean = numpy.asarray(mean) / pic_num
    std = numpy.asarray(std) / pic_num
    print(mean)
    print(std)
if __name__ == '__main__':
    # traindata_path = '/home/magic/Data/8_19/small_set84'
    traindata_path = '/home/magic/Data/8_19/train_data'
    # calculateMeanandStd(traindata_path)
    calculateMeanandStd_useTorch(traindata_path)