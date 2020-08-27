import numpy as np
import os
import time
startT = time.time()
import cv2
cap = cv2.VideoCapture('/home/magic/Project/jinyankeji/train_data/train/1/1.jpg')
res,frame = cap.read()
# os.system("/home/magic/Downloads/ncnn/build/examples/shuffleNetv2Mine /home/magic/Project/jinyankeji/train_data/train/1/1.jpg")
# os.system("/home/magic/Downloads/ncnn/build/examples/shuffleNetv2Mine /home/magic/Project/jinyankeji/train_data/train/1/1.jpg")
# os.system("/home/magic/Downloads/ncnn/build/examples/shuffleNetv2Mine /home/magic/Project/jinyankeji/train_data/train/1/1.jpg")
res = os.system("/home/magic/Downloads/ncnn/build/examples/shuffleNetv2Mine /home/magic/Downloads/ncnn/build/examples/28_25.bmp")
res>>=8
print(res)
endT = time.time()
print(endT-startT)