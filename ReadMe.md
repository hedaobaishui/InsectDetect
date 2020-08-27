@[toc]
# 本项目为晶雁电子智能移虫机项目
##1.依赖环境
- cuda-10.2
- ptorch>=1.2.0
- python=3.6
- (imgaug)[]
- OpenCV=3.4
##2.建立项目
###2.1 首先使用运行 imgaugment.py 实现图像增强
 在imgaugment 中包含了以下图像增强方法：
 - 高斯模糊 gaussianblur
 - 高斯噪声 gaussiannoise
 - 图像色彩变化
 - 图像随机旋转 裁切
 - 图像抖动 imgaug::water
 - 图像加云(模拟摄像头失真)
###2.2 运行 trainBigModel.py训练模型 
这里训练的模型比较大
###2.3 运行trainDistillation.py
使用大模型和小模型联合训练 实现模型压缩
