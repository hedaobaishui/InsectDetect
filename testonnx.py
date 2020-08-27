import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import onnx
import cv2
import onnxruntime

# load onnx model
onnx_model = onnx.load("./ShuffleNetV2.onnx")
pic = '/home/magic/Downloads/ncnn/build/examples/1.bmp'
pic2 = '/home/magic/Downloads/ncnn/build/examples/2.jpg'

input = cv2.imread(pic)
input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
input = cv2.resize(input, (256, 256))

# numpy image(H x W x C) to torch image(C x H x W)
input = np.transpose(input, (2, 0, 1)).astype(np.float32)
# input = np.array(input).astype(np.float32)
# normalize
input = input / 255.0
mean=[0.35307208,0.43874484,0.53854634]
std=[0.28877657,0.25837516,0.22828328]
for i in range(256):
    for j in range(256):
        for k in range(3):
            input[k][i][j] = (input[k][i][j]-mean[k])/std[k]
# input = (input-mean)/std
input = Variable(torch.from_numpy(input))
# add one dimension in 0
input = input.unsqueeze(0)
# check onnx model
onnx.checker.check_model(onnx_model)
ort_session = onnxruntime.InferenceSession("./ShuffleNetV2.onnx")


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input)}
ort_outs = ort_session.run(None, ort_inputs)

# softmax
tensor_ort_out = torch.from_numpy(ort_outs[0])
onnx_test_out = F.softmax(tensor_ort_out, dim=1)

print("the onnx result is {}".format(onnx_test_out))
print(max(onnx_test_out[0]))
# compare onnx Runtime and PyTorch results
# np.testing.assert_allclose(to_numpy(tensor_out["linear"]), ort_outs[0], rtol=1e-01, atol=1e-05)
# 比对两种模型测试结果 
# print("Exported model has been tested with ONNXRuntime, and the result looks good!")