# ops_perf
performance analysis

## 使用方法
ncu --print-summary per-kernel  --section SpeedOfLight python xx.py repeat_count


## 测试环境
| 算子名称    | 环境     |  版本|
| ----------- | ----------------- |----------------- |
| poly_nms |  yellow.hub.cambricon.com/cnnl/nvidia/pytorch:1.9.0-pnms | pytorch 1.9.0	|
| generate_proposals_v2  | yellow.hub.cambricon.com/cnnl/nvidia/paddle:develop-py3  <br>tips:使用python3执行,默认python是python2）| paddle develop |
| trunc	|  pytorch 1.9.0 / 1.6.0| pytorch 1.9.0 / 1.6.0 |

## 参考链接
### poly_nms
1. 参考链接
https://github.com/dingjiansw101/AerialDetection/blob/master/mmdet/ops/poly_nms/src/poly_nms_kernel.cu
2. 应用网络
FasterRCNN trans obb

###  generate_proposals_v2
1. 参考链接
- Mask R-CNN的工作原理:https://baijiahao.baidu.com/s?id=1595621180643410921&wfr=spider&for=pc
- v2 头文件 : https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/generate_proposals_v2_kernel.h
- v2 gpu kernel: https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/generate_proposals_v2_kernel.cu
- v2 cpu kernel : https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/cpu/generate_proposals_v2_kernel.cc
- v2 xpu kernel: https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/detection/generate_proposals_v2_op_xpu.cc
- PYTHON API ：https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/vision/ops.py
- v2 接口注册与说明 : https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/detection/generate_proposals_v2_op.cc
- v1 cpu kernel: https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/detection/generate_proposals_op.cc
- v1 cuda kernel: https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/detection/generate_proposals_op.cu
2. 应用网络
maskrcnn

### trunc
1. 框架版本+源码链接
- PT1.6：https://pytorch.org/docs/1.6.0/generated/torch.trunc.html#torch.trunc
- PT1.9：https://pytorch.org/docs/1.9.0/generated/torch.trunc.html?highlight=trunc#torch.trunc
- PT1.6 CPU：https://github.com/pytorch/pytorch/blob/release/1.6/aten/src/ATen/native/cpu/UnaryOpsKernel.cpp#L419
- PT1.6 GPU：https://github.com/pytorch/pytorch/blob/release/1.6/aten/src/ATen/native/cuda/UnaryFractionKernels.cu#L145
- PT1.9 CPU：https://github.com/pytorch/pytorch/blob/release/1.9/aten/src/ATen/native/cpu/UnaryOpsKernel.cpp#L63
- PT1.9 GPU：https://github.com/pytorch/pytorch/blob/release/1.9/aten/src/ATen/native/cuda/UnaryFractionKernels.cu#L146
2. 应用网络
MTCNN
3. 需求规模
[433, 5]， [39, 5]