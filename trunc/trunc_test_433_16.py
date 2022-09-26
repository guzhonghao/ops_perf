import torch
import numpy as np
import sys

print(torch.__version__)

def trunc_test(repeat):
    a_cpu = np.asarray(np.random.rand(433, 5)*100 - 50, dtype=np.float16)
    a_gpu = torch.from_numpy(a_cpu).cuda()
    for i in range(repeat):
        b_gpu = torch.trunc(a_gpu)
        b_cpu = b_gpu.cpu().numpy()
        print(b_cpu)

trunc_test(int(sys.argv[1]))
