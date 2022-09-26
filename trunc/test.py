import torch
import numpy as np
a = torch.Tensor(
    [8388607.5, 8388608.5,8388609.5 , np.inf, np.nan])
print(a)
print(torch.trunc(a.float()))
