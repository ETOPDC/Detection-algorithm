import torch
import  torch.nn as nn
import numpy as np

m = nn.BatchNorm2d(3, affine=True)

# 2*3*2*2
a = np.array([[[[1,1],[1,1]],
              [[0,2], [0,0]],
              [[1,-1], [0,2]]],
             [[[2, 2], [2, 2]],
              [[0, 2], [0, 0]],
              [[2, -2], [0,2]]]])
# input = torch.randn(1,2,2,2)
input = torch.from_numpy(a).to(torch.float32)
output = m(input)

print(input.size())
print(input)
print(output.size())
print(output)