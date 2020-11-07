import torch
import torch.nn as nn
import numpy as np

N = 1
K = 3
B = 1

# m = nn.Conv2d(1, 4, 1)  # (in_channels, out_channels, kernel_size)
# input_data = torch.randn(1, 1, N, K)  # (batch_size, n_channels, height, width)
# output_data = m(input_data)

input_data = torch.randn(B, K, N)  # (batch_size, n_channels, n_examples)
batch_size, n_channels, n_examples = input_data.size()
print(batch_size, n_channels, n_examples )
m = nn.MaxPool1d(N)
bn = nn.BatchNorm1d(K)
max_pool_1 = m(input_data)
max_pool_2 = torch.max(input_data, 2, keepdim=True)[0]

init = torch.eye(3).repeat(B, 1, 1)

init_2 = torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1])).view(1, 9).repeat(B, 1)

pool = nn.MaxPool1d(input_data.size(-1))(input_data)
flat = nn.Flatten(2)(pool)

ident = nn.Identity()
iden_out = ident(input_data)
