import torch
import torch.nn as nn
import numpy as np

n_examples = 2
n_channels = 3
batch_size = 1

# m = nn.Conv2d(1, 4, 1)  # (in_channels, out_channels, kernel_size)
# input_data = torch.randn(1, 1, n_examples, n_channels)  # (batch_size, n_channels, height, width)
# output_data = m(input_data)

input_data = torch.randn(batch_size, n_channels, n_examples)  # (batch_size, n_channels, n_examples)
matrix = torch.eye(n_channels).repeat(batch_size, 1, 1) * 2
product = torch.matmul(matrix, input_data)

print(input_data)
print(matrix)
print(product)

#
# batch_size, n_channels, n_examples = input_data.size()
# print(batch_size, n_channels, n_examples )
# m = nn.MaxPool1d(n_examples)
# bn = nn.BatchNorm1d(n_channels)
# max_pool_1 = m(input_data)
# max_pool_2 = torch.max(input_data, 2, keepdim=True)[0]
#
# init = torch.eye(3).repeat(batch_size, 1, 1)
#
# init_2 = torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1])).view(1, 9).repeat(batch_size, 1)
#
# pool = nn.MaxPool1d(input_data.size(-1))(input_data)
# flat = nn.Flatten(2)(pool)
#
# ident = nn.Identity()
# iden_out = ident(input_data)
