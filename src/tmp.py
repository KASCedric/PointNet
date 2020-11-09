import math
from torch.utils.tensorboard import SummaryWriter


# Write to tensorboard
def dump_data():
    for ind in range(10):
        x = 2 * math.pi * ind
        y = math.sin(x)
        writer.add_scalar("Loss/train", y, ind)


# default `log_dir` for tensorboard is "runs" - we'll be more specific here
writer = SummaryWriter('runs/point_net_cls')
dump_data()
writer.flush()


import torch
import torch.nn as nn
import numpy as np

# batch_size = 1
# n_channels = 3
# n_examples = 3

# m = nn.Conv2d(1, 4, 1)  # (in_channels, out_channels, kernel_size)
# input_data = torch.randn(1, 1, n_examples, n_channels)  # (batch_size, n_channels, height, width)
# output_data = m(input_data)

# x1 = torch.zeros(10, 10)
# x2 = x1.unsqueeze(2)
# print(x2.size())

# input_data = torch.randn(batch_size, n_channels, n_examples)  # (batch_size, n_channels, n_examples)
# test = torch.randn(batch_size, 10)
# print(test.size())
# test = test.unsqueeze(2)
# print(test.size())
# sortie = test.repeat(1, 1, n_examples)
# print(sortie.size())
# print(
#     sortie[0][0][0], "\n",
#     sortie[0][1][0], "\n",
#     sortie[0][2][0], "\n",
# )
# print(
#     sortie[0][0][1], "\n",
#     sortie[0][1][1], "\n",
#     sortie[0][2][1], "\n",
# )
# identity = torch.eye(3).repeat(batch_size, 1, 1)
# print(sortie.size())
# print(identity.size())
#
# tensor2 = torch.cat([identity, sortie], 1)
# print(sortie)
# print(identity)
# print(tensor2)


# matrix = torch.eye(n_channels).repeat(batch_size, 1, 1) * 2
# product = torch.matmul(matrix, input_data)
# print(input_data)
# print(matrix)
# print(product)

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
