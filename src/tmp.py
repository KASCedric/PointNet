import json

import numpy as np
from pathlib import Path

from src.utils import bin_to_ply, get_input_file_dir, rgb_from_label

dataset_root = Path("/media/cedric/Data/Documents/Datasets/kitti_velodyne/data")
sequence = "00"
point_cloud = "002586"
file_out = Path("/media/cedric/Data/Documents/Datasets/kitti_velodyne/processed") / point_cloud

pc_dir, label_dir = get_input_file_dir(dataset_root, sequence, point_cloud)

bin_to_ply(pc_dir, label_dir, file_out.with_suffix(".ply"))

# label = np.fromfile(label_dir, dtype=np.int32).reshape((-1))
# label = label & 0xFFFF       # get lower half for semantics
# pc = np.fromfile(pc_dir, dtype=np.float32).reshape([-1, 4])
#
#
# with open("config/semantic-kitti.json") as json_file:
#     color_map = json.load(json_file)["color_map"]
#
# nodes = np.fromfile(pc_dir, dtype=np.float32).reshape([-1, 4])
# labels = np.fromfile(label_dir, dtype=np.int32).reshape((-1))
# labels = labels & 0xFFFF  # get lower half for semantics
# labels = labels.reshape([-1, 1])
# # r, g, b = rgb_from_label(labels[:, 0])
# print(rgb_from_label(np.array([0, 1, 10])))
# print(rgb_from_label(1))
# print(rgb_from_label(10))

# root = Path("/media/cedric/Data/Documents/Datasets/kitti/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/")
# file_in = root / "0000000000.bin"
# file_out = Path("./0000000000.ply")
#
# bin_to_ply(file_in, file_out)
# pointcloud = np.fromfile(fileName, dtype=np.float32).reshape([-1, 4])

# with open(fileName, "rb") as f:
#     while byte := f.read(1):
#         # Do stuff with byte.
#         print(byte)
#

# with open(fileName, mode='rb') as file:  # b is important -> binary
#     fileContent = file.read()

# with open('configuration.json') as json_file:
#     data = json.load(json_file)

# import math
# from torch.utils.tensorboard import SummaryWriter
# from tqdm import tqdm
# import time
#
#
# def blue(func):
#     code = 94
#     def wrapper_blue(*args, **kwargs):
#         func('\033[' + str(code) + 'm' + args[0] + '\033[0m')
#     return wrapper_blue
#
#
# test = blue(print)
# test("hello")
# print("Hello")
#
#
# @blue
# def print_2(*args, **kwargs):
#     print(*args, **kwargs)
#
#
# print_2("world")
#
# # data = ["a", "b", "c", "d", "e", "f", "g"]
# # test = []
# # ind = []
# # for i, d in enumerate(tqdm(data)):
# #     time.sleep(1)
# #     test.append(d)
# #     ind.append(i)
# #
#
# # pbar = tqdm(["a", "b", "c", "d"])
# # pbar.set_description("Processing %s" % "N/A")
# # for char in pbar:
# #     time.sleep(1)
# #     pbar.set_description("Processing %s" % char)
# #     pbar.update()
# #     print("\n")
#
# # from tqdm import tqdm
# # s = range(5)
# # t = tqdm(s)
# # for b in range(5):
# #     for x in s:
# #         t.update()
# #         time.sleep(1)
# #     t.refresh()  # force print final state
# #     t.reset()  # reuse bar
# # t.close()  # close the bar permanently
#
# # # Write to tensorboard
# # def dump_data():
# #     for ind in range(10):
# #         x = 2 * math.pi * ind
# #         y = math.sin(x)
# #         writer.add_scalar("Loss/train", y, ind)
# #
# #
# # # default `log_dir` for tensorboard is "runs" - we'll be more specific here
# # writer = SummaryWriter('runs/point_net_cls')
# # dump_data()
# # writer.flush()
#
# # import torch
# # import torch.nn as nn
# # import numpy as np
#
# # batch_size = 1
# # n_channels = 3
# # n_examples = 3
#
# # m = nn.Conv2d(1, 4, 1)  # (in_channels, out_channels, kernel_size)
# # input_data = torch.randn(1, 1, n_examples, n_channels)  # (batch_size, n_channels, height, width)
# # output_data = m(input_data)
#
# # x1 = torch.zeros(10, 10)
# # x2 = x1.unsqueeze(2)
# # print(x2.size())
#
# # input_data = torch.randn(batch_size, n_channels, n_examples)  # (batch_size, n_channels, n_examples)
# # test = torch.randn(batch_size, 10)
# # print(test.size())
# # test = test.unsqueeze(2)
# # print(test.size())
# # sortie = test.repeat(1, 1, n_examples)
# # print(sortie.size())
# # print(
# #     sortie[0][0][0], "\n",
# #     sortie[0][1][0], "\n",
# #     sortie[0][2][0], "\n",
# # )
# # print(
# #     sortie[0][0][1], "\n",
# #     sortie[0][1][1], "\n",
# #     sortie[0][2][1], "\n",
# # )
# # identity = torch.eye(3).repeat(batch_size, 1, 1)
# # print(sortie.size())
# # print(identity.size())
# #
# # tensor2 = torch.cat([identity, sortie], 1)
# # print(sortie)
# # print(identity)
# # print(tensor2)
#
#
# # matrix = torch.eye(n_channels).repeat(batch_size, 1, 1) * 2
# # product = torch.matmul(matrix, input_data)
# # print(input_data)
# # print(matrix)
# # print(product)
#
# #
# # batch_size, n_channels, n_examples = input_data.size()
# # print(batch_size, n_channels, n_examples )
# # m = nn.MaxPool1d(n_examples)
# # bn = nn.BatchNorm1d(n_channels)
# # max_pool_1 = m(input_data)
# # max_pool_2 = torch.max(input_data, 2, keepdim=True)[0]
# #
# # init = torch.eye(3).repeat(batch_size, 1, 1)
# #
# # init_2 = torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1])).view(1, 9).repeat(batch_size, 1)
# #
# # pool = nn.MaxPool1d(input_data.size(-1))(input_data)
# # flat = nn.Flatten(2)(pool)
# #
# # ident = nn.Identity()
# # iden_out = ident(input_data)
