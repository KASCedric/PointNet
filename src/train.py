import math
import torch
from torch.utils.tensorboard import SummaryWriter

from src.dataset import fake_data_loader
from src.model import PointNetCls

# default `log_dir` for tensorboard is "runs" - we'll be more specific here
writer = SummaryWriter('runs/point_net_cls')

n_classes = 5
n_fake_data = 256
batch_size = 32
n_points_per_cloud = 1024

# Model
point_net_cls = PointNetCls(n_classes=n_classes, bn=True, do=True)
# Dataset
fake_loader = fake_data_loader(task="cls",
                               n_classes=n_classes,
                               n_fake_data=n_fake_data,
                               batch_size=batch_size,
                               n_points_per_cloud=n_points_per_cloud)

# Sample of the dataset
fake_iter = iter(fake_loader)
input_sample, label_sample = next(fake_iter)


# Write to tensorboard
def dump_data():
    for ind in range(10):
        x = 2 * math.pi * ind
        y = math.sin(x)
        writer.add_scalar("Loss/train", y, ind)


dump_data()
writer.flush()
