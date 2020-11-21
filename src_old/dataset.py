import torch
from torch.utils import data
from torch.utils.data.dataset import Dataset
import os
import numpy as np
from plyfile import PlyData

# TODO: script to list in dataset.csv all data available
# TODO: script to split dataset.csv into train dev and test sets
# TODO: Functions to do data augmentation (rotation and jitter)
# TODO: cpp script to down sample point clouds /!\ dont forget labels
# TODO: bash script data preprocess new folder architecture & downsampled point clouds
# TODO: Optional (in Dataset class)- Choose labels to keep and have the others in background
# TODO: Dataset class and dataloader function to handle it
from src.utils import true_label


class MyCustomDatasetV2(Dataset):
    def __init__(self):
        super(MyCustomDataset, self).__init__()
        self.data_folder = "/media/cedric/Data/Documents/Datasets/kitti_velodyne"
        self.data_raw = "data"
        self.sequence = 0

        root_dir = "%s/%s/velodyne/dataset/sequences/%02d/velodyne" % (self.data_folder, self.data_raw, self.sequence)
        self.count = len([name for name in os.listdir(root_dir) if os.path.isfile(root_dir + "/" + name)])

    def __getitem__(self, index):
        points_dir = self.get_(index, features=True)
        labels_dir = self.get_(index, features=False)
        points = np.fromfile(points_dir, dtype=np.float32).reshape([-1, 4])
        points = torch.from_numpy(points[:, :3].T).float()
        labels = np.fromfile(labels_dir, dtype=np.int32).reshape((-1))
        labels = labels & 0xFFFF  # get lower half for semantics
        labels = torch.from_numpy(true_label(labels))
        return points, labels

    def __len__(self):
        return self.count

    def get_(self, index, features):
        if features:
            output = "%s/%s/velodyne/dataset/sequences/%02d/velodyne/%06d.bin" \
                     % (self.data_folder, self.data_raw, self.sequence, index)
        else:
            output = "%s/%s/labels/dataset/sequences/%02d/labels/%06d.label" \
                     % (self.data_folder, self.data_raw, self.sequence, index)
        return output


class MyCustomDataset(Dataset):
    def __init__(self, train=True):
        super(MyCustomDataset, self).__init__()
        self.data_folder = "/media/cedric/Data/Documents/Datasets/kitti_velodyne"
        self.data_processed = "processed"
        self.sequence = 0

        self.root_dir = f"{self.data_folder}/{self.data_processed}/{self.sequence:02}"
        if train:
            self.count = len([name for name in os.listdir(self.root_dir) if os.path.isfile(self.root_dir + "/" + name)])
        else:
            self.count = 2

    def __getitem__(self, index):
        points_dir = f"{self.root_dir}/{index:06}.ply"

        with open(points_dir, 'rb') as f:
            plydata = PlyData.read(f)
            num_verts = plydata['vertex'].count
            points = np.zeros(shape=[num_verts, 3], dtype=np.float32)
            points[:, 0] = plydata['vertex'].data['x']
            points[:, 1] = plydata['vertex'].data['y']
            points[:, 2] = plydata['vertex'].data['z']
            labels = plydata['vertex'].data['label'].reshape((-1))

        points = torch.from_numpy(points.T).float()
        labels = torch.from_numpy(true_label(labels))
        return points, labels

    def __len__(self):
        return self.count


def dataloader():

    train = data.DataLoader(dataset=MyCustomDataset(), batch_size=1, shuffle=True)
    dev = data.DataLoader(dataset=MyCustomDataset(train=False), batch_size=1, shuffle=True)

    return train, dev, dev

# def dataloader(task, n_classes, batch_size, path_to_data=None):
#
#     n_examples = 8192  # number of point clouds
#     n_points_per_cloud = 1024  # number of points per point cloud
#
#     train = data.DataLoader(dataset=MyCustomDataset, batch_size=1, shuffle=True)
#
#     # dev = fake_data_loader(task=task,
#     #                        n_classes=n_classes,
#     #                        n_fake_data=int(n_examples * 0.1),
#     #                        batch_size=batch_size,
#     #                        n_points_per_cloud=n_points_per_cloud)
#     #
#     # test = fake_data_loader(task=task,
#     #                         n_classes=n_classes,
#     #                         n_fake_data=int(n_examples * 0.1),
#     #                         batch_size=batch_size,
#     #                         n_points_per_cloud=n_points_per_cloud)
#     #
#     # train = fake_data_loader(task=task,
#     #                          n_classes=n_classes,
#     #                          n_fake_data=n_examples,
#     #                          batch_size=batch_size,
#     #                          n_points_per_cloud=n_points_per_cloud)
#     #
#     # dev = fake_data_loader(task=task,
#     #                        n_classes=n_classes,
#     #                        n_fake_data=int(n_examples * 0.1),
#     #                        batch_size=batch_size,
#     #                        n_points_per_cloud=n_points_per_cloud)
#     #
#     # test = fake_data_loader(task=task,
#     #                         n_classes=n_classes,
#     #                         n_fake_data=int(n_examples * 0.1),
#     #                         batch_size=batch_size,
#     #                         n_points_per_cloud=n_points_per_cloud)
#
#     return train, train, train


def fake_data_loader(task,
                     n_classes=5,
                     n_fake_data=256,
                     batch_size=32,
                     n_points_per_cloud=1024):
    # n_classes: Number of classes for the classification
    # n_fake_data: Number of point clouds
    # n_points_per_cloud: Number of points per cloud
    # batch_size: The batch size

    n_channels = 3  # x, y, z channels of a point cloud

    fake_data_features = torch.randn(n_fake_data, n_channels, n_points_per_cloud)

    if task == "cls":
        # Classification task
        fake_data_labels = torch.randint(n_classes, (n_fake_data,))
    elif task == "semseg":
        # Semantic segmentation task
        fake_data_labels = torch.randint(n_classes, (n_fake_data, n_points_per_cloud))
    else:
        assert False, "Unknown task. Task should be 'cls' for classification or 'semseg' for semantic segmentation"

    fake_data_set = data.TensorDataset(fake_data_features, fake_data_labels)

    return data.DataLoader(fake_data_set, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":

    train_loader, dev_loader, _ = dataloader()
    train_inputs, train_labels = next(iter(train_loader))

    print(train_inputs.size())
    print(train_labels.size())

    # fake_loader = fake_data_loader(task="semseg")
    # for n, (inputs, labels) in enumerate(fake_loader, 0):
    #     print(inputs.size())
    #     print(labels.size())

    # x1 = torch.randn(2, 3)
    # x2 = torch.randn(2, 5)
    # donnee = torch.stack((x1, x2), 0)
    # print(donnee)

