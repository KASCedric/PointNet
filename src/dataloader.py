import torch
from torch.utils import data
from torch.utils.data.dataset import Dataset
import os
from plyfile import PlyData
import numpy as np
from utils import split


class SemanticKittiDataset(Dataset):
    def __init__(self, sequence, data_folder):
        super(SemanticKittiDataset, self).__init__()

        self.data_folder = data_folder
        self.sequence = sequence

        self.root_dir = f"{self.data_folder}/{self.sequence:02}"
        self.count = len([name for name in os.listdir(self.root_dir) if os.path.isfile(self.root_dir + "/" + name)])

    def __getitem__(self, index):
        points_dir = f"{self.root_dir}/{index:06}.ply"

        with open(points_dir, 'rb') as f:
            ply_data = PlyData.read(f)['vertex'].data
            points = np.stack([
                ply_data['x'],
                ply_data['y'],
                ply_data['z']
            ], axis=1)
            labels = np.array(ply_data['label'], dtype=int)

        points = torch.from_numpy(points.T).float()
        labels = torch.from_numpy(labels)
        return points, labels

    def __len__(self):
        return self.count


def semantic_kitti_dataloader(sequence, data_folder):
    dataset = SemanticKittiDataset(sequence=sequence, data_folder=data_folder)

    n_train, n_test = split(len(dataset), (0.8, 0.2))
    n_dev = 2  # For computing reasons
    n_test -= n_dev

    train_ds, dev_ds, test_ds = torch.utils.data.random_split(dataset=dataset, lengths=(n_train, n_dev, n_test))

    train_loader = data.DataLoader(dataset=train_ds, batch_size=1, shuffle=True)
    dev_loader = data.DataLoader(dataset=dev_ds, batch_size=1, shuffle=True)
    test_loader = data.DataLoader(dataset=test_ds, batch_size=1, shuffle=True)

    return train_loader, dev_loader, test_loader
