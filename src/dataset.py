import torch
from torch.utils import data


def dataloader(task, n_classes, batch_size, path_to_data=None):
    # TODO: Wrap the fake dataloader to be able to finish the train loop
    # TODO: Dataset class and dataloader function to handle it

    n_examples = 8192  # number of point clouds
    n_points_per_cloud = 1024  # number of points per point cloud

    train = fake_data_loader(task=task,
                             n_classes=n_classes,
                             n_fake_data=n_examples,
                             batch_size=batch_size,
                             n_points_per_cloud=n_points_per_cloud)

    dev = fake_data_loader(task=task,
                           n_classes=n_classes,
                           n_fake_data=int(n_examples * 0.1),
                           batch_size=batch_size,
                           n_points_per_cloud=n_points_per_cloud)

    test = fake_data_loader(task=task,
                            n_classes=n_classes,
                            n_fake_data=int(n_examples * 0.1),
                            batch_size=batch_size,
                            n_points_per_cloud=n_points_per_cloud)

    return train, dev, test


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

    fake_loader = fake_data_loader(task="semseg")
    for n, (inputs, labels) in enumerate(fake_loader, 0):
        print(inputs.size())
        print(labels.size())
