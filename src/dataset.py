import torch
from torch.utils import data


def dataloader(path_to_data):
    # TODO: Wrap the fake dataloader to be able to finish the train loop
    # TODO: Dataset class and dataloader function to handle it
    return 0


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
