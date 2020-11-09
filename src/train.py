import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import fake_data_loader
from src.model import PointNetCls, PointNetSemSeg, compute_regularization

# Setting up a random generator seed so that the experiment can be replicated identically on any machine
torch.manual_seed(29071997)
# Setting the device used for the computations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(config):
    # TODO: Create config.yml / json ? file and parse it
    # TODO: Handle train and eval datasets to train the model
    # TODO: Plot the learning curves
    # TODO: Pretty print training steps on terminal
    # TODO: Save the model frequently
    # TODO: fire.Fire arg parsing

    # Configuration & hyper parameters
    n_epoch = 2  # Number of epochs
    batch_size = 32  # Batch size
    lr = 0.001  # Learning rate
    reg_weight = 0.0001  # Regularization weight

    n_classes = 5  # number of classes for the classification
    n_fake_data = 8192  # number of point clouds
    n_points_per_cloud = 1024  # number of points per point cloud
    task = "semseg"

    # Dataloader
    fake_loader = fake_data_loader(task=task,
                                   n_classes=n_classes,
                                   n_fake_data=n_fake_data,
                                   batch_size=batch_size,
                                   n_points_per_cloud=n_points_per_cloud)
    data_loader = fake_loader

    if task == "cls":
        # Classification task
        net = PointNetCls(n_classes=n_classes, bn=True, do=True).to(device=device)
    elif task == "semseg":
        # Semantic segmentation task
        net = PointNetSemSeg(n_classes=n_classes, bn=True).to(device=device)
    else:
        assert False, "Unknown task. Task should be 'cls' for classification or 'semseg' for semantic segmentation"

    # Negative Log Likelihood Loss function used according to the LogSoftmax final layer activation in our networks
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    # Scheduler used to divide by 2 the learning rate each 20 epochs as said in PointNet paper
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Train loop
    for epoch in range(n_epoch):  # loop over the dataset multiple times
        running_loss = 0.0  # Used to monitor the loss over the training
        for i, (inputs, labels) in enumerate(data_loader, 0):
            inputs = inputs.to(device=device)
            labels = labels.to(device=device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # feature_matrix: 64x64 features transform matrix. Used to add regularization to the loss
            outputs, feature_matrix = net(inputs)
            regularization_term = compute_regularization(feature_matrix) * reg_weight
            loss = criterion(outputs, labels) + regularization_term
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # print every 10 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
        scheduler.step()

    print('Finished Training')


if __name__ == "__main__":
    train("test")
