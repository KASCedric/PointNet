import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import fake_data_loader
from src.model import PointNetCls, compute_regularization

# Setting up a random generator seed so that the experiment can be replicated identically on any machine
torch.manual_seed(29071997)
# Setting the device used for the computations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration & hyper parameters
n_epoch = 2  # Number of epochs
batch_size = 32  # Batch size
lr = 0.001  # Learning rate
reg_weight = 0.0001  # Regularization weight
n_classes = 5  # number of classes for the classification
n_fake_data = 8192  # number of point clouds
n_points_per_cloud = 1024  # number of points per point cloud

point_net_cls = PointNetCls(n_classes=n_classes, bn=True, do=True).to(device=device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(point_net_cls.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# Dataset
fake_loader = fake_data_loader(task="cls",
                               n_classes=n_classes,
                               n_fake_data=n_fake_data,
                               batch_size=batch_size,
                               n_points_per_cloud=n_points_per_cloud)
data_loader = fake_loader

for epoch in range(n_epoch):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(data_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device=device)
        labels = labels.to(device=device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs, feature_matrix = point_net_cls(inputs)
        regularization_term = compute_regularization(feature_matrix) * reg_weight
        loss = criterion(outputs, labels) + regularization_term
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0
    scheduler.step()


print('Finished Training')
