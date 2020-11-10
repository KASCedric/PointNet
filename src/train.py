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
    # TODO: Pretty print training steps on terminal, model type, etc...
    # TODO: Create config.yml / json ? file and parse it

    # TODO: Plot the learning curves tensorboard
    # TODO: Save the model frequently
    # TODO: fire.Fire arg parsing

    # Configuration & hyper parameters
    n_epoch = 2  # Number of epochs
    batch_size = 32  # Batch size
    lr = 0.001  # Learning rate
    reg_weight = 0.0001  # Regularization weight

    n_classes = 5  # number of classes for the classification
    n_examples = 8192  # number of point clouds
    n_points_per_cloud = 1024  # number of points per point cloud
    task = "cls"  # Do classification (cls) or semantic segmentation (semseg)

    # Number of prints to do during one epoch
    # Note: If n_batches = n_examples / batch_size < n_print then n_print = n_batches
    n_print = 10

    n_batches = int(n_examples / batch_size)  # number of batches

    # Do a print each "print_freq" batch during one epoch.
    if n_batches < n_print:
        print_freq = 1
    else:
        print_freq = int(n_batches / n_print)

    # Dataloader
    train_loader = fake_data_loader(task=task,
                                    n_classes=n_classes,
                                    n_fake_data=n_examples,
                                    batch_size=batch_size,
                                    n_points_per_cloud=n_points_per_cloud)
    dev_loader = fake_data_loader(task=task,
                                  n_classes=n_classes,
                                  n_fake_data=int(n_examples*0.1),
                                  batch_size=batch_size,
                                  n_points_per_cloud=n_points_per_cloud)
    dev_loader = None

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
        running_loss = 0.0  # Used to monitor the training loss
        validation_loss = 0.0  # Used to monitor the validation loss
        accuracy = 0.0  # Used to monitor the accuracy
        for i, (train_inputs, train_labels) in enumerate(train_loader, 0):
            train_inputs = train_inputs.to(device=device)
            train_labels = train_labels.to(device=device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # feature_matrix: 64x64 features transform matrix. Used to add regularization to the loss
            train_outputs, feature_matrix = net(train_inputs)
            regularization_term = compute_regularization(feature_matrix) * reg_weight
            train_loss = criterion(train_outputs, train_labels) + regularization_term
            train_loss.backward()
            optimizer.step()

            # print statistics
            running_loss += train_loss.item()
            if i % print_freq == print_freq - 1:  # print every "print_freq" mini-batches
                running_loss /= print_freq  # current loss after iterating over "print_freq" mini-batches

                if dev_loader is not None:
                    # We evaluate the model on dev set
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for j, (dev_inputs, dev_labels) in enumerate(dev_loader, 0):
                            dev_inputs = dev_inputs.to(device=device)
                            dev_labels = dev_labels.to(device=device)
                            dev_outputs, _ = net(dev_inputs)
                            dev_loss = criterion(dev_outputs, dev_labels)
                            validation_loss += dev_loss.item()
                            total += dev_labels.size(0) * (1 if task == "cls" else dev_labels.size(1))
                            dev_outputs = dev_outputs.data.max(1)[1]
                            # correct += dev_outputs.eq(dev_labels, 0).sum().item()
                            correct += (dev_outputs == dev_labels).sum()
                        validation_loss /= j
                        accuracy += 100 * correct / total

                    # We print the metrics (running_loss, validation_loss, accuracy)
                    print('[Epoch: %d / %d, Batch: %d / %d] train loss: %.3f - dev loss: %.3f - accuracy: %.3f' %
                          (epoch + 1, n_epoch, i + 1, n_batches, running_loss, validation_loss, accuracy))
                else:
                    # The dev set is not available, thus we only print the running_loss
                    print('[Epoch: %d / %d, Batch: %d / %d] train loss: %.3f' %
                          (epoch + 1, n_epoch, i + 1, n_batches, running_loss))

                running_loss = 0.0
                validation_loss = 0.0
                accuracy = 0.0

        scheduler.step()

    print('Finished Training')


if __name__ == "__main__":
    train("test")
