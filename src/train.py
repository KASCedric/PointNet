import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.dataset import dataloader
from src.model import PointNetCls, PointNetSemSeg, compute_regularization
from src.utils import white, blue, green, save_model, red

# Setting up a random generator seed so that the experiment can be replicated identically on any machine
torch.manual_seed(29071997)
# Setting the device used for the computations
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


def train(bn=True):

    with open("configuration.json") as json_file:
        config = json.load(json_file)["train"]

    # Configuration & hyper parameters
    n_epoch = config["n_epoch"]  # Number of epochs
    batch_size = config["batch_size"]  # Batch size
    lr = config["lr"]  # Learning rate
    reg_weight = config["reg_weight"]  # Regularization weight
    n_classes = config["n_classes"]  # number of classes for the classification
    task = config["task"]  # Do classification (cls) or semantic segmentation (semseg)
    model_save_freq = config["model_save_freq"]  # Save the model each "model_save_freq" epoch(s).
    validate = config["validate"]  # If True we use dev data to validate the model while training

    # Number of times the statistics (losses, accuracy) are updated / printed during one epoch
    # Note: If n_batches = n_examples / batch_size < n_print then n_print = n_batches
    n_print = 5000

    # Dataloader
    train_loader, dev_loader, _ = dataloader()
    if validate:
        assert dev_loader is not None, "The dev loader should not be None if you want to validate the model"

    n_batches = len(train_loader)  # number of batches

    # Do a print each "print_freq" batch during one epoch.
    print_freq = int(n_batches / n_print) if n_batches >= n_print else 1

    network = "classification" if task == "cls" else "semantic segmentation"

    before_training = "Training PointNet {} network, and using dev set for validation.".format(network) if validate \
        else "Training PointNet {} network.".format(network)
    before_training += "\nThe models will be saved each {} epoch(s) at the following dir: models/{}"\
        .format(model_save_freq, task)
    print(blue(before_training))

    if task == "cls":
        # Classification task
        net = PointNetCls(n_classes=n_classes, bn=bn, do=True).to(device=device)
    elif task == "semseg":
        # Semantic segmentation task
        net = PointNetSemSeg(n_classes=n_classes, bn=bn).to(device=device)
    else:
        assert False, "Unknown task. Task should be 'cls' for classification or 'semseg' for semantic segmentation"
    net.train()

    # Negative Log Likelihood Loss function used according to the LogSoftmax final layer activation in our networks
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    # Scheduler used to divide by 2 the learning rate each 20 epochs as said in PointNet paper
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    running_loss_before = 0.0
    # Train loop
    for epoch in range(n_epoch):  # loop over the dataset multiple times
        running_loss = 0.0  # Used to monitor the training loss
        validation_loss = 0.0  # Used to monitor the validation loss
        accuracy = 0.0  # Used to monitor the accuracy

        # Pretty print
        progress_bar = tqdm(train_loader, bar_format=white("{l_bar}{bar}{r_bar}"))
        progress_bar.set_description('Epoch: {:d}/{:d} - train'.format(epoch + 1, n_epoch))
        if dev_loader is None:
            progress_bar.set_postfix(loss='N/A')
        else:
            progress_bar.set_postfix(train_loss='N/A', dev_loss='N/A', accuracy='N/A')

        for i, (train_inputs, train_labels) in enumerate(train_loader, 0):
            progress_bar.update()

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

                # We evaluate the model on dev set to compute the validation_loss & the accuracy
                if validate:
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        net.eval()
                        for j, (dev_inputs, dev_labels) in enumerate(dev_loader, 0):
                            dev_inputs = dev_inputs.to(device=device)
                            dev_labels = dev_labels.to(device=device)
                            dev_outputs, _ = net(dev_inputs)
                            dev_loss = criterion(dev_outputs, dev_labels)
                            validation_loss += dev_loss.item()
                            total += dev_labels.size(0) * (1 if task == "cls" else dev_labels.size(1))
                            dev_outputs = dev_outputs.data.max(1)[1]
                            correct += dev_outputs.eq(dev_labels).sum()
                        validation_loss /= j
                        accuracy += 100 * correct / total
                    net.train()

                    # We print the metrics (running_loss, validation_loss, accuracy)
                    progress_bar.set_postfix(train_loss='{:.2f}'.format(running_loss),
                                             dev_loss='{:.2f}'.format(validation_loss),
                                             accuracy='{:.2f}'.format(accuracy))
                    # print('[Epoch: %d / %d, Batch: %d / %d] train loss: %.3f - dev loss: %.3f - accuracy: %.3f' %
                    #       (epoch + 1, n_epoch, i + 1, n_batches, running_loss, validation_loss, accuracy))
                else:
                    # The dev set is not available, thus we only print the running_loss
                    if running_loss_before < running_loss:
                        status = green("↗")
                    else:
                        status = red("↘")

                    progress_bar.set_postfix(loss='{:.2f} {}'.format(running_loss, status))
                    # print('[Epoch: %d / %d, Batch: %d / %d] train loss: %.3f' %
                    #       (epoch + 1, n_epoch, i + 1, n_batches, running_loss))

                running_loss_before = running_loss
                running_loss = 0.0
                validation_loss = 0.0
                accuracy = 0.0

        scheduler.step()

        progress_bar.refresh()
        progress_bar.close()

        if epoch % model_save_freq == model_save_freq - 1:
            model_path = config["models"] + task + "-model-{:05d}.pth".format(epoch+1)
            save_model(model=net, path=model_path)
            after_saving = "Model successfully saved at: {}".format(model_path)
            print(green(after_saving))

    after_training = "Finished Training ! \n" \
                     "Next steps: evaluate your model using the command line: \n" \
                     "python src/evaluate.py"
    print(blue(after_training))


if __name__ == "__main__":
    train(bn=False)
