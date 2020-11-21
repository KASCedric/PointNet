import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.dataloader import semantic_kitti_dataloader
from src.evaluate import eval_model
from src.model import PointNetSemSeg, compute_regularization
from src.utils import save_model

# Setting up a random generator seed so that the experiment can be replicated identically on any machine
torch.manual_seed(29071997)
# Setting the device used for the computations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():

    with open("configuration.json") as json_file:
        config = json.load(json_file)["train"]

    # Configuration & hyper parameters
    n_epoch = config["n_epoch"]  # Number of epochs
    batch_size = config["batch_size"]  # Batch size
    lr = config["lr"]  # Learning rate
    reg_weight = config["reg_weight"]  # Regularization weight
    n_classes = config["n_classes"]  # number of classes for the classification
    model_save_freq = config["model_save_freq"]  # Save the model each "model_save_freq" batch(es).
    validate = config["validate"]  # If True we use dev data to validate the model while training
    bn = False  # Batch normalization
    models_folder = config["models_folder"]
    data_folder = config["data_folder"]

    if not os.path.exists(models_folder):
        os.makedirs(models_folder)

    # Number of times the statistics (losses, accuracy) are updated / printed during one epoch
    # Note: If n_batches = n_examples / batch_size < n_print then n_print = n_batches
    n_print = 5000
    train_loader, dev_loader, _ = semantic_kitti_dataloader(data_folder)  # Dataloader
    n_batches = len(train_loader)  # number of batches
    # Actualize the running loss each "print_freq" batch during one epoch.
    print_freq = int(n_batches / n_print) if n_batches >= n_print else 1

    # Message before training
    before_training = f"Using device: {device}"
    before_training += "\nTraining PointNet semantic segmentation network. "
    if validate:
        before_training += "Using dev set for validation."
    before_training += f"\nBatch size: {batch_size}; Number of batches: {n_batches}"
    before_training += f"\nThe model will be saved each {model_save_freq} batch(es) and each 2 epoch(s) " \
                       f"at the following dir: {models_folder}/semseg-model-exx-bxxxx.pth"
    print(f"\033[94m{before_training}\033[0m")

    # Semantic segmentation task
    net = PointNetSemSeg(n_classes=n_classes, bn=bn).to(device=device)
    net.train()

    # Negative Log Likelihood Loss function used according to the LogSoftmax final layer activation in the network
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    # Scheduler used to divide by 2 the learning rate each 20 epochs as said in PointNet paper
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Train loop
    for epoch in range(n_epoch):  # loop over the dataset multiple times
        running_loss = 0.0  # Used to monitor the training loss

        # Pretty print
        progress_bar = tqdm(train_loader, bar_format="\033[30m{l_bar}{bar}{r_bar}\033[0m")
        progress_bar.set_description(f'Epoch: {epoch+1}/{n_epoch} - train')
        progress_bar.set_postfix(train_loss='N/A', val_loss='N/A', accuracy='N/A')

        for batch, (train_inputs, train_labels) in enumerate(train_loader, 0):
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

            # Save Model each "model_save_freq" batch(es)
            if batch % model_save_freq == model_save_freq - 1:
                model_path = f"{models_folder}/semseg-model-e{epoch+1:02d}-b{batch+1:04d}.pth"
                save_model(model=net, path=model_path)
                after_saving = f"\nModel successfully saved at: {model_path}\n"
                print(f"\033[92m{after_saving}\033[30m")

            # print statistics
            running_loss += train_loss.item()
            if batch % print_freq == print_freq - 1:  # print every "print_freq" mini-batches
                running_loss /= print_freq  # current loss after iterating over "print_freq" mini-batches
                # We evaluate the model on dev set to compute the validation_loss & the accuracy
                if validate:
                    val_loss, accuracy = eval_model(model=net, device=device, loader=dev_loader, criterion=criterion)
                    # We update the metrics (validation_loss, accuracy)
                    progress_bar.set_postfix(train_loss=f'{running_loss:.2f}',
                                             val_loss=f"{val_loss:.2f}",
                                             accuracy=f"{accuracy:.2f}")
                    # print(f"[Epoch: {epoch+1}/{n_epoch}, Batch: {batch+1}/{n_batches}] \
                    #       train loss: {running_loss:.2f} - val loss: {val_loss:.2f} - acc: {accuracy:.2f}")
                    net.train()
                else:
                    progress_bar.set_postfix(train_loss=f'{running_loss:.2f}',
                                             val_loss="N/A",
                                             accuracy="N/A")
                    # print(f"[Epoch: {epoch+1}/{n_epoch}, Batch:{batch+1}/{n_batches}] train loss: {running_loss:.2f}")
                running_loss = 0.0
        scheduler.step()

        progress_bar.refresh()
        progress_bar.close()

        # Save Model each 2 epoch(s)
        if epoch % 2 == 1:
            model_path = f"{models_folder}/semseg-model-e{epoch+1:02d}-b{n_batches:04d}.pth"
            save_model(model=net, path=model_path)
            after_saving = f"\nModel successfully saved at: {model_path}\n"
            print(f"\033[92m{after_saving}\033[30m")

    after_training = "Finished Training ! \n" \
                     "Next steps: evaluate your model using the command line: \n" \
                     "python src/evaluate.py"
    print(f"\033[94m{after_training}\033[0m")


if __name__ == "__main__":
    train()
