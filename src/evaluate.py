import torch
import numpy as np
import fire
from pathlib import Path
import os
import json
from tqdm import tqdm


from dataloader import semantic_kitti_dataloader
from model import PointNetSemSeg
from utils import load_model, save_eval

# Setting up a random generator seed so that the experiment can be replicated identically on any machine
torch.manual_seed(29071997)
# Setting the device used for the computations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_model(model, loader, criterion):
    validation_loss = accuracy = correct = total = 0
    with torch.no_grad():
        model.eval()
        for j, (dev_inputs, dev_labels) in enumerate(loader, 0):
            dev_inputs = dev_inputs.to(device=device)
            dev_labels = dev_labels.to(device=device)
            dev_outputs, _ = model(dev_inputs)
            dev_loss = criterion(dev_outputs, dev_labels)
            validation_loss += dev_loss.item()
            total += dev_labels.size(0) * dev_labels.size(1)
            dev_outputs = dev_outputs.data.max(1)[1]
            correct += dev_outputs.eq(dev_labels).sum()
        validation_loss /= j
        accuracy += 100 * correct / total
    return validation_loss, accuracy


def evaluate(model, confusion_matrix_file=None, summary_file=None, data_folder=None, sequence=0, n_classes=34, bn=False):

    # Import the names of the classes
    semantic_kitti = "src/semantic-kitti.json"
    with open(semantic_kitti) as json_file:
        all_categories = list(json.load(json_file)["labels"].values())

    if data_folder is None:
        data_folder = "/media/cedric/Data/Documents/Datasets/kitti_velodyne/processed"
    if confusion_matrix_file is None:
        confusion_matrix_file = f"{Path(model).parent}/eval/matrix-{Path(model).stem}.png"
    if summary_file is None:
        summary_file = f"{Path(model).parent}/eval/sum-{Path(model).stem}.png"
    if not os.path.exists(Path(confusion_matrix_file).parent):
        os.makedirs(Path(confusion_matrix_file).parent)
    if not os.path.exists(Path(summary_file).parent):
        os.makedirs(Path(summary_file).parent)

    print(f"Using device: {device}\n"
          f"Data folder: {data_folder}\n"
          f"Sequence: {sequence}\n"
          f"Nb Classes: {n_classes}\n"
          f"Model: {model}\n")

    _, _, test_loader = semantic_kitti_dataloader(sequence, data_folder)  # Dataloader

    # Load model
    net = PointNetSemSeg(n_classes=n_classes, bn=bn).to(device=device)
    net = load_model(net, model)

    # Pretty print
    progress_bar = tqdm(test_loader)
    progress_bar.set_description(f'Evaluating')

    confusion_matrix = torch.zeros(n_classes, n_classes)
    correct = total = 0
    with torch.no_grad():
        for j, (inputs, labels) in enumerate(test_loader, 0):
            progress_bar.update()
            inputs = inputs.to(device=device)
            labels = labels.to(device=device)
            outputs, _ = net(inputs)
            outputs = outputs.data.max(1)[1]
            for t, p in zip(labels.view(-1), outputs.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            total += labels.size(0) * labels.size(1)
            correct += outputs.eq(labels).sum()
        progress_bar.refresh()
        progress_bar.close()
        accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}")

    save_eval(accuracy, confusion_matrix_file, summary_file, confusion_matrix, all_categories)
    print("Done !")


if __name__ == "__main__":
    fire.Fire(evaluate)
