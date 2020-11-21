import torch


def eval_model(model, device, loader, criterion):
    # Evaluate , confusion matrix, FP, FN, TP, TN, Accuracy, Precision, Recall, F1 Score, etc...
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
