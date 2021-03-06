import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def split(n, percentages):
    assert sum(percentages) == 1, "Sum of percentages must be 1"
    xs = n * np.array(percentages)
    rs = [round(x) for x in xs]
    k = sum(xs) - sum(rs)
    assert k == round(k)
    fs = [x - round(x) for x in xs]
    indices = [i for order, (e, i) in enumerate(reversed(sorted((e, i) for i, e in enumerate(fs)))) if order < k]
    ys = [R + 1 if i in indices else R for i, R in enumerate(rs)]
    return ys


def rgb_from_label(color_map, label):
    keys = list(color_map.keys())

    def wanted_func(label_):
        b = color_map[keys[label_]][0] / 256
        g = color_map[keys[label_]][1] / 256
        r = color_map[keys[label_]][2] / 256
        return r, g, b

    return np.vectorize(wanted_func)(label)


def save_ply(points, labels, red, green, blue, output_file):
    len_nodes = len(points)

    header = f"ply\n" \
             f"format binary_little_endian 1.0\n" \
             f"element vertex {len_nodes}\n" \
             f"property float x\n" \
             f"property float y\n" \
             f"property float z\n" \
             f"property uint label\n" \
             f"property float red\n" \
             f"property float green\n" \
             f"property float blue\n" \
             f"end_header\n"

    d_type_vertex = [('vertex', '<f4', 7)]
    vertex = np.empty(len_nodes, dtype=d_type_vertex)
    vertex['vertex'] = np.stack((points[:, 0],
                                 points[:, 1],
                                 points[:, 2],
                                 labels[:, 0],
                                 red[:, 0],
                                 green[:, 0],
                                 blue[:, 0]),
                                axis=-1)

    with open(output_file, 'wb') as fp:
        fp.write(bytes(header.encode()))
        fp.write(vertex.tobytes())


# function from https://deeplizard.com/learn/video/0LhiS6yu2qQ and https://github.com/nikitakaraevv/pointnet
def save_eval(accuracy, confusion_matrix_file, summary_file, confusion_matrix, classes):
    cmap = plt.get_cmap("Blues")
    confusion_matrix = np.array(confusion_matrix)
    numerator = confusion_matrix.astype('float')
    counts = confusion_matrix.sum(axis=1)
    denominator = counts[:, np.newaxis]
    confusion_matrix = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)

    tick_marks = np.arange(len(classes))

    plt.figure(figsize=(20, 20))
    title = f'Normalized confusion matrix, Accuracy: {accuracy:.2f}'
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, f"{confusion_matrix[i, j]:.2f}",
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(confusion_matrix_file)
    print(f"Confusion matrix saved at: {confusion_matrix_file}")

    plt.figure(figsize=(20, 8))
    title = f'Normalized distribution of points per classe. Number of points: {sum(counts):.0f}'
    plt.title(title)
    plt.bar(classes, 100*counts/sum(counts))
    for i, v in enumerate(counts):
        plt.text(i, 100*v/sum(counts)+.1, f"{v:.0f}", color='black',
                 horizontalalignment="center")
    plt.xticks(tick_marks, classes, rotation=45)
    plt.ylabel('Percentage (%)')
    plt.xlabel('Classes')
    plt.tight_layout()
    plt.savefig(summary_file)
    print(f"Summary saved at: {summary_file}")
