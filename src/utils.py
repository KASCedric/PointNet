import torch
import numpy as np


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
