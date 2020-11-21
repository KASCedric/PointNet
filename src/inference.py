import torch
import os
import json
from plyfile import PlyData
import numpy as np
from model import PointNetSemSeg
from utils import load_model

# Setting the device used for the computations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rgb_from_label(color_map, label):
    keys = list(color_map.keys())

    def wanted_func(label_):
        b = color_map[keys[label_]][0] / 256
        g = color_map[keys[label_]][1] / 256
        r = color_map[keys[label_]][2] / 256
        return r, g, b

    return np.vectorize(wanted_func)(label)


def bin_to_ply(points, labels, red, green, blue, file_out):
    labels = np.array(labels, dtype=int).reshape((-1, 1))
    red = red.reshape((-1, 1))
    green = green.reshape((-1, 1))
    blue = blue.reshape((-1, 1))
    points = np.array(points).T.reshape((-1, 3))
    len_nodes = len(points)
    header = \
        "ply\n" \
        "format binary_little_endian 1.0\n" \
        "element vertex " + str(len_nodes) + "\n" \
                                             "property float x\n" \
                                             "property float y\n" \
                                             "property float z\n" \
                                             "property uint label\n" \
                                             "property float red\n" \
                                             "property float green\n" \
                                             "property float blue\n" \
                                             "end_header\n"

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

    with open(file_out, 'wb') as fp:
        fp.write(bytes(header.encode()))
        fp.write(vertex.tobytes())


if __name__ == "__main__":
    n_classes = 34
    bn = False
    semantic_kitti = "semantic-kitti.json"
    path_to_model = "../models/sample-model.pth"
    # path_to_data = "../data/raw-data.bin"
    path_to_data = "../data/ds-data.ply"
    path_to_processed = "../processed"
    file_out = f"{path_to_processed}/sample-processed.ply"

    if not os.path.exists(path_to_processed):
        os.makedirs(path_to_processed)

    with open(semantic_kitti) as json_file:
        c_map = json.load(json_file)["color_map"]

    net = PointNetSemSeg(n_classes=n_classes, bn=bn).to(device=device)
    net = load_model(net, path_to_model)

    with open(path_to_data, 'rb') as f:
        ply_data = PlyData.read(f)['vertex'].data
        pts = np.stack([
            ply_data['x'],
            ply_data['y'],
            ply_data['z']
        ], axis=1)
    pts = torch.from_numpy(pts.T).float().unsqueeze(0).to(device=device)

    lbl, _ = net(pts)
    lbl = lbl.data.max(1)[1]
    lbl = lbl.cpu()
    pts = pts.cpu()

    # pts = np.fromfile(path_to_data, dtype=np.float32).reshape([-1, 4])
    # pts = torch.from_numpy(pts[:, :3].T).float().unsqueeze(0).to(device=device)
    #
    # lbl, _ = net(pts)
    # lbl = lbl.data.max(1)[1]

    r, g, b = rgb_from_label(c_map, lbl)
    bin_to_ply(pts, lbl, r, g, b, file_out)
