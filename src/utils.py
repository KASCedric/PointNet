import torch
import numpy as np
from pathlib import Path
import json


def white(x):
    #  Wrapper to print white text in terminal
    return '\033[30m' + str(x) + '\033[0m'


def blue(x):
    #  Wrapper to print blue text in terminal
    return '\033[94m' + str(x) + '\033[0m'


def green(x):
    #  Wrapper to print blue text in terminal
    return '\033[92m' + str(x) + '\033[0m'


def red(x):
    #  Wrapper to print blue text in terminal
    return '\033[91m' + str(x) + '\033[0m'


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def get_input_file_dir(dataset_root, sequence, cloud):
    path_to_sequence = Path("dataset/sequences") / sequence
    pc_dir = dataset_root / "velodyne" / path_to_sequence / "velodyne" / cloud
    label_dir = dataset_root / "labels" / path_to_sequence / "labels" / cloud
    return pc_dir.with_suffix(".bin"), label_dir.with_suffix(".label")


def rgb_from_label(label):
    with open("semantic-kitti.json") as json_file:
        color_map = json.load(json_file)["color_map"]

    def wanted_func(label_):
        b = color_map[str(label_)][0] / 256
        g = color_map[str(label_)][1] / 256
        r = color_map[str(label_)][2] / 256
        return r, g, b
    return np.vectorize(wanted_func)(label)


def true_label(label):
    with open("semantic-kitti.json") as json_file:
        labels_names = json.load(json_file)["labels"]

    def wanted_func(label_):
        return list(labels_names.keys()).index(str(label_))

    return np.vectorize(wanted_func)(label)


def bin_to_ply(pc_dir, label_dir, file_out):

    nodes = np.fromfile(pc_dir, dtype=np.float32).reshape([-1, 4])
    labels = np.fromfile(label_dir, dtype=np.int32).reshape((-1))
    labels = labels & 0xFFFF  # get lower half for semantics
    labels = labels.reshape([-1, 1])
    r, g, b = rgb_from_label(labels[:, 0])

    len_nodes = len(nodes)

    header = \
        "ply\n" \
        "format binary_little_endian 1.0\n" \
        "element vertex "+str(len_nodes)+"\n" \
        "property float x\n" \
        "property float y\n" \
        "property float z\n" \
        "property float intensity\n" \
        "property uint label\n" \
        "property float red\n" \
        "property float green\n" \
        "property float blue\n" \
        "end_header\n"

    d_type_vertex = [('vertex', '<f4', 8)]
    vertex = np.empty(len_nodes, dtype=d_type_vertex)
    vertex['vertex'] = np.stack((nodes[:, 0],
                                 nodes[:, 1],
                                 nodes[:, 2],
                                 nodes[:, 3],
                                 labels[:, 0],
                                 r[:],
                                 g[:],
                                 b[:]),
                                axis=-1)

    with open(file_out, 'wb') as fp:
        fp.write(bytes(header.encode()))
        fp.write(vertex.tostring())


if __name__ == "__main__":
    data_folder = "/media/cedric/Data/Documents/Datasets/kitti_velodyne"
    data_raw = "data"
    sequence = 4
    pts = "/media/cedric/Data/Documents/Datasets/kitti_velodyne/data/velodyne/dataset/sequences/00/velodyne/000000.bin"
    lbls = "/media/cedric/Data/Documents/Datasets/kitti_velodyne/data/labels/dataset/sequences/00/labels/000000.label"

    bin_to_ply(pts, lbls, "test.ply")

    # print(true_label([0, 11]))

    # file = 0
    # pc_dir = "%s/%s/velodyne/dataset/sequences/%02d/velodyne/%06d.bin" % (data_folder, data_raw, sequence, file)
    # points = np.fromfile(pc_dir, dtype=np.float32).reshape([-1, 4])
    # print(points[0, :])
    #
    # pc_dir = "/media/cedric/Data/Documents/Datasets/kitti_velodyne/processed/00/000000.ply"
    # from plyfile import PlyData
    # with open(pc_dir, 'rb') as f:
    #     plydata = PlyData.read(f)
    #     num_verts = plydata['vertex'].count
    #     vertices = np.zeros(shape=[num_verts, 4], dtype=np.float32)
    #     vertices[:,0] = plydata['vertex'].data['x']
    #     vertices[:,1] = plydata['vertex'].data['y']
    #     vertices[:,2] = plydata['vertex'].data['z']
    #     vertices[:, 3] = plydata['vertex'].data['label']
    #
    # print(vertices[20348, :])

    # lbl_f = "/media/cedric/Data/Documents/Datasets/kitti_velodyne/data/labels/dataset/sequences/00/labels/000000.label"
    # lbl = np.fromfile(lbl_f, dtype=np.int32).reshape((-1))
    # lbl2 = lbl & 0xFFFF  # get lower half for semantics

    # for sequence in range(11):
    #     length = []
    #     for file in range(50, 100):
    #         pc_dir = "%s/%s/velodyne/dataset/sequences/%02d/velodyne/%06d.bin" % (data_folder, data_raw, sequence, file)
    #         points = np.fromfile(pc_dir, dtype=np.float32).reshape([-1, 4])
    #         length.append(points.shape[0])
    #     print(max(length))

    # with open("semantic-kitti.json") as json_file:
    #     labels_names = json.load(json_file)["labels"]
