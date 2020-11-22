import torch
import os
import json
from plyfile import PlyData
import numpy as np
from pathlib import Path
import fire

from model import PointNetSemSeg
from utils import load_model, rgb_from_label, save_ply

# Setting the device used for the computations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(model, input_file, output_file, n_classes=34, bn=False):
    print(f"Input file: {input_file}\n"
          f"Output file: {output_file}\n"
          f"Model: {model}\n"
          f"Predicting labels ...")
    # Create output parents folders if not exist
    output_file = Path(output_file)
    if not os.path.exists(output_file.parent):
        os.makedirs(output_file.parent)

    # Import the color map
    semantic_kitti = "src/semantic-kitti.json"
    with open(semantic_kitti) as json_file:
        c_map = json.load(json_file)["color_map"]

    # Load model
    net = PointNetSemSeg(n_classes=n_classes, bn=bn).to(device=device)
    net = load_model(net, model)

    # Load points
    input_type = input_file.split(".")[-1]
    if input_type == "ply":
        with open(input_file, 'rb') as f:
            ply_data = PlyData.read(f)['vertex'].data
            points = np.stack([
                ply_data['x'],
                ply_data['y'],
                ply_data['z']
            ], axis=1)
    elif input_type == "bin":
        points = np.fromfile(input_file, dtype=np.float32).reshape([-1, 4])[:, :3]
    else:
        print("Unknown input type")
        return 1

    # Prediction using cuda if available
    pts_ = torch.from_numpy(points.T).float().unsqueeze(0).to(device=device)
    labels, _ = net(pts_)
    del pts_
    labels = labels.data.max(1)[1]
    labels = np.array(labels, dtype=int).reshape((-1, 1))

    red, green, blue = rgb_from_label(c_map, labels)

    save_ply(points, labels, red, green, blue, output_file)

    print("Done !")


if __name__ == "__main__":
    fire.Fire(predict)
