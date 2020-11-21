# PointNet
Own implementation of the [PointNet](https://arxiv.org/abs/1612.00593) __semantic segmentation__ neurral network, trained on the [semantic-kitti](http://www.semantic-kitti.org/) dataset.



## Installation on Ubuntu 20.04

#### Requirements
- Install virtualenv
    ```
    sudo apt-get update
    sudo apt-get install build-essential libssl-dev libffi-dev python-dev
    sudo apt install python3-pip
    sudo pip3 install virtualenv 
    ```
- Install Requirements
    ```
    virtualenv -p python3 venv
    source venv/bin/activate
    which python  # Check which python you use
    python --version  # Check python version
    pip install -r requirements.txt
    ```

## Usage


### Prediction


You can download a [pre-trained model]() and [sample point cloud]() as following:
```
mkdir -p models data
wget url/to/model models/sample-model.pth
wget url/to/data data/sample-data.bin
```

And use the following command to predict the labels:
```
python src/inference.py --model=models/sample-model.pth --data=data/sample-data.bin
```

### Train
```
python src/train.py --n_epoch=10 --validate=True --models_folder=path/to/models --data_folder=path/to/data --sequence=0
```
Where `path/to/models` is the folder where the trained models are saved.
`path/to/data` contains the training dataset. Folder `path/to/data/00` correspond to sequence 0, etc.

Use command `python src/train.py --help` to produce help message

Data folder architecture:

![Processed folder architecture](misc/train_data_folder.png)


![Training the model](misc/training.png)

### Evaluate

Under construction
