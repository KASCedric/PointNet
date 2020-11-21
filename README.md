# PointNet
Own implementation of the [PointNet](https://arxiv.org/abs/1612.00593) __semantic segmentation__ neurral network, trained on the [semantic-kitti](http://www.semantic-kitti.org/) dataset.



## Installation on Ubuntu 20.04

##### Install virtualenv
```
sudo apt-get update
sudo apt-get install build-essential libssl-dev libffi-dev python-dev
sudo apt install python3-pip
sudo pip3 install virtualenv 
```
##### Install Requirements
```
virtualenv -p python3 venv
source venv/bin/activate
which python  # Check which python you use
python --version  # Check python version
pip install -r requirements.txt
```

## Usage


### Prediction

##### Model
A pre-trained model is available [here](). Use the following command line to download it.
```
mkdir -p models
wget url/to/model models/sample-model.pth
```
##### Data
- You can directly download a [downsampled point cloud]() for the inference:
    ```
    mkdir -p data
    wget url/to/downsample data/ds-data.ply
    ```
- Or download a [raw data]() and downsample it so that is has the same distribution as the pointclouds used to train the model:
    ```
    mkdir -p data
    wget url/to/data data/raw-data.bin
    cxx/preprocess/build/downsample --input_file=data/raw-data.bin --output_file=data/ds-data.ply
    ```
    Note: Please read the [documentation](https://github.com/KASCedric/PointNet/tree/main/cxx/preprocess) to build the cxx `downsample` executable 

##### Inference 

Use the following command to predict the labels:
```
python src/inference.py --model=models/sample-model.pth --data=data/ds-data.bin
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
