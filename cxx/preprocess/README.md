# C++ code for  <a href="http://www.semantic-kitti.org/" target="_blank">semantic-kitti</a> dataset preprocessing

## Run on Ubuntu 20.04

### Dependencies
```
sudo apt install cmake
sudo apt install libpcl-dev
sudo apt-get install libomp-dev
```

### Build
```
mkdir build
cd build
cmake ..
make
```
### Usage
```
./preprocess --help
```
```
./preprocess --sequence=<sequence> --raw_folder=path/to/raw --processed_folder=path/to/processed --semantic_kitti=path/to/semantic/kitti
```
Where `<sequence>` is the sequence of the kitti velodyne dataset (0 to 10). 
`path/to/processed` is an empty folder where the processed data will be saved.
`path/to/semantic/kitti` is the path to the `root_dir/src/semantic-kitti.json` file used to store the semantic-kitti dataset configuration.
`path/to/raw` is the folder that contains the raw <a href="http://www.semantic-kitti.org/" target="_blank">semantic-kitti</a> dataset.

Raw folder architecture:

![Raw folder architecture](../../misc/raw_folder.png)

Tip: Consider using the `wget` command line to download the kitti odometry velodyne dataset.

Processed folder architecture:

![Processed folder architecture](../../misc/processed_folder.png)


The processed .ply files contain `(x, y, z, label)` data.