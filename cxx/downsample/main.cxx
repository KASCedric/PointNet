#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <iostream>
#include <fstream>
#include <string>

#include <pcl/point_types.h>
#include <pcl/search/organized.h>
#include <pcl/search/octree.h>

pcl::PointCloud<pcl::PointXYZI>::Ptr readPointCloud(std::string file);

int main () {
    std::string data_folder = "/media/cedric/Data/Documents/Datasets/kitti_velodyne";
    std::string data_raw = "data";
    std::string data_processed = "processed";

    int sequence = 0;

    std::string points_raw = (boost::format("%s/%s/velodyne/dataset/sequences/%02d/velodyne")
            % data_folder % data_raw % sequence).str();

    for ( boost::filesystem::recursive_directory_iterator end, dir(points_raw); dir != end; ++dir ) {
        std::string points = dir->path().string();
        std::string labels = (boost::format("%s/%s/labels/dataset/sequences/%02d/labels/%s.label")
                % data_folder % data_raw % sequence % dir->path().stem().string()).str();
    }

    std::string labels = (boost::format("%s/%s/velodyne/dataset/sequences/%02d/velodyne/%s.bin")
                          % data_folder % data_raw % sequence % "000000").str();
    std::cout << labels << std::endl;

    pcl::PointCloud<pcl::PointXYZI>::Ptr points = readPointCloud(labels);
    std::cout << points->at(0) << std::endl;

}


pcl::PointCloud<pcl::PointXYZI>::Ptr readPointCloud(std::string file){

    pcl::PointCloud<pcl::PointXYZI>::Ptr points (new pcl::PointCloud<pcl::PointXYZI>);

    // allocate 4 MB buffer (only ~130*4*4 KB are needed)
    int32_t num = 1000000;
    float *data = (float*)malloc(num*sizeof(float));

    // pointers
    float *px = data+0;
    float *py = data+1;
    float *pz = data+2;
    float *pr = data+3;

    // load point cloud
    FILE *stream;
    char *filename = &file[0];
    stream = fopen (filename,"rb");
    num = fread(data,sizeof(float),num,stream)/4;
    for (int32_t i=0; i<num; i++) {
        pcl::PointXYZI point;
        point.x = *px;
        point.y = *py;
        point.z = *pz;
        point.intensity = *pr;
        points->push_back(point);
        px+=4; py+=4; pz+=4; pr+=4;
    }
    fclose(stream);

    return points;
}
