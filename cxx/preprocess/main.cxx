#include <string>

// boost
#include <boost/filesystem.hpp>
#include <boost/format.hpp>

// pcl
#include <pcl/point_types.h>
#include <pcl/search/organized.h>
#include <pcl/search/octree.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/voxel_grid.h>

// OpenMP
#include <omp.h>

// Prototypes
pcl::PointCloud<pcl::PointXYZL>::Ptr readPointCloud(std::string points_file, std::string labels_file);
pcl::PointCloud<pcl::PointXYZL>::Ptr down_sample(const pcl::PointCloud<pcl::PointXYZL>::Ptr& input_points);
void save_ply(const pcl::PointCloud<pcl::PointXYZL>::Ptr& points,
              std::string processed_folder,
              int sequence,
              std::string filename);

int main () {

    std::string data_folder = "/media/cedric/Data/Documents/Datasets/kitti_velodyne";
    std::string data_raw = "data";
    std::string raw_folder = "/media/cedric/Data/Documents/Datasets/kitti_velodyne/data";
    std::string processed_folder = "/media/cedric/Data/Documents/Datasets/kitti_velodyne/processed";

    int sequence = 0;

    std::string points_raw_folder = (boost::format("%s/velodyne/dataset/sequences/%02d/velodyne")
            % raw_folder % sequence).str();

    std::string labels_raw_folder = (boost::format("%s/labels/dataset/sequences/%02d/labels")
            % raw_folder % sequence).str();

    for ( boost::filesystem::recursive_directory_iterator end, dir(points_raw_folder); dir != end; ++dir ) {
        std::string filename = dir->path().stem().string();
        std::string points_file = dir->path().string();
        std::string labels_file = (boost::format("%s/%s.label") % labels_raw_folder % filename).str();

        pcl::PointCloud<pcl::PointXYZL>::Ptr points = readPointCloud(points_file, labels_file);
        pcl::PointCloud<pcl::PointXYZL>::Ptr points_ds;
        points_ds = down_sample(points);
        save_ply(points_ds, processed_folder, sequence, filename);
    }
    std::cout << "Done !" << std::endl;

//    std::string points_file = (boost::format("%s/%s.bin")
//                               % points_raw_folder % "000000").str();
//    std::cout << points_file << std::endl;
//
//    std::string labels_file = (boost::format("%s/%s.label")
//                               % labels_raw_folder % "000000").str();
//    std::cout << labels_file << std::endl;

//    pcl::PointCloud<pcl::PointXYZL>::Ptr points = readPointCloud(points_file, labels_file);
//    pcl::PointCloud<pcl::PointXYZL>::Ptr points_ds;
//    points_ds = down_sample(points);
//    save_ply(points_ds, processed_folder, 0, "000000");

}


pcl::PointCloud<pcl::PointXYZL>::Ptr readPointCloud(std::string points_file, std::string labels_file){

    pcl::PointCloud<pcl::PointXYZL>::Ptr points (new pcl::PointCloud<pcl::PointXYZL>);

    // allocate 4 MB buffer (only ~130*4*4 KB are needed)
    int32_t points_num = 1000000;
    int32_t labels_num = 1000000;
    auto *points_data = (float*)malloc(points_num*sizeof(float));
    auto *labels_data = (int32_t*)malloc(labels_num*sizeof(int32_t));

    // pointers
    float *px = points_data+0;
    float *py = points_data+1;
    float *pz = points_data+2;

    // load point cloud and labels
    FILE *points_stream;
    FILE *labels_stream;
    char *points_filename = &points_file[0];
    char *labels_filename = &labels_file[0];
    points_stream = fopen (points_filename,"rb");
    labels_stream = fopen (labels_filename,"rb");
    points_num = fread(points_data,sizeof(float),points_num,points_stream)/4;
    labels_num = fread(labels_data,sizeof(float),labels_num,labels_stream);

    for (int32_t i=0, j=0; i<points_num && j<labels_num; i++, j++) {
        pcl::PointXYZL point;
        uint label = *labels_data;

        point.x = *px;
        point.y = *py;
        point.z = *pz;
        point.label = label & 0xFFFFu;
        points->push_back(point);
        px+=4; py+=4; pz+=4; labels_data+=1;
    }
    fclose(points_stream);
    fclose(labels_stream);

    return points;
}


void save_ply(const pcl::PointCloud<pcl::PointXYZL>::Ptr& points,
              std::string processed_folder,
              int sequence,
              std::string filename){

    std::string ply_file = (boost::format("%s/%02d/%s.ply") % processed_folder % sequence % filename).str();
    boost::filesystem::create_directories(boost::filesystem::path(ply_file).parent_path());
    pcl::io::savePLYFileBinary(ply_file, *points);

}


pcl::PointCloud<pcl::PointXYZL>::Ptr down_sample(const pcl::PointCloud<pcl::PointXYZL>::Ptr& input_points){
    float leaf_size = 0.07f;
    pcl::PointCloud<pcl::PointXYZL>::Ptr output_points(new pcl::PointCloud<pcl::PointXYZL>);;
    pcl::VoxelGrid<pcl::PointXYZL> sor;
    sor.setInputCloud (input_points);
    sor.setLeafSize (leaf_size, leaf_size, leaf_size);
    sor.filter (*output_points);
    return(output_points);
}
