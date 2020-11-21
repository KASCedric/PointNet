#include <string>
#include <exception>

// boost
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>

// pcl to deal with point clouds
#include <pcl/point_types.h>
#include <pcl/search/organized.h>
#include <pcl/search/octree.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/voxel_grid.h>

// OpenMP for multiprocessing
#include <omp.h>

// Json to deal with point clouds
#include "json.hpp"
using json = nlohmann::json;

// Defines & Namespaces
#define THREAD_NUM 6
namespace po = boost::program_options;

// Prototypes
void process(std::string raw_folder,
             const std::string& processed_folder,
             const std::string& semantic_kitti,
             int sequence);
pcl::PointCloud<pcl::PointXYZL>::Ptr down_sample(const pcl::PointCloud<pcl::PointXYZL>::Ptr& input_points);
void save_ply(const pcl::PointCloud<pcl::PointXYZL>::Ptr& points,
              std::string processed_folder,
              int sequence,
              std::string filename);

pcl::PointCloud<pcl::PointXYZL>::Ptr readPointCloud(std::string points_file,
                                                       std::string labels_file,
                                                       json semantic_kitti_json);


int main(int argc, char** argv){

    try {
        std::string raw_folder;
        std::string processed_folder;
        std::string semantic_kitti;
        int sequence;

        po::options_description desc("Allowed options");
        desc.add_options()
                ("help", "produce help message")
                ("raw_folder", po::value<std::string>(), "Path to raw data folder")
                ("processed_folder", po::value<std::string>(), "Path to processed data folder")
                ("semantic_kitti", po::value<std::string>(), "Path to semantic_kitti.json config file")
                ("sequence", po::value<int>(), "Sequence to process");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 0;
        }

        if (vm.count("raw_folder")) {
            raw_folder = vm["raw_folder"].as<std::string>();
        } else {
            raw_folder = "/media/cedric/Data/Documents/Datasets/kitti_velodyne/data";
        }

        if (vm.count("processed_folder")) {
            processed_folder = vm["processed_folder"].as<std::string>();
        } else {
            processed_folder = "/media/cedric/Data/Documents/Datasets/kitti_velodyne/processed";
        }

        if (vm.count("semantic_kitti")) {
            semantic_kitti = vm["semantic_kitti"].as<std::string>();
        } else {
            semantic_kitti = "/home/cedric/Documents/Workspace/ML/PointNet/src/semantic-kitti.json";
        }

        if (vm.count("sequence")) {
            sequence = vm["sequence"].as<int>();
        } else {
            std::cerr << "Sequence value was not set.\n";
            return 1;
        }

        std::cout << "Input folder: " << raw_folder << std::endl
        << "Output folder: " << processed_folder << std::endl
        << "Sequence: " << sequence << std::endl
        << "Preprocessing data..." << std::endl;

        process(raw_folder, processed_folder, semantic_kitti, sequence);

        std::cout << "Done !" << std::endl;

    }
    catch(std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch(...) {
        std::cerr << "Exception of unknown type!\n";
        return 1;
    }
    return 0;
}


pcl::PointCloud<pcl::PointXYZL>::Ptr readPointCloud(std::string points_file,
                                                       std::string labels_file,
                                                       json semantic_kitti_json){

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
//        pcl::PointXYZRGBL point;
        pcl::PointXYZL point;
        uint label = *labels_data;
        label &= 0xFFFFu;
        uint label_norm = semantic_kitti_json["labels_norm"][std::to_string(label)];
//        std::vector<int> bgr = semantic_kitti_json["color_map"][std::to_string(label)];

        point.x = *px;
        point.y = *py;
        point.z = *pz;
        point.label = label_norm;
//        point.b = bgr.at(0);
//        point.g = bgr.at(1);
//        point.r = bgr.at(2);

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
    float leaf_size = 0.1f;
    pcl::PointCloud<pcl::PointXYZL>::Ptr output_points(new pcl::PointCloud<pcl::PointXYZL>);
    pcl::VoxelGrid<pcl::PointXYZL> sor;
    sor.setInputCloud (input_points);
    sor.setLeafSize (leaf_size, leaf_size, leaf_size);
    sor.filter (*output_points);
    return(output_points);
}


void process(std::string raw_folder,
             const std::string& processed_folder,
             const std::string& semantic_kitti,
             int sequence){


    std::ifstream semantic_kitti_stream(semantic_kitti);
    json semantic_kitti_json = json::parse(semantic_kitti_stream);

    std::string points_raw_folder = (boost::format("%s/velodyne/dataset/sequences/%02d/velodyne")
                                     % raw_folder % sequence).str();

    std::string labels_raw_folder = (boost::format("%s/labels/dataset/sequences/%02d/labels")
                                     % raw_folder % sequence).str();

    boost::filesystem::recursive_directory_iterator end, dir(points_raw_folder);
    std::vector<boost::filesystem::directory_entry> dirs;

    std::copy(dir, end, back_inserter(dirs));

    omp_set_num_threads(THREAD_NUM);
    #pragma omp parallel for \
        default(none) \
        shared(dirs, labels_raw_folder, processed_folder, sequence, semantic_kitti_json)
    for(std::size_t i=0; i<dirs.size(); ++i){

        std::string filename = dirs[i].path().stem().string();
        const std::string& points_file = dirs[i].path().string();
        std::string labels_file = (boost::format("%s/%s.label") % labels_raw_folder % filename).str();

        pcl::PointCloud<pcl::PointXYZL>::Ptr points = readPointCloud(points_file, labels_file, semantic_kitti_json);
        pcl::PointCloud<pcl::PointXYZL>::Ptr points_ds;
        points_ds = down_sample(points);
        save_ply(points_ds, processed_folder, sequence, filename);
    }
}