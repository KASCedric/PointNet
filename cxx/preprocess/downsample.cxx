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
//#include <pcl/io/ply_io.h>
#include "pcl/io/ply_io.h"
#include <pcl/filters/voxel_grid.h>

// Namespaces
namespace po = boost::program_options;

// Prototypes
pcl::PointCloud<pcl::PointXYZ>::Ptr down_sample(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_points);
void save_ply(const pcl::PointCloud<pcl::PointXYZ>::Ptr& points, std::string output_file);
pcl::PointCloud<pcl::PointXYZ>::Ptr readPointCloud(std::string points_file, const std::string& input_type);


int main(int argc, char** argv){

    try {
        std::string input_file;
        std::string input_type;
        std::string output_file;

        po::options_description desc("Allowed options");
        desc.add_options()
                ("help", "produce help message")
                ("input_file", po::value<std::string>(), "Input file (.bin or ply)")
                ("input_type", po::value<std::string>(), "Input type (bin or ply)")
                ("output_file", po::value<std::string>(), "Output file (.ply)");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 0;
        }

        if (vm.count("input_file")) {
            input_file = vm["input_file"].as<std::string>();
        } else {
            std::cerr << "Argument required.\n";
            return 1;
        }

        if (vm.count("input_type")) {
            input_type = vm["input_type"].as<std::string>();
        } else {
            std::cerr << "Argument required.\n";
            return 1;
        }

        if (vm.count("output_file")) {
            output_file = vm["output_file"].as<std::string>();
        } else {
            std::cerr << "Argument required.\n";
            return 1;
        }

        std::cout << "Input file: " << input_file << std::endl
                << "Output file: " << output_file << std::endl
                << "Downsampling the pointcloud ..." << std::endl;

        pcl::PointCloud<pcl::PointXYZ>::Ptr points = readPointCloud(input_file, input_type);
        pcl::PointCloud<pcl::PointXYZ>::Ptr points_ds = down_sample(points);
        save_ply(points_ds, output_file);

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






void save_ply(const pcl::PointCloud<pcl::PointXYZ>::Ptr& points, std::string output_file){
    boost::filesystem::create_directories(boost::filesystem::path(output_file).parent_path());
    pcl::io::savePLYFileBinary(output_file, *points);
}


pcl::PointCloud<pcl::PointXYZ>::Ptr down_sample(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_points){
    float leaf_size = 0.1f;
    pcl::PointCloud<pcl::PointXYZ>::Ptr output_points(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud (input_points);
    sor.setLeafSize (leaf_size, leaf_size, leaf_size);
    sor.filter (*output_points);
    return(output_points);
}


pcl::PointCloud<pcl::PointXYZ>::Ptr readPointCloud(std::string points_file, const std::string& input_type){

    pcl::PointCloud<pcl::PointXYZ>::Ptr points (new pcl::PointCloud<pcl::PointXYZ>);

    if (input_type == "ply"){
        std::cout << "Using ply reader ..." << std::endl;
        pcl::io::loadPLYFile(points_file, *points);
    }
    else if(input_type == "bin"){
        std::cout << "Using bin reader ..." << std::endl;
        // allocate 4 MB buffer (only ~130*4*4 KB are needed)
        int32_t points_num = 1000000;
        auto *points_data = (float*)malloc(points_num*sizeof(float));

        // pointers
        float *px = points_data+0;
        float *py = points_data+1;
        float *pz = points_data+2;

        // load point cloud
        FILE *points_stream;
        char *points_filename = &points_file[0];
        points_stream = fopen (points_filename,"rb");
        points_num = fread(points_data,sizeof(float),points_num,points_stream)/4;

        for (int32_t i=0; i<points_num; i++) {
            pcl::PointXYZ point;

            point.x = *px;
            point.y = *py;
            point.z = *pz;

            points->push_back(point);
            px+=4; py+=4; pz+=4;
        }
        fclose(points_stream);
    }
    else{
        std::cout << "Reading  the defined input is not implemented" << std::endl;
        return nullptr;
    }

    return points;
}

