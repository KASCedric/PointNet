// local
#include "utils/utils.h"

// pcl
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/common.h>
using namespace pcl;


// eigen
#include <Eigen/Eigen>


//using Eigen::MatrixXd;
using namespace Eigen;
using namespace Eigen::internal;
using namespace Eigen::Architecture;

// boost
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/join.hpp>
#include <boost/archive/iterators/base64_from_binary.hpp>
#include <boost/archive/iterators/insert_linebreaks.hpp>
#include <boost/archive/iterators/transform_width.hpp>
#include <boost/archive/iterators/ostream_iterator.hpp>

// others
#include <sstream>
#include <string>
#include "boost/algorithm/string/yes_no_type.hpp"
#include <stdlib.h>
#include <stdio.h>

// bitset (manipulation de bits
#include <bits/stdc++.h>

// prototypes
void tests();
void ply_to_csv(std::string root_dir, std::string ply_file_name);
void read_ply(std::string path_to_file, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

//#define mem 32768 // 32 * 32 * 32
#define mem 3


void voxel_grid_bounds(const pcl::PointXYZ &center,
                       std::vector<Eigen::Vector4f> &bound_min,
                       std::vector<Eigen::Vector4f> &bound_max,
                       const Eigen::Vector3f &grid_resolution,
                       const Eigen::Vector3i &grid_shape);

Point occupancy_grid(const pcl::PointXYZ &current_point,
                     const pcl::PointCloud<pcl::PointXYZ> &point_neighborhood,
                     const Eigen::Vector3f &grid_resolution,
                     const Eigen::Vector3i &grid_shape);


Eigen::ArrayXf occupancy_grid_V2(const pcl::PointXYZ &current_point,
                        const pcl::PointCloud<pcl::PointXYZ> &point_neighborhood,
                        const Eigen::Vector3f &grid_resolution,
                        const Eigen::Vector3i &grid_shape);


int main(int argc,  char* argv[])
{
  string path_to_file, output_dir, filename;
  if (argc == 4)
  {
    path_to_file = argv[1];
    output_dir = argv[2];
    filename = argv[3];
  }
  else
  {
    std::cout << "Invalid number of arguments, please check the inputs and try again" << std::endl;

    return 0;
  }
// **************************************************************************
  // DO NOT EDIT !!

// ---------------------------------------------------------------------------

//  pcl::PointXYZ current_point(0, 0, 0);

//  pcl::PointCloud<pcl::PointXYZ> point_neighborhood;
//  point_neighborhood.push_back( pcl::PointXYZ(-1, 1, 1) );
//  point_neighborhood.push_back( pcl::PointXYZ(2, 2, 2) );

//  Eigen::Vector3f grid_resolution(1, 1, 1);
//  Eigen::Vector3i grid_shape(2, 2, 2);

// ---------------------------------------------------------------------------

//  pcl::PointXYZ current_point(0.5, 1, 0);

//  pcl::PointCloud<pcl::PointXYZ> point_neighborhood;
//  point_neighborhood.push_back( pcl::PointXYZ(10, 10, 10) );
//  point_neighborhood.push_back( pcl::PointXYZ(0.45f, 0.95f, -0.05f) );
//  point_neighborhood.push_back( pcl::PointXYZ(0.45f, 0.95f, 0.05f) );
//  point_neighborhood.push_back( pcl::PointXYZ(0.55f, 1.05f, -0.05f) );
//  point_neighborhood.push_back( pcl::PointXYZ(0.55f, 1.05f, 0.05f) );

//  Eigen::Vector3f grid_resolution(0.1f, 0.1f, 0.1f);
//  Eigen::Vector3i grid_shape(2, 2, 2);

//// ---------------------------------------------------------------------------


//  Point occupancy = occupancy_grid_V2(current_point,
//                                      point_neighborhood,
//                                      grid_resolution,
//                                      grid_shape);

//  std::cout << occupancy << std::endl;

//-----------------------------------------------------------------------------

//  int num_examples = 10;
//  int num_features = 5;
//  int ckpt_freq = 2;

//  Eigen::MatrixXf test(0, num_features);

//  for (int ex = 0; ex < num_examples; ++ex)
//  {

//    Eigen::ArrayXf VoxelGrid = Eigen::ArrayXf::Zero(num_features);
//    test.conservativeResize(test.rows()+1, test.cols());
//    test.row(test.rows()-1) = VoxelGrid;

//    if (ex % ckpt_freq == 0)
//    {
//      std::cout << "Checkpointing ...\n";

//      fstream f("filename.ext", f.out | f.app);
//      f << test << "\n" ;
//      f.close();
//      test.conservativeResize(0, test.cols());

//      std::cout << "End of checkpoint !\n";
//    }

//    if (ex == num_examples - 1 && (ex / ckpt_freq) * ckpt_freq != num_examples-1)
//    {
//      std::cout << "Checkpointing ...\n";

//      fstream f("filename.ext", f.out | f.app);
//      f << test << "\n" ;
//      f.close();
//      test.conservativeResize(0, test.cols());

//      std::cout << "End of checkpoint !\n";
//    }
//  }

// **************************************************************************



//------------------------------------------------------------------------------


  // nuage de points (x, y, z, label) stocke dans la variable cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
//  std::string path_to_file = "/home/cedric/Documents/PycharmProjects/"\
//                             "msDeepVoxScene/data/Lille2/76_lille2.ply";
  read_ply(path_to_file, cloud);


  // frequence d'affichage des infos concernant le calcul des features
  double percentil = 0.001;
  size_t warnEvery = static_cast<size_t>( cloud->size() * percentil );

  // evaluation du temps de calcul des features
  std::chrono::time_point<std::chrono::system_clock> tic, toc;


  int gshape = 32; // nombre de voxels dans la voxelGrid
  double res = 0.1; // (en metres) taille d'un voxel soit 10 cm

//  double radius = 0.25; // 25 cm => occupancy: 3600s pour 353117 points
//  double radius = 0.125; // 12.5 cm => occupancy: 2800 pour 353117 points
//  double radius = 0.0625; // 6.25 cm => occupancy: 1800 pour 353117 points

  double radius = sqrt(3) * gshape * res; // rayon de recherche de voisins

  // diimensions et resolution du voxelGrid
  Eigen::Vector3i grid_shape(gshape, gshape, gshape);
  Eigen::Vector3f grid_resolution(static_cast<float>(radius)/grid_shape[0],
                                  static_cast<float>(radius)/grid_shape[1],
                                  static_cast<float>(radius)/grid_shape[2]);


  // voisinage sur un rayon <radius> de chacun des points du nuage
//  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> all_points_neighborhood; // tmp

  // ooccupancy sur un rayon <radius> de chacun des points du nuage
//  std::vector<Point> all_points_occupancy; // tmp

  // create a kdtree
  pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtreeCloud(new pcl::KdTreeFLANN<pcl::PointXYZ>());
  kdtreeCloud->setInputCloud(cloud);

  // variables pour le temps de calcul de l'occupancy de chacun des points
  std::chrono::time_point<std::chrono::system_clock> tic_tmp, toc_tmp;

  // on declenche les chronos
  tic = std::chrono::system_clock::now();
  tic_tmp = std::chrono::system_clock::now();

  // matrice temporaire qui contiendra les occupancy
  // (var reset a chaque checkpoint pour mieux gerer la memoire)
  Eigen::MatrixXf occ_tmp(0, grid_shape.prod());

  // retrieve neighborhood
  std::string output_name = output_dir +
      "/occupancy_res_010_shape_32_" + filename + ".csv";
  for (size_t pointIndex = 0; pointIndex < cloud->size(); ++pointIndex)
  {

    // Get current point
    pcl::PointXYZ p = cloud->points[pointIndex];

    // get nearest neighbor indexes of the query point
    std::vector<int> nearestIndex;
    std::vector<float> nearestDist;

    kdtreeCloud->radiusSearch(p, radius, nearestIndex, nearestDist);

    // get the neighborhood of the query point
    pcl::PointCloud<pcl::PointXYZ>::Ptr neighborhood (new pcl::PointCloud<pcl::PointXYZ>);

//    std::vector<Point> point_neighborhood; // tmp

    for (unsigned int neighIndex = 0; neighIndex < nearestIndex.size(); ++neighIndex)
    {
      pcl::PointXYZ p_tmp = cloud->points[static_cast<size_t>( nearestIndex[neighIndex] )];
      neighborhood->push_back(p_tmp);

//      point_neighborhood.push_back(Point({p_tmp.x, p_tmp.y, p_tmp.z})); // tmp
    }

    // computation of the occupancy
//    all_points_occupancy.push_back
//    (
//          occupancy_grid_V2(p, *neighborhood, grid_resolution, grid_shape)
//    );

    Eigen::ArrayXf VoxelGrid = occupancy_grid_V2(p,
                                                 *neighborhood,
                                                 grid_resolution,
                                                 grid_shape);

//    all_points_neighborhood.push_back(neighborhood); // tmp

    occ_tmp.conservativeResize(occ_tmp.rows()+1, occ_tmp.cols());
    occ_tmp.row(occ_tmp.rows()-1) = VoxelGrid;

    if (pointIndex % warnEvery == 0)
    {
      toc_tmp = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed_seconds_tmp = toc_tmp - tic_tmp;

      std::cout << "Number of points processed: " << pointIndex
                << " / " << cloud->size()
                << " | Elapsed time: "
                << elapsed_seconds_tmp.count() << " s."
                << std::endl;

      std::cout << "Checkpointing ...\n";

      fstream f(output_name, f.out | f.app);
      f << occ_tmp << "\n" ;
      f.close();
      occ_tmp.conservativeResize(0, occ_tmp.cols());

      std::cout << "End of checkpoint !\n";

      tic_tmp = std::chrono::system_clock::now();
    }

    if (pointIndex == cloud->size() - 1
        && (pointIndex / warnEvery) * warnEvery != cloud->size()-1)
    {
      std::cout << "Checkpointing ...\n";

      fstream f(output_name, f.out | f.app);
      f << occ_tmp << "\n" ;
      f.close();
      occ_tmp.conservativeResize(0, occ_tmp.cols());

      std::cout << "End of checkpoint !\n";
    }

  }
  toc = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = toc - tic;

  std::cout << "Elapsed time: " << elapsed_seconds.count() << " s." << std::endl;


  // Number of neighbors and occupancy of the point point_index
//  unsigned long point_index = 5000;
//  std::cout << "Number of neighbors of the point "
//            << point_index << " : "
//            <<  all_points_neighborhood[point_index]->points.size()
//            << std::endl;

//  std::cout << "Occupancy grid of point "
//            << point_index << " neighborhood : "
//            << all_points_occupancy[point_index]
//            << std::endl;

//  IO io;
////  io.save_csv("radius1_shape2_occ_grid.csv", all_points_occupancy);
////  io.save_csv("radius"+to_string(static_cast<int>(radius))
////              +"_shape"+to_string(grid_shape[0])+"_occ_grid.csv", all_points_occupancy);

//  io.save_csv("radius_3-2_shape"+to_string(grid_shape[0])+"_occ_grid.csv", all_points_occupancy);



}


void read_ply(std::string path_to_file, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
  pcl::PLYReader Reader;

  if (Reader.read(path_to_file, *cloud) == 0)
  {
    std::cout << "PointCloud: " << path_to_file << " successfully loaded !"
              << std::endl
              << std::endl;
  }
  else
  {
    std::cout << "Impossible to laod pointcloud: " << path_to_file
              << std::endl
              << std::endl;
  }
}


void voxel_grid_bounds(const pcl::PointXYZ &center,
                       std::vector<Eigen::Vector4f> &bound_min,
                       std::vector<Eigen::Vector4f> &bound_max,
                       const Eigen::Vector3f &grid_resolution,
                       const Eigen::Vector3i &grid_shape)
{

  Eigen::Vector3f c(center.x, center.y, center.z);

  Eigen::ArrayXf x_min, x_max, y_min, y_max, z_min, z_max;

  x_min = grid_resolution[0] * Eigen::ArrayXf::LinSpaced(grid_shape[0], -grid_shape[0]/2, grid_shape[0]/2 - 1) + center.x;

  y_min = grid_resolution[1] * Eigen::ArrayXf::LinSpaced(grid_shape[1], -grid_shape[1]/2, grid_shape[1]/2 - 1) + center.y;

  z_min = grid_resolution[2] * Eigen::ArrayXf::LinSpaced(grid_shape[2], -grid_shape[2]/2, grid_shape[2]/2 - 1) + center.z;



  for (int i = 0; i < grid_shape[0]; i++)
  {
    for (int j = 0; j < grid_shape[1]; j++)
    {
      for (int k = 0; k < grid_shape[2]; k++)
      {
        Eigen::Vector4f tmp_min(x_min[i], y_min[j], z_min[k], 0);
        Eigen::Vector4f tmp_max(x_min[i]+grid_resolution[0],
                               y_min[j]+grid_resolution[1],
                               z_min[k]+grid_resolution[2], 0);
        bound_min.push_back(tmp_min);
        bound_max.push_back(tmp_max);
      }
    }
  }
}



Point occupancy_grid(const pcl::PointXYZ &current_point,
                     const pcl::PointCloud<pcl::PointXYZ> &point_neighborhood,
                     const Eigen::Vector3f &grid_resolution,
                     const Eigen::Vector3i &grid_shape)
{
  Point occupancy({});

  std::vector<Eigen::Vector4f> bound_min;
  std::vector<Eigen::Vector4f> bound_max;
  voxel_grid_bounds(current_point,
                    bound_min,
                    bound_max,
                    grid_resolution,
                    grid_shape);

  for (unsigned int grid_id = 0; grid_id < bound_min.size(); ++grid_id)
  {
    std::vector< int > indices_tmp;
    pcl::getPointsInBox(point_neighborhood, bound_min[grid_id], bound_max[grid_id], indices_tmp );

    if(indices_tmp.size() != 0)
    {
      occupancy.data().push_back(1);
    }
    else
    {
      occupancy.data().push_back(0);
    }
  }

  return occupancy;
}



Eigen::ArrayXf occupancy_grid_V2(const pcl::PointXYZ &current_point,
                              const pcl::PointCloud<pcl::PointXYZ> &point_neighborhood,
                                 const Eigen::Vector3f &grid_resolution,
                                 const Eigen::Vector3i &grid_shape)
{
  // idea proposed by pierre

  // initialization with zeros(grid_shape)
  Eigen::ArrayXf VoxelGrid = Eigen::ArrayXf::Zero(grid_shape.prod());

  // orientation of voxel grid (= Eye(3,3) as long as the cloud is gravity aligned)
  Eigen::MatrixXf R = Eigen::MatrixXf::Identity(3, 3);

  // scale matrix ( diagonale )
  Eigen::MatrixXf D = Eigen::MatrixXf::Identity(3, 3);
  D(0, 0) /= grid_resolution[0];
  D(1, 1) /= grid_resolution[1];
  D(2, 2) /= grid_resolution[2];


  for (size_t neighIndex = 0; neighIndex < point_neighborhood.size(); ++neighIndex)
  {

    // Get current nneighbor
    pcl::PointXYZ current_neigh = point_neighborhood.points[neighIndex];

    // current neighbor in the voxel referentiel
    Eigen::VectorXf current_neigh_new_ref;

    // translation ( y = x - x0 )
    current_neigh_new_ref = current_neigh.getArray3fMap() -
        current_point.getArray3fMap();

    // rotation ( y = R_T ( x - x0 ) )
    current_neigh_new_ref = R.transpose() * current_neigh_new_ref;

    // scale ( y = D * R_T * (x - x0) ): y en coordonnees voxeliques
    current_neigh_new_ref = D * current_neigh_new_ref;

    for (unsigned int itmp = 0; itmp < current_neigh_new_ref.size(); ++itmp)
    {
      if (current_neigh_new_ref[itmp] < 0)
      {
        current_neigh_new_ref[itmp] += 0.000001f;
      }
      if (current_neigh_new_ref[itmp] >= 0)
      {
        current_neigh_new_ref[itmp] -= 0.000001f;
      }
    }


//    std::cout << current_neigh_new_ref.x() << " " <<
//                 current_neigh_new_ref.y() << " " <<
//                 current_neigh_new_ref.z() << " " <<
//                 std::endl;

    // conversion en coordonnees entieres
    current_neigh_new_ref = current_neigh_new_ref.array().floor();

    // on verif que le neigh est a l'interieur de la voxel grid
    if (current_neigh_new_ref.x() < grid_shape[0]/2 &&
        current_neigh_new_ref.x() >= -grid_shape[0]/2 &&
        current_neigh_new_ref.y() < grid_shape[1]/2 &&
        current_neigh_new_ref.y() >= -grid_shape[1]/2 &&
        current_neigh_new_ref.z() < grid_shape[2]/2 &&
        current_neigh_new_ref.z() >= -grid_shape[2]/2 )
    {
//      std::cout << neighIndex << std::endl;
//      std::cout << current_neigh_new_ref.x() << " " <<
//                   current_neigh_new_ref.y() << " " <<
//                   current_neigh_new_ref.z() << " " <<
//                   std::endl;


      // set voxelGrid[current_neigh_new_ref] = 1
      int i, j, k;
      i = static_cast<int>(current_neigh_new_ref.x() + grid_shape[0]/2);
      j = static_cast<int>(current_neigh_new_ref.y() + grid_shape[1]/2);
      k = static_cast<int>(current_neigh_new_ref.z() + grid_shape[2]/2);

//      std::cout << i << " " <<
//                   j << " " <<
//                   k << " " <<
//                   std::endl;


      VoxelGrid[k + j * grid_shape[2] + i * grid_shape[1] * grid_shape[2]] = 1;


    }

  }

//  std::cout << VoxelGrid.transpose() << std::endl;

//  for (int i = 0; i < grid_shape[0]; ++i)
//  {
//    for (int j = 0; j < grid_shape[1]; ++j)
//    {
//      for (int k = 0; k < grid_shape[2]; ++k)
//      {
//        std::cout << "(" << i << ", "
//                  << j << ", "
//                  << k << ") -> "
//                  << k + j * grid_shape[2] + i * grid_shape[1] * grid_shape[2]
//                  << std::endl;
//      }
//    }
//  }

//  std::vector<float> tmp(VoxelGrid.data(),
//                         VoxelGrid.data() + VoxelGrid.rows() * VoxelGrid.cols());
//  Point occupancy(tmp);

  return VoxelGrid;
}


/*
void occ_grid(const pcl::PointCloud<pcl::PointXYZ> &cloud,
              std::vector<Eigen::Vector4f> &bound_min,
              std::vector<Eigen::Vector4f> &bound_max,
              Point &occupancy)
{

  assert ( bound_min.size() == bound_max.size() );


  std::cout << "------ debug" << std::endl;

  for (unsigned int grid_id = 0; grid_id < bound_min.size(); ++grid_id)
  {
    std::vector< int > indices_tmp;
    pcl::getPointsInBox(cloud, bound_min[grid_id], bound_max[grid_id], indices_tmp );

    std::cout << indices_tmp.size() << std::endl;

    if(indices_tmp.size() != 0)
    {
      occupancy.data().push_back(1);
    }
    else
    {
      occupancy.data().push_back(0);
    }
  }
}
*/

void ply_to_csv(std::string root_dir, std::string ply_file_name)
{

  if (root_dir.back() != '/')
  {
    root_dir += "/";
  }

  std::string path_to_ply = root_dir + ply_file_name + ".ply";

  std::string output_file_name = ply_file_name;
  std::string path_to_csv_output = root_dir + output_file_name + ".csv";

  IO io;
  std::vector<Point> ply = io.read_ply(path_to_ply);

  std::cout << ply.size() << std::endl;

  io.save_csv(path_to_csv_output, ply);

}



void tests()
{

  IO io;
  OccupancyGrig og;

  // 1.
  io.read("texte");
  std::cout << io.file_content() << std::endl;
  io.write("texte2", "okok\nya man\n1 2 3");

  // 2.
  std::vector<float> tmp = {0,1,2,3,4,5,6,7,8,9};
  Point pt(tmp);
  std::cout << pt << std::endl;
  std::vector<Point> data = {pt};
  io.write("texte3.csv", data);

  // 3.
  std::vector<Point> ply = io.read_ply("texte.ply", " ");
  io.save_csv("texte.csv", ply);

  // 4.
  Point p1( {-1, -1, -1} );
  Point p2( {1, 1, 1} );
  Point p( {0, 0, 0});
  std::cout << static_cast<int>( og.is_point_in_voxel(p1, p2, p) ) << std::endl;

  // 5.
  Point center( {0, 0, 0});
  std::vector<float> grid_resolution({10, 10, 10});
  std::vector<int> grid_shape( {2, 2, 2});
  std::vector<Point> el_vox = og.elementary_voxel(center, grid_resolution, grid_shape);
  std::cout << el_vox[1] << std::endl;

  // 6.
  std::vector<std::vector<Point>> vox_grid = og.voxel_grid_bounds(center, grid_resolution, grid_shape);
  for (unsigned int k = 0; k < vox_grid.size(); k++)
  {
  std::cout << vox_grid[k][0] << std::endl;
  std::cout << vox_grid[k][1] << std::endl << std::endl;
  }
  io.save_csv("test_.csv", vox_grid);


  // 7.
  std::vector<Point> points = { Point({0, 0, 0}),
  //                                Point({0, 0, 0}),
                                Point({1, 1, 1.0f}) };

  std::vector<float> grid_resolution_({1, 1, 1});
  std::vector<int> grid_shape_( {2, 2, 2});

  std::vector<Point> occ_grid =  og.compute_occupancy(points,
                                          grid_resolution_,
                                          grid_shape_);

  for (unsigned int i = 0; i < occ_grid.size(); i++)
  {
  std::cout << occ_grid[i] << std::endl;
  }
  io.save_csv("test_.csv", occ_grid);



  // 8.
  std::vector<Point> pointcloud = {Point({0, 0, 0}), Point({0, 0, 0})};

  unsigned int self_id;
  unsigned int others_id;
  #ifdef _OPENMP
  #pragma omp parallel for collapse(2)
  #else
  std::cout << "could not find openMP" << std:: endl;
  #endif
  for ( self_id = 0; self_id < static_cast<unsigned int>( pointcloud.size() ); self_id++)
  {

    for (others_id = 0; others_id < static_cast<unsigned int>( pointcloud.size() ); others_id++)
    {
      if(pointcloud[self_id].data().size() != pointcloud[others_id].data().size())
      {
        std::cout << pointcloud[self_id] << std::endl << pointcloud[others_id] << std::endl;
      }
    }
  }

}






