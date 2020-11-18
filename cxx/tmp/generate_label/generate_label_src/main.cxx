// std
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
using namespace std;

// boost
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/join.hpp>

// prototypes
void simplify_ply(int argc,  char* argv[]);
bool isin(vector<string> &word_list, const string &word);
vector<string> read_file(string path_to_file);
void generate_label(vector<string> xyzc,
                    vector<string> class_id,
                    vector<string> map,
                    string output_dir,
                    string filename);


int main(int argc,  char* argv[])
{

  vector<string> pointcloud_xyz_classId, pointcloud_classId, mapping;
  string output_dir, filename;
  if (argc == 6)
  {
    pointcloud_xyz_classId = read_file(argv[1]);
    pointcloud_classId = read_file(argv[2]);
    mapping = read_file(argv[3]);
    output_dir = argv[4];
    filename = argv[5];
  }
  else
  {
//    pointcloud_xyz_classId = read_file("/home/cedric/Documents/PycharmProjects/tmp_project/data/raw/raw_lille2.ply");

//    pointcloud_classId = read_file( "/home/cedric/Documents/PycharmProjects/tmp_project/data/raw/class_id_raw_lille2.txt");

//    mapping = read_file( "/home/cedric/Documents/PycharmProjects/tmp_project/xml/map-label-id-to-class-idx.txt");
//    output_dir = "/home/cedric/Documents/PycharmProjects/tmp_project/data";
//    filename = "test";

    std::cout << "Invalid number of arguments, please check the inputs and try again" << std::endl;



//    string path_to_input = "/home/cedric/Documents/PycharmProjects/tmp_project/data/raw";
//    string path_to_output = "/home/cedric/Documents/PycharmProjects/tmp_project/data";

////    string input_name = "raw_lille1.ply";
//    string input_name = "raw_paris.ply";
//    string train = "train_" + input_name;
//    string dev = "dev_" + input_name;
//    string test = "test_" + input_name;

//    vector<string> file_content = read_file(path_to_input + "/" + input_name);

//    float train_percentil = 0.6f;
//    int split_train_idx = static_cast<int>( train_percentil * file_content.size() );
//    int split_dev_idx = static_cast<int>((1- train_percentil)/2 * file_content.size());

//    string head_train = "ply\n"
//    "format ascii 1.0\n"
//    "element vertex "+std::to_string(split_train_idx)+"\n"
//    "property float x\n"
//    "property float y\n"
//    "property float z\n"
//    "property float class_id\n"
//    "end_header\n" ;

//    string head_dev = "ply\n"
//    "format ascii 1.0\n"
//    "element vertex "+std::to_string(split_dev_idx)+"\n"
//    "property float x\n"
//    "property float y\n"
//    "property float z\n"
//    "property float class_id\n"
//    "end_header\n" ;

//    string head_test = "ply\n"
//    "format ascii 1.0\n"
//    "element vertex "+std::to_string(file_content.size()-(split_train_idx+split_dev_idx))+"\n"
//    "property float x\n"
//    "property float y\n"
//    "property float z\n"
//    "property float class_id\n"
//    "end_header\n" ;


//    vector<string> train_vec, dev_vec, test_vec;
//    train_vec.insert(train_vec.end(),
//                     file_content.begin(),
//                     file_content.begin() + split_train_idx);

//    dev_vec.insert(dev_vec.end(),
//                   file_content.begin() + split_train_idx,
//                   file_content.begin() + split_dev_idx + split_train_idx);

//    test_vec.insert(test_vec.end(),
//                     file_content.begin()+split_dev_idx + split_train_idx, file_content.end());

//    std::ofstream outfile;
//    outfile.open(path_to_output + "/" + train);
//    outfile << head_train << boost::algorithm::join(train_vec, "\n");
//    outfile.close();

//    outfile.open(path_to_output + "/" + dev);
//    outfile << head_dev << boost::algorithm::join(dev_vec, "\n");
//    outfile.close();

//    outfile.open(path_to_output + "/" + test);
//    outfile << head_test << boost::algorithm::join(test_vec, "\n");
//    outfile.close();

//    std::cout << file_content.size() << "\n"
//              << split_train_idx << "\n"
//              << split_dev_idx << "\n"
//              << split_train_idx << "\n" ;

    return 0;
  }

//  std::cout << pointcloud_xyz_classId[0] << std::endl;
//  std::cout << pointcloud_xyz_classId[1] << std::endl;
//  std::cout << pointcloud_xyz_classId.size() << std::endl;
//  std::cout << pointcloud_xyz_classId[pointcloud_xyz_classId.size()-1] << std::endl;
//  std::cout << std::endl;


//  std::cout << pointcloud_classId[0] << std::endl;
//  std::cout << pointcloud_classId[1] << std::endl;
//  std::cout << pointcloud_classId.size() << std::endl;
//  std::cout << pointcloud_classId[pointcloud_classId.size()-1] << std::endl;
//  std::cout << std::endl;


//  std::cout << mapping[0] << std::endl;
//  std::cout << mapping[1] << std::endl;
//  std::cout << mapping.size() << std::endl;
//  std::cout << mapping[mapping.size()-1] << std::endl;
//  std::cout << std::endl;
//  return 0;

  generate_label(pointcloud_xyz_classId,
                 pointcloud_classId,
                 mapping,
                 output_dir,
                 filename);



}

bool isin(vector<string> &word_list, const string &word)
{
  vector<string>::iterator it = find (word_list.begin(), word_list.end(), word);

  if (it != word_list.end())
  {
    return true;
  }
  else
  {
    return false;
  }
}

vector<string>read_file(string path_to_file)
{
  vector<string> file_content, tmp_file;
  std::string line;
  std::fstream file(path_to_file);
  while (getline(file, line))
  {
    tmp_file.push_back(line);
  }
  file.close();

  vector<string>::iterator it_end_header = find (tmp_file.begin(), tmp_file.end(), "end_header");

  if (it_end_header != tmp_file.end())
  {
    file_content.insert(file_content.end(), it_end_header+1, tmp_file.end());
  }
  else
  {
    file_content.insert(file_content.end(), tmp_file.begin(), tmp_file.end());
  }
  return file_content;
}


void generate_label(vector<string> xyzc,
                    vector<string> class_id,
                    vector<string> map,
                    string output_dir,
                    string filename)
{
  string label = "";
  string output_ply = "";

  unsigned int warnEvery = static_cast<unsigned int>(0.3 * class_id.size() );

  for (unsigned int num_points = 0; num_points < class_id.size(); ++num_points)
  {
    string current_class_id = class_id[num_points];
    output_ply += xyzc[num_points];

    if (num_points % warnEvery == 0)
    {
      std::cout << "Number of points processed: "
                << num_points << " / " << class_id.size() << std::endl;
    }



    for (unsigned int class_num = 0; class_num < map.size(); ++class_num)
    {

      vector<string> current_class_map, current_classid_list;
      boost::split(current_class_map, map[class_num], boost::is_any_of(" "));
      current_classid_list.insert(current_classid_list.end(),
                                  current_class_map.begin()+4,
                                  current_class_map.end());

      if (class_num == map.size() - 1 && !(isin(current_classid_list, current_class_id)))
      {
        label += "0\n"; // background

        // for visualization
        output_ply += "0 0 0 0\n"; // 0 for background and black color (RGB 0, 0, 0)
      }

      if (isin(current_classid_list, current_class_id))
      {
        label += current_class_map[0] + "\n"; // corresponding class

        // for visualization
        vector<string> tmp_vec;
        tmp_vec.insert(tmp_vec.end(), current_class_map.begin(),
                       current_class_map.begin() + 4);
        output_ply += boost::algorithm::join(tmp_vec, " ") + "\n";

        break;
      }
    }

  }

  string head = "ply\n"
  "format ascii 1.0\n"
  "element vertex "+std::to_string(class_id.size())+"\n"
  "property float x\n"
  "property float y\n"
  "property float z\n"
  "property float class_id\n"
  "property uint gt_label\n"
  "property uint gt_red\n"
  "property uint gt_green\n"
  "property uint gt_blue\n"
  "end_header\n" ;

  // ecriture ply x y z gt(class_id label r g b)
  std::string path_to_ply_output = output_dir + "/gt_" + filename + ".ply";

  // ecriture csv gt_label
  std::string path_to_csv_output = output_dir + "/gt_label_" + filename + ".csv";

  std::ofstream outfile;

  outfile.open(path_to_ply_output);
  outfile << head << output_ply;
  outfile.close();

  outfile.open(path_to_csv_output);
  outfile << label;
  outfile.close();


}
