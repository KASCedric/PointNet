#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <assert.h>
#include <omp.h>
#include <ctime>
#include <cmath>
#include <chrono>



#define NUM_MAX_OF_IO_INSTANCES 1

using namespace std;


// -----------------------------------------------------------------------------

inline std::vector<std::string> split(const std::string& str, const std::string& delim)
{
		vector<string> tokens;
		size_t prev = 0, pos = 0;
		do
		{
				pos = str.find(delim, prev);
				if (pos == string::npos) pos = str.length();
				string token = str.substr(prev, pos-prev);
				if (!token.empty()) tokens.push_back(token);
				prev = pos + delim.length();
		}
		while (pos < str.length() && prev < str.length());
		return tokens;
}

// -----------------------------------------------------------------------------


//##############################################################################
// CLASS Point
//##############################################################################

class Point
{
private:
	int _num_features = 3;
	std::vector<float> _data;
	std::string _sep = ",";

public:
  Point(const int &num_features);
  template<class Type>
  Point(const std::vector<Type> &data);
  Point(const std::vector<float> &data);

	const std::vector<float> & data() const;
	std::vector<float> & data();

	const int & num_features() const;
	int & num_features();

	const std::string & sep () const;
	std::string & sep ();

  friend ostream& operator<<(ostream& os, const Point& p);

  friend bool operator<=(const Point & l, const Point & r);
  friend bool operator>=(const Point & l, const Point & r);
  friend bool operator==(const Point & l, const Point & r);

  friend Point operator+(Point l, const Point & r);
  friend Point operator-(Point l, const Point & r);

  friend Point operator*(Point l, const Point & r);
  template<class Type>
  friend Point operator*(Point l, const std::vector<Type> & r);
  template<class Type>
  friend Point operator*(const std::vector<Type> &l, Point r);
  template<class Type>
  friend Point operator*(const Type &l, Point r);

  float & operator[](std::size_t idx)       { return _data[idx]; }
  const float & operator[](std::size_t idx) const { return _data[idx]; }

  Point slice(int begin, int end);


};

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------


inline Point str_to_point(std::string str, std::string sep=",")
{
	std::vector<std::string> tmp = split(str, sep);
  Point p(static_cast<int>( tmp.size() ));

  omp_set_nested(1);
  //#pragma omp parallel for
  for (unsigned int i = 0; i < static_cast<unsigned int>(tmp.size()); ++i)
	{
		p.data()[i] = static_cast<float>( std::stof(tmp[i]) );
	}
	return p;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

//##############################################################################
// CLASS IO
//##############################################################################

enum OPEN_TYPE
{
  read_,
  write_
};

class IO
{
private:
  static int count;

  std::ifstream _inFile;
  std::ofstream _outFile;

  std::string _file_content;

public:
  IO();

  const std::string & file_content() const;

  bool open(std::string, OPEN_TYPE type);
  bool close();

  bool read(std::string input_name);
  bool write(std::string input_name, std::string data);
  bool write(std::string output_name, std::vector<Point> data);

  bool reset();

  std::vector<Point> read_ply(const std::string &input_name, const std::string &sep=" ");
  bool save_ply(const std::string &output_name, const std::vector<Point> &data);

  std::vector<Point> read_csv(const std::string &input_name, const std::string &sep=" ");
  bool save_csv(const std::string &output_name, const std::vector<Point> &data);
  bool save_csv(const std::string &output_name, const std::vector<std::vector<Point>> &data);

};




//##############################################################################
// CLASS OCCUPANCY_GRID
//##############################################################################


class OccupancyGrig
{
private:
  std::vector<Point> _input_data;
  std::vector<Point> _output_data;
  std::vector<int> _grid_shape;
  std::vector<float> _grid_resolution;

public:
  OccupancyGrig();


  const std::vector<Point> & input_data() const;
  std::vector<Point> & input_data();

  const std::vector<Point> & output_data() const;
  std::vector<Point> & output_data();

  const std::vector<int> & grid_shape() const;
  std::vector<int> & grid_shape();

  const std::vector<float> & grid_resolution() const;
  std::vector<float> & grid_resolution();


  bool is_point_in_voxel(Point voxel_xyz_min,
                         Point voxel_xyz_max,
                         Point p);

  Point is_point_in_grid(std::vector<std::vector<Point>> voxel_grid_boundaries,
                        Point p);

  std::vector<Point> elementary_voxel(Point center,
                                      std::vector<float> grid_resolution,
                                      std::vector<int> grid_shape);


  std::vector<std::vector<Point>> voxel_grid_bounds(Point center,
                                        std::vector<float> grid_resolution,
                                        std::vector<int> grid_shape);


  std::vector<Point> compute_occupancy(std::vector<Point> points,
                                       std::vector<float> grid_resolution,
                                       std::vector<int> grid_shape/*,
                                       string output_name="occupancy_5cm.csv"*/);

  std::vector<Point> get_point_neighborhood(Point center,
                                            std::vector<Point> all_points,
                                            std::vector<float> grid_resolution,
                                            std::vector<int> grid_shape);

  Point one_point_occ_grid(Point center,
                          std::vector<Point> point_neighborhood,
                          std::vector<float> grid_resolution,
                          std::vector<int> grid_shape/*,
                          string output_name="occupancy_5cm.csv"*/);

};



