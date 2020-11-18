#include "utils.h"
#include <omp.h>

//##############################################################################
// CLASS Point
//##############################################################################

Point::Point(const int &num_features)
{
	_num_features = num_features;
  _data.assign( static_cast<unsigned long>(_num_features ), 0);
}

// -----------------------------------------------------------------------------

template<class Type>
Point::Point(const std::vector<Type> &data)
{
  std::copy(data.begin(), data.end(), back_inserter(_data));
  _num_features = static_cast<int>( _data.size() );
}

// -----------------------------------------------------------------------------

Point::Point(const std::vector<float> &data)
{
  std::copy(data.begin(), data.end(), back_inserter(_data));
  _num_features = static_cast<int>( _data.size() );
}

// -----------------------------------------------------------------------------


const std::vector<float> & Point::data() const
{
	return _data;
}

// -----------------------------------------------------------------------------

std::vector<float> & Point::data()
{
	return _data;
}

// -----------------------------------------------------------------------------

const int & Point::num_features() const
{
	return _num_features;
}

// -----------------------------------------------------------------------------

int & Point::num_features()
{
	return _num_features;
}

// -----------------------------------------------------------------------------

const std::string & Point::sep () const
{
	return _sep;
}

// -----------------------------------------------------------------------------

std::string & Point::sep ()
{
	return _sep;
}

// -----------------------------------------------------------------------------

ostream& operator<<(ostream& os, const Point& p)
{
  unsigned int i;
  omp_set_nested(1);
//#pragma omp parallel for ordered schedule(dynamic)
  for (i = 0; i < static_cast<unsigned int>( p.data().size() - 1 ); ++i)
  {
//#pragma omp ordered
    os << p.data()[i] << p._sep;

  }
  os << p.data()[i] << endl;
  return os;
}


// -----------------------------------------------------------------------------

bool operator<=(const Point & l, const Point & r)
{
  assert(l.data().size() == r.data().size());
  bool ans = true;

  omp_set_nested(1);
//#pragma omp parallel for ordered schedule(dynamic)
  for  (unsigned int i = 0; i < static_cast<unsigned int>(l.data().size()); i++)
  {
//#pragma omp ordered
    ans &= l.data()[i] <= r.data()[i];
  }
  return ans;
}

// -----------------------------------------------------------------------------

bool operator>=(const Point & l, const Point & r)
{
  assert(l.data().size() == r.data().size());
  bool ans = true;

  omp_set_nested(1);
//#pragma omp parallel for ordered schedule(dynamic)
  for  (unsigned int i = 0; i < static_cast<unsigned int>(l.data().size()); i++)
  {
//#pragma omp ordered
    ans &= l.data()[i] >= r.data()[i];
  }
  return ans;
}

// -----------------------------------------------------------------------------

bool operator==(const Point & l, const Point & r)
{
  assert(l.data().size() == r.data().size());
  bool ans = true;

  float epsilon = 0.0001f;

  omp_set_nested(1);
//#pragma omp parallel for ordered schedule(dynamic)
  for  (unsigned int i = 0; i < static_cast<unsigned int>(l.data().size()); i++)
  {
//#pragma omp ordered
    ans &= abs(l.data()[i] - r.data()[i]) <= epsilon;
  }
  return ans;
}



// -----------------------------------------------------------------------------

Point operator+(Point l, const Point & r)
{
  assert(l.data().size() == r.data().size());
  Point p(static_cast<int>( l.data().size() ));

  omp_set_nested(1);
//#pragma omp parallel for ordered schedule(dynamic)
  for  (unsigned int i = 0; i < static_cast<unsigned int>(l.data().size()); i++)
  {
//#pragma omp ordered
    p.data()[i] = l.data()[i] + r.data()[i];
  }
  return p;
}


// -----------------------------------------------------------------------------

Point operator-(Point l, const Point & r)
{
  if(l.data().size() != r.data().size())
  {
    std::cout << l.data().size() << std::endl << r.data().size() << std::endl;
  }
  assert(l.data().size() == r.data().size());
  Point p(static_cast<int>( l.data().size() ));

  omp_set_nested(1);
//#pragma omp parallel for ordered schedule(dynamic)
  for  (unsigned int i = 0; i < static_cast<unsigned int>(l.data().size()); i++)
  {
//#pragma omp ordered
    p.data()[i] = l.data()[i] - r.data()[i];
  }
  return p;
}

// -----------------------------------------------------------------------------


Point operator*(Point l, const Point & r)
{
  assert(l.data().size() == r.data().size());
  Point p(static_cast<int>( l.data().size() ));

  omp_set_nested(1);
//#pragma omp parallel for ordered schedule(dynamic)
  for  (unsigned int i = 0; i < static_cast<unsigned int>(l.data().size()); i++)
  {
//#pragma omp ordered
    p.data()[i] = l.data()[i] * r.data()[i];
  }
  return p;
}

// -----------------------------------------------------------------------------

template<class Type>
Point operator*(Point l, const std::vector<Type> & r)
{
  assert(l.data().size() == r.size());
  Point p(static_cast<int>( l.data().size() ));

  omp_set_nested(1);
//#pragma omp parallel for ordered schedule(dynamic)
  for  (unsigned int i = 0; i < static_cast<unsigned int>(l.data().size()); i++)
  {
//#pragma omp ordered
    p.data()[i] = l.data()[i] * r[i];
  }
  return p;
}

// -----------------------------------------------------------------------------

template<class Type>
Point operator*(const std::vector<Type> &l, Point r)
{
  assert(r.data().size() == l.size());
  Point p(static_cast<int>( r.data().size() ));

  omp_set_nested(1);
//#pragma omp parallel for ordered schedule(dynamic)
  for  (unsigned int i = 0; i < r.data().size(); i++)
  {
//#pragma omp ordered
    p.data()[i] = r.data()[i] * l[i];
  }
  return p;
}
// -----------------------------------------------------------------------------

template<class Type>
Point operator*(const Type &l, Point r)
{
  Point p(static_cast<int>( r.data().size() ));

  omp_set_nested(1);
//#pragma omp parallel for ordered schedule(dynamic)
  for  (unsigned int i = 0; i < r.data().size(); i++)
  {
//#pragma omp ordered
    p.data()[i] = r.data()[i] * l;
  }
  return p;
}


// -----------------------------------------------------------------------------

Point Point::slice(int begin, int end)
{
  std::vector<float> sliced(_data.begin() + begin, _data.begin() + end);
  return Point(sliced);
}


// -----------------------------------------------------------------------------



// -----------------------------------------------------------------------------




// -----------------------------------------------------------------------------




// -----------------------------------------------------------------------------

//##############################################################################
// CLASS IO
//##############################################################################


int IO::count = 0;
// -----------------------------------------------------------------------------

IO::IO()
{}
// -----------------------------------------------------------------------------


const std::string & IO::file_content() const
{

  return _file_content;
}

// -----------------------------------------------------------------------------


bool IO::open(std::string input_name, OPEN_TYPE type)
{
  assert(type == OPEN_TYPE::read_ || type == OPEN_TYPE::write_);
  if(count == NUM_MAX_OF_IO_INSTANCES)
  {
    std::cout << "Another file is opened.\nClose it and try again" << std::endl;
    throw "An other file is opened.\nClose it and try again";
  }
  ++count;

  switch (type)
  {
    case OPEN_TYPE::read_:
      _inFile.open(input_name);
      break;
    case OPEN_TYPE::write_:
      _outFile.open(input_name);
      break;
  }

  return true;
}
// -----------------------------------------------------------------------------


bool IO::close()
{
  if (_inFile.is_open())
  {
    _inFile.close();
  }
  if (_outFile.is_open())
  {
    _outFile.close();
  }

  if (!(_inFile.is_open() || _outFile.is_open()))
  {
    --count;
    return true;
  }
  return false;
}
// -----------------------------------------------------------------------------


bool IO::read(std::string input_name)
{
  this->_file_content = "";

  this->open(input_name, OPEN_TYPE::read_);

  std::string line;

  while (getline(this->_inFile, line))
  {
    this->_file_content += line + "\n";
  }

  this->close();

  return true;
}
// -----------------------------------------------------------------------------


bool IO::write(std::string output_name, std::string data)
{
	this->reset();

  this->open(output_name, OPEN_TYPE::write_);

	this->_outFile << data;

	this->close();

	return true;
}

// -----------------------------------------------------------------------------


bool IO::write(std::string output_name, std::vector<Point> data)
{
	this->reset();

  this->open(output_name, OPEN_TYPE::write_);

//#pragma omp parallel for ordered schedule(dynamic)
  for (unsigned int i = 0; i < static_cast<unsigned int>(data.size()); ++i)
	{
//#pragma omp ordered
		this->_outFile << data[i];
	}

	this->close();

	return true;
}



// -----------------------------------------------------------------------------


bool IO::reset()
{
  this->_file_content = "";
  this->close();
  return true;
}
// -----------------------------------------------------------------------------


std::vector<Point> IO::read_ply(const std::string &input_name, const std::string &sep)
{

	std::string extension = split(input_name, ".")[1];

	if (extension.compare("ply") != 0)
	{
		std::cout << "File: " << input_name << " not found !" << std::endl;
		throw("file not found");
	}

	this->read(input_name);

	this->_file_content = split(this->_file_content, "end_header\n")[1];

	std::vector<std::string> split_data = split(this->_file_content, "\n");

	std::vector<Point> data;

//#pragma omp parallel for ordered schedule(dynamic)
  for (unsigned long  i= 0; i < split_data.size(); i++)
	{
//#pragma omp ordered
    data.push_back( str_to_point(split_data[i], sep) );
	}

  return data;
}


// -----------------------------------------------------------------------------

bool IO::save_ply(const std::string &output_name, const std::vector<Point> &data)
{
	//TODO: Implement save_ply
	output_name.size();
	data.size();
	std::cout << "not implemented yet" << std::endl;
	throw("not implemented yet");
}


// -----------------------------------------------------------------------------


std::vector<Point> IO::read_csv(const std::string &input_name, const std::string &sep)
{

	std::string extension = split(input_name, ".")[1];

	if (extension.compare("csv") != 0)
	{
		std::cout << "File: " << input_name << " not found !" << std::endl;
		throw("file not found");
	}

	this->read(input_name);

	std::vector<std::string> split_data = split(this->_file_content, "\n");

	std::vector<Point> data;

//#pragma omp parallel for ordered schedule(dynamic)
  for (unsigned int i = 0; i < static_cast<unsigned int>(split_data.size()); i++)
	{
//#pragma omp ordered
    data.push_back( str_to_point(split_data[i], sep) );
	}

	return data;
}


// -----------------------------------------------------------------------------

bool IO::save_csv(const std::string &output_name, const std::vector<Point> &data)
{
	std::string extension = split(output_name, ".")[1];

	if (extension.compare("csv") != 0)
	{
		std::cout << "Impossible to write data in : " << output_name << std::endl;
		throw("error");
	}

	this->write(output_name, data);

	return true;
}


// -----------------------------------------------------------------------------


bool IO::save_csv(const std::string &output_name, const std::vector<std::vector<Point>> &data)
{
  std::string extension = split(output_name, ".")[1];

  if (extension.compare("csv") != 0)
  {
    std::cout << "Impossible to write data in : " << output_name << std::endl;
    throw("error");
  }

  std::vector<Point> _data;

//#pragma omp parallel for ordered schedule(dynamic)
  for (unsigned int k = 0; k < static_cast<unsigned int>(data.size()); k++)
  {
//#pragma omp ordered
    _data.push_back(data[k][0]);
    _data.push_back(data[k][1]);
  }

  this->write(output_name, _data);

  return true;
}


// -----------------------------------------------------------------------------


//##############################################################################
// CLASS OCCUPANCY_GRID
//##############################################################################

OccupancyGrig::OccupancyGrig()
  : _grid_shape( static_cast<unsigned long>(3 ), 32),
    _grid_resolution( static_cast<unsigned long>(3 ), 15)
{}

const std::vector<Point> & OccupancyGrig::input_data() const
{
  return _input_data;
}

std::vector<Point> & OccupancyGrig::input_data()
{
  return _input_data;
}


const std::vector<Point> & OccupancyGrig::output_data() const
{
  return _output_data;
}

std::vector<Point> & OccupancyGrig::output_data()
{
  return _output_data;
}

const std::vector<int> & OccupancyGrig::grid_shape() const
{
  return _grid_shape;
}

std::vector<int> & OccupancyGrig::grid_shape()
{
  assert(_grid_shape[0] % 2 == 0
      && _grid_shape[1] % 2 == 0
      && _grid_shape[2] % 2 == 0 );
  return _grid_shape;
}

const std::vector<float> & OccupancyGrig::grid_resolution() const
{
  return _grid_resolution;
}

std::vector<float> & OccupancyGrig::grid_resolution()
{
  return _grid_resolution;
}


bool OccupancyGrig::is_point_in_voxel(Point voxel_xyz_min,
                                      Point voxel_xyz_max,
                                      Point p)
{
  return p <= voxel_xyz_max && p >= voxel_xyz_min;
}


Point OccupancyGrig::is_point_in_grid(std::vector<std::vector<Point>> voxel_grid_boundaries,
                      Point p)
{
  std::vector<bool> og;

//#pragma omp parallel for ordered schedule(dynamic)
  for (unsigned int i = 0; i < static_cast<unsigned int>(voxel_grid_boundaries.size()); i++)
  {
//#pragma omp ordered
    og.push_back(static_cast<int>( is_point_in_voxel(voxel_grid_boundaries[i][0],
                                   voxel_grid_boundaries[i][1],
                                   p)) );
  }

  return Point(og);
}


std::vector<Point> OccupancyGrig::elementary_voxel(Point center,
                                    std::vector<float> grid_resolution,
                                    std::vector<int> grid_shape)
{

  assert(grid_resolution.size() == grid_shape.size());
  std::vector<float> tmp(grid_resolution.size(), 0.5);
  Point bound( Point(grid_resolution) * tmp * grid_shape);

  Point el_vox_xyz_min( center - bound);
  Point el_vox_xyz_max( center - bound + Point(grid_resolution) );

  std::vector<Point> el_vox;
  el_vox.push_back(el_vox_xyz_min);
  el_vox.push_back(el_vox_xyz_max);

  return el_vox;
}



std::vector<std::vector<Point>> OccupancyGrig::voxel_grid_bounds(Point center,
                                      std::vector<float> grid_resolution,
                                      std::vector<int> grid_shape)
{
  std::vector<std::vector<Point>> vox_grid;
  std::vector<Point> el_vox = elementary_voxel(center,
                                               grid_resolution,
                                               grid_shape);

  //#pragma omp parallel for ordered schedule(dynamic) collapse(3)
  for (int i = 0; i < grid_shape[0]; i++)
  {
    for (int j = 0; j < grid_shape[1]; j++)
    {
      for (int k = 0; k < grid_shape[2]; k++)
      {
        std::vector<Point> temp_vox(el_vox);
        temp_vox[0].data()[0] += i * grid_resolution[0];
        temp_vox[1].data()[0] += i * grid_resolution[0];
        temp_vox[0].data()[1] += j * grid_resolution[1];
        temp_vox[1].data()[1] += j * grid_resolution[1];
        temp_vox[0].data()[2] += k * grid_resolution[2];
        temp_vox[1].data()[2] += k * grid_resolution[2];
//#pragma omp ordered
        vox_grid.push_back(temp_vox);
      }
    }
  }

  return vox_grid;
}




std::vector<Point> OccupancyGrig::compute_occupancy(std::vector<Point> points,
                                     std::vector<float> grid_resolution,
                                     std::vector<int> grid_shape)
{
  std::vector<Point> nn_input;
  time_t tstart = 0;
  time_t tend = 600;

  unsigned int freq = 1;

  unsigned int num_points = static_cast<unsigned int>( points.size() );

//#pragma omp parallel for ordered schedule(dynamic)
  for (unsigned int self_id = 0; self_id < num_points; self_id++)
  {
    if (self_id % freq == 0)
    {
//#pragma omp ordered
      std::cout << "Number of points processed: " << self_id << " / "
                << num_points - 1 <<  "  tps:  "
                << static_cast<float>(tend - tstart) /  static_cast<float>(freq)
                << " seconds" << std::endl;

      IO io;
      // TODO: append the file pour ne pas perdre d info en cas de pb

    }

    tstart = time(nullptr); // tic

    std::vector<std::vector<Point>> voxel_grid_boundaries;
//#pragma omp ordered
    voxel_grid_boundaries = voxel_grid_bounds(points[self_id],
                                              grid_resolution,
                                              grid_shape);

//#pragma omp parallel for ordered schedule(dynamic)
    for (unsigned int others_id = 0; others_id < num_points; others_id++)
    {
      if (self_id != others_id)
      {
        Point point_coors = points[others_id];
//#pragma omp ordered
        nn_input.push_back(is_point_in_grid(voxel_grid_boundaries, point_coors));
      }
    }

    tend = time(nullptr); // toc

  }

  return nn_input;
}



std::vector<Point> OccupancyGrig::get_point_neighborhood(Point center,
                                          std::vector<Point> all_points,
                                          std::vector<float> grid_resolution,
                                          std::vector<int> grid_shape)
{
  std::vector<Point> neighborhood;


  Point bbox_min(center - 0.5f * Point(grid_resolution) * grid_shape );
  Point bbox_max(center + 0.5f * Point(grid_resolution) * grid_shape);

  for (unsigned int i = 0; i < all_points.size(); i++)
  {
    if (i % 100000 == 0)
    {
      std::cout << "Number of points processed:"
                << i << " / " << all_points.size() << std::endl;
    }
    if (! (center == all_points[i]))
    {
      if (is_point_in_voxel(bbox_min, bbox_max, all_points[i]))
      {
        neighborhood.push_back(all_points[i]);
      }
    }
  }

  return neighborhood;

}




Point OccupancyGrig::one_point_occ_grid(Point center,
                                      std::vector<Point> point_neighborhood,
                                      std::vector<float> grid_resolution,
                                      std::vector<int> grid_shape)
{


  return is_point_in_grid(voxel_grid_bounds(center,grid_resolution, grid_shape), center);
}
