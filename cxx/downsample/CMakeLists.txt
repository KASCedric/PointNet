cmake_minimum_required(VERSION 3.17)
project(Downsample CXX)

set(CMAKE_CXX_STANDARD 14)

find_package(PCL 1.8 REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


add_executable(${PROJECT_NAME} main.cxx)

target_link_libraries(${PROJECT_NAME} PUBLIC utils OpenMP::OpenMP_CXX  ${PCL_LIBRARIES})
