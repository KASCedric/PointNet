cmake_minimum_required(VERSION 3.9)

project(compute_occupancy CXX)

find_package(OpenMP REQUIRED)
find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


add_subdirectory (utils)

add_executable(${PROJECT_NAME} main.cxx)

target_link_libraries(${PROJECT_NAME} PUBLIC utils OpenMP::OpenMP_CXX  ${PCL_LIBRARIES})
