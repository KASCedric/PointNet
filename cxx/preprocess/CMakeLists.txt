cmake_minimum_required(VERSION 3.17)
project(preprocess CXX)

set(CMAKE_CXX_STANDARD 14)

find_package(PCL 1.8 REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

#add_subdirectory (utils)

add_executable(${PROJECT_NAME} main.cxx json.hpp)

target_link_libraries(${PROJECT_NAME} PUBLIC boost_program_options OpenMP::OpenMP_CXX  ${PCL_LIBRARIES})
