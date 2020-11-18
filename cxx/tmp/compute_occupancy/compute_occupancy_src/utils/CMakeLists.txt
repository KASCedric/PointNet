find_package(OpenMP REQUIRED)

add_library(utils utils.cxx utils.h )

target_link_libraries(utils PUBLIC OpenMP::OpenMP_CXX ${PCL_LIBRARIES})
