cmake_minimum_required(VERSION 3.10)
include(FindPkgConfig)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")


find_package(CAFFE REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV 3 REQUIRED)
find_package(Boost REQUIRED)

include_directories("/usr/include/glog")

add_executable(xor src/main.cc)
target_link_libraries(xor "/usr/lib/x86_64-linux-gnu/libglog.so.0" "/usr/lib/x86_64-linux-gnu/libgflags.so.2.2")
target_link_libraries(xor "/usr/local/lib/caffe/libcaffe.so")
target_link_libraries(xor "/usr/lib/x86_64-linux-gnu/libboost_system.so")