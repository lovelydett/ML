CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

project(libtorch_helloworld)

set(CMAKE_CXX_STANDARD 14) # Must >= 14!!

set(CMAKE_PREFIX_PATH "/usr/local/libtorch")

find_package(Torch REQUIRED)

add_executable(xor_infer xor_infer.cc)
target_link_libraries(xor_infer ${TORCH_LIBRARIES})