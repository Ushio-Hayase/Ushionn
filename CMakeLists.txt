cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(tesron LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

add_subdirectory(kernels)
add_subdirectory(src/main)

target_include_directories(kernels PUBLIC kernels)
target_link_libraries(main kernels)