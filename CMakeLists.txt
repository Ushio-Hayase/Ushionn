cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(tesron LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

add_subdirectory(src/tensor)

target_include_directories(tensor PUBLIC include)