cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(tesron LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

add_subdirectory("lib")

set(USE_CUDNN ON)

include(FetchContent)

FetchContent_Declare(googletest URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip)
FetchContent_MakeAvailable(googletest)

if(USE_CUDNN)
    add_definitions(-DUSE_CUDNN)
endif()

add_subdirectory("src")







