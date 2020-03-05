cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(thrust)

find_package(CUDA REQUIRED)

file(GLOB cu *.cu)

CUDA_ADD_EXECUTABLE(thrust.out ${cu})
