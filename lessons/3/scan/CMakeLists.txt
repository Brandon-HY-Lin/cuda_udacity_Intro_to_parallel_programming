# CMakeList.txt : CMake project for scan, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("scan")

find_package (CUDA REQUIRED)

file (GLOB src *.cu *.cpp)
file (GLOB headers *.h *.cuh)

# Add source to this project's executable.
CUDA_ADD_EXECUTABLE (scan ${src} ${headers})

# TODO: Add tests and install targets if needed.
