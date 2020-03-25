# CMakeList.txt : CMake project for gpu, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("gpu")

find_package (CUDA REQUIRED)

# Add source to this project's executable.
CUDA_ADD_EXECUTABLE (gpu "gpu.cu")

# TODO: Add tests and install targets if needed.
