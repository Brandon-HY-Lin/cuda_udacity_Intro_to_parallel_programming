# CMakeList.txt : CMake project for radix_sort, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("radix_sort")

file (GLOB src *.cpp)
file (GLOB headers *.h)
# Add source to this project's executable.
add_executable (radix_sort ${src} ${headers})

# TODO: Add tests and install targets if needed.
