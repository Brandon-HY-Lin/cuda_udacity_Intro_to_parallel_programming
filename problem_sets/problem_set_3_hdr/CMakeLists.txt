# CMakeList.txt : CMake project for problem_set_3_hdr, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("hdr")

find_package (CUDA REQUIRED)


IF (WIN32)
	SET (OpenCV_DIR  "C:/Users/cvtoe/Documents/opencv/build/x64/vc15/lib")
ENDIF()

find_package (OpenCV REQUIRED)
include_directories (${OpenCV_INCLUDE_DIRS})

file (GLOB src *.cpp *.cu)
file (GLOB headers *.hpp *.h *.cuh)

# Add source to this project's executable.
CUDA_ADD_EXECUTABLE (hdr ${src} ${headers})

target_link_libraries (hdr ${OpenCV_LIBS})
# TODO: Add tests and install targets if needed.
