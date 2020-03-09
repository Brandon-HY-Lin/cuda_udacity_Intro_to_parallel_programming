# CMakeList.txt : CMake project for problem_set_1_rgb_to_grey_cmake, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)
project (rbg_to_grey)

find_package(CUDA REQUIRED)

IF(WIN32)
	SET(OpenCV_DIR "C:/Users/cvtoe/Documents/opencv/build/x64/vc15/lib" )
ENDIF()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add source to this project's executable.
file(GLOB cu *.cu *.cpp)
file(GLOB headers *.h *.hpp *.cuh)

CUDA_ADD_EXECUTABLE(rbg_to_grey ${cu} ${headers})
target_link_libraries(rbg_to_grey ${OpenCV_LIBS})
# TODO: Add tests and install targets if needed.
