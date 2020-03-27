# CMakeList.txt : CMake project for problem_set_4_red_eye_removal, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("problem_set_4_red_eye_removal")

IF (WIN32)
	SET (OpenCV_DIR "C:/Users/cvtoe/Documents/opencv/build/x64/vc15/lib")
ENDIF() 
 
find_package (OpenCV REQUIRED)

include_directories (${OpenCV_INCLUDE_DIRS} )


find_package (CUDA REQUIRED)

file (GLOB src *.cu *.cpp *.c)
file (GLOB headers *.h *.hpp *.cuh)

# Add source to this project's executable.
CUDA_ADD_EXECUTABLE (problem_set_4_red_eye_removal ${src} ${headers})

target_link_libraries (problem_set_4_red_eye_removal ${OpenCV_LIBS})
# TODO: Add tests and install targets if needed.
