cmake_minimum_required(VERSION 2.8.12)
project(dlibExamples)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_CXX_STANDARD 11)

include(thirdparty/dlib/cmake)
include_directories(include)
# equals to
# add_subdirectory(../dlib dlib_build)


set(CMAKE_PREFIX_PATH "/home/cuizhou/Downloads/opencv-3.3.0/build_cuda")
find_package(OpenCV 3.3.0 REQUIRED)

add_subdirectory(src)
add_subdirectory(executable)
