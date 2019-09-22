cmake_minimum_required(VERSION 3.14)
project(Realtime_facedetection)

set(CMAKE_CXX_STANDARD 11)

find_package( OpenCV REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(Realtime_facedetection main.cpp)
TARGET_LINK_LIBRARIES(Realtime_facedetection ${OpenCV_LIBS})