#define minumum CMake version
cmake_minimum_required (VERSION 2.6)

#define the project name
project (proyecto)

find_package( OpenCV REQUIRED )

#add executable to build environment
add_executable(proyecto fase1.cpp)

#copy executable to bin directory
install (TARGETS proyecto DESTINATION bin)

#import the library of openCV 
target_link_libraries( proyecto ${OpenCV_LIBS} )