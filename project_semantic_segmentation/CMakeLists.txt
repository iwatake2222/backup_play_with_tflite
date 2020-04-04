cmake_minimum_required(VERSION 3.0)

set(ProjectName "main")

# Switch build target
set(ARCH_TYPE x64 CACHE STRING "Build target? [x64, armv7, aarch64]")
set(USE_EDGETPU on CACHE BOOL "Use Edge TPU? [on/off]")

# Compile option
if(NOT WIN32)
	set(CMAKE_CXX_FLAGS "-Wall -std=c++11")
	set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
	set(CMAKE_CXX_FLAGS_DEBUG "-g3")
	if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
		message(STATUS "No build type selected, default to Release")
		set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (default Debug)" FORCE)
	endif()
endif()

# Create project
project(${ProjectName})
add_executable(${ProjectName} main.cpp)

# For OpenCV
find_package(OpenCV REQUIRED)
target_include_directories(${ProjectName} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${ProjectName} ${OpenCV_LIBS})

# For Tensorflow Lite and Edge TPU
include(${CMAKE_SOURCE_DIR}/../third_party/cmakes/tensorflow.cmake)
if(USE_EDGETPU)
	include(${CMAKE_SOURCE_DIR}/../third_party/cmakes/edgetpu.cmake)
endif()

# Copy files
file(COPY ${CMAKE_SOURCE_DIR}/../resource DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
add_definitions(-DRESOURCE="resource")
