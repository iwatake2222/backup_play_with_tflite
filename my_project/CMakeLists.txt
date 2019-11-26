cmake_minimum_required(VERSION 3.10)

set(ProjectName "main")

# Switch build target
set(BUILD_TARGET PC CACHE STRING "Build target?")
if(${BUILD_TARGET} STREQUAL RASPI)
	message("[BUILD] Raspberry Pi")
	set(LIB_TENSORFLOW_LITE ${CMAKE_SOURCE_DIR}/../third_party/tensorflow_prebuilt/rpi_armv7l/lib/libtensorflow-lite.a)
	set(LIB_EDGETPU         ${CMAKE_SOURCE_DIR}/../third_party/edgetpu/libedgetpu/direct/armv7a/libedgetpu.so.1.0)
elseif(${BUILD_TARGET} STREQUAL JETSON)
	message("[BUILD] Jetson")
	set(LIB_TENSORFLOW_LITE ${CMAKE_SOURCE_DIR}/../third_party/tensorflow_prebuilt/generic-aarch64_armv8-a/lib/libtensorflow-lite.a)
	set(LIB_EDGETPU         ${CMAKE_SOURCE_DIR}/../third_party/edgetpu/libedgetpu/direct/aarch64/libedgetpu.so.1.0)
else()
	message(FATAL_ERROR "[BUILD] Invalid target")
endif()

# Compile option
set(CMAKE_CXX_FLAGS "-Wall -std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "No build type selected, default to Release")
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (default Debug)" FORCE)
endif()

# Create project
project(${ProjectName})
add_executable(${ProjectName} main.cpp model_utils.cc model_utils.h)

# For OpenCV
find_package(OpenCV REQUIRED)
target_include_directories(${ProjectName} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${ProjectName} ${OpenCV_LIBS})

# For Tensorflow Lite and Edge TPU
target_include_directories(${ProjectName} PUBLIC ${CMAKE_SOURCE_DIR}/../third_party/edgetpu)
target_include_directories(${ProjectName} PUBLIC ${CMAKE_SOURCE_DIR}/../third_party/edgetpu/libedgetpu)
target_include_directories(${ProjectName} PUBLIC ${CMAKE_SOURCE_DIR}/../third_party/tensorflow)
target_include_directories(${ProjectName} PUBLIC ${CMAKE_SOURCE_DIR}/../third_party/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include)
target_link_libraries(${ProjectName} ${LIB_TENSORFLOW_LITE})
target_link_libraries(${ProjectName} ${LIB_EDGETPU})
target_link_libraries(${ProjectName} pthread m dl)

# Copy files
file(COPY resource DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
add_definitions(-DRESOURCE="resource")
file(COPY ${LIB_EDGETPU} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
