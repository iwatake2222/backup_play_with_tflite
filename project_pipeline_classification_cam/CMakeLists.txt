cmake_minimum_required(VERSION 3.0)

# Create project
set(ProjectName "main")
project(${ProjectName})

# Switch build system
set(BUILD_SYSTEM auto CACHE STRING "Build target? [auto, x64_windows, x64_linux, armv7, aarch64]")
if(${BUILD_SYSTEM} STREQUAL "auto")
	if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "armv7l")
		set(BUILD_SYSTEM armv7)
	elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
		set(BUILD_SYSTEM aarch64)
	else()
		if(WIN32)
			set(BUILD_SYSTEM x64_windows)
		else()
			set(BUILD_SYSTEM x64_linux)
		endif()
	endif()
endif()
message("[main] CMAKE_SYSTEM_PROCESSOR = " ${CMAKE_SYSTEM_PROCESSOR} ", BUILD_SYSTEM = " ${BUILD_SYSTEM})

# Switch option to use EdgeTPU
set(USE_EDGETPU on CACHE BOOL "Use Edge TPU? [on/off]")

# Compile option
if(NOT MSVC)
	set(CMAKE_C_FLAGS "-Wall")
	set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG")
	set(CMAKE_C_FLAGS_DEBUG "-g3 -O0")
	set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -std=c++11 -lstdc++")
	set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})
	set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
	if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
		message(STATUS "No build type selected, default to Release")
		set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (default Debug)" FORCE)
	endif()
endif()

# Create executable file
add_executable(${ProjectName}
	main.cpp
)

# For OpenCV
find_package(OpenCV REQUIRED)
target_include_directories(${ProjectName} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${ProjectName} ${OpenCV_LIBS})

# For Tensorflow Lite and Edge TPU
include(${CMAKE_SOURCE_DIR}/../third_party/cmakes/tensorflow.cmake)
if(USE_EDGETPU)
	include(${CMAKE_SOURCE_DIR}/../third_party/cmakes/edgetpu.cmake)
	include(${CMAKE_SOURCE_DIR}/../third_party/cmakes/edgetpu_pipeline.cmake)
endif()

# Copy files
file(COPY ${CMAKE_SOURCE_DIR}/../resource DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
add_definitions(-DRESOURCE="resource")
