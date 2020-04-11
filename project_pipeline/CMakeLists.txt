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
add_executable(${ProjectName} main.cpp
	${CMAKE_SOURCE_DIR}/../third_party/edgetpu/src/cpp/pipeline/allocator.h
	${CMAKE_SOURCE_DIR}/../third_party/edgetpu/src/cpp/pipeline/common.h
	${CMAKE_SOURCE_DIR}/../third_party/edgetpu/src/cpp/pipeline/utils.cc
	${CMAKE_SOURCE_DIR}/../third_party/edgetpu/src/cpp/pipeline/utils.h
	${CMAKE_SOURCE_DIR}/../third_party/edgetpu/src/cpp/pipeline/pipelined_model_runner.cc
	${CMAKE_SOURCE_DIR}/../third_party/edgetpu/src/cpp/pipeline/pipelined_model_runner.h
	${CMAKE_SOURCE_DIR}/../third_party/edgetpu/src/cpp/pipeline/internal/aligned_alloc.h
	${CMAKE_SOURCE_DIR}/../third_party/edgetpu/src/cpp/pipeline/internal/default_allocator.h
	${CMAKE_SOURCE_DIR}/../third_party/edgetpu/src/cpp/pipeline/internal/memory_pool_allocator.cc
	${CMAKE_SOURCE_DIR}/../third_party/edgetpu/src/cpp/pipeline/internal/memory_pool_allocator.h
	${CMAKE_SOURCE_DIR}/../third_party/edgetpu/src/cpp/pipeline/internal/segment_runner.cc
	${CMAKE_SOURCE_DIR}/../third_party/edgetpu/src/cpp/pipeline/internal/segment_runner.h
	${CMAKE_SOURCE_DIR}/../third_party/edgetpu/src/cpp/pipeline/internal/thread_safe_queue.h
)

# For OpenCV
find_package(OpenCV REQUIRED)
target_include_directories(${ProjectName} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${ProjectName} ${OpenCV_LIBS})

# For Tensorflow Lite and Edge TPU
include(${CMAKE_SOURCE_DIR}/../third_party/cmakes/tensorflow.cmake)
if(USE_EDGETPU)
	include(${CMAKE_SOURCE_DIR}/../third_party/cmakes/edgetpu.cmake)
endif()

# For Edge TPU Pipiline
set(DEPEND_LIB_ROOT ${PROJECT_BINARY_DIR}/Depends)

target_include_directories(${ProjectName} PUBLIC ${CMAKE_SOURCE_DIR}/../third_party/edgetpu/src/cpp/pipeline)
target_include_directories(${ProjectName} PUBLIC ${CMAKE_SOURCE_DIR}/../third_party/edgetpu/src/cpp/pipeline/internal)
target_include_directories(${ProjectName} PUBLIC ${CMAKE_SOURCE_DIR}/../third_party/tensorflow/tensorflow/lite/tools/make/downloads/absl)
target_include_directories(${ProjectName} PUBLIC ${DEPEND_LIB_ROOT}/include)

include(ExternalProject)
ExternalProject_Add(
  glog
  GIT_REPOSITORY https://github.com/google/glog
  GIT_TAG  3ba8976592274bc1f907c402ce22558011d6fc5e
  SOURCE_DIR ${DEPEND_LIB_ROOT}/src/glog
  BINARY_DIR ${DEPEND_LIB_ROOT}/build/glog
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${DEPEND_LIB_ROOT}
)
add_definitions("-DGLOG_NO_ABBREVIATED_SEVERITIES")
if(WIN32)
	target_link_libraries(${ProjectName} ${DEPEND_LIB_ROOT}/lib/glog.lib)
else()
	target_link_libraries(${ProjectName} ${DEPEND_LIB_ROOT}/lib/libglog.a)
endif()
add_dependencies(${ProjectName} glog)

ExternalProject_Add(
  absl
  GIT_REPOSITORY https://github.com/abseil/abseil-cpp/
  GIT_TAG  1112609635037a32435de7aa70a9188dcb591458
  SOURCE_DIR ${DEPEND_LIB_ROOT}/src/absl
  BINARY_DIR ${DEPEND_LIB_ROOT}/build/absl
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${DEPEND_LIB_ROOT}
)
if(WIN32)
	file(GLOB ABSL_LIBS "${DEPEND_LIB_ROOT}/lib/*.lib")
	target_link_libraries(${ProjectName} ${ABSL_LIBS})
else()
	file(GLOB ABSL_LIBS "${DEPEND_LIB_ROOT}/lib/libabsl_*.a")
	target_link_libraries(${ProjectName} ${ABSL_LIBS})
endif()
add_dependencies(${ProjectName} absl)


# Copy files
file(COPY ${CMAKE_SOURCE_DIR}/../resource DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
add_definitions(-DRESOURCE="resource")
