cmake_minimum_required(VERSION 3.0)

set(ProjectName "main")

set(USE_EDGETPU on CACHE BOOL "Use Edge TPU? [on/off]")

# Switch build target
set(ARCH_TYPE x64 CACHE STRING "Build target? [x64, armv7, aarch64]")
if(${ARCH_TYPE} STREQUAL armv7)
	message("[BUILD] armv7")
	set(LIB_TENSORFLOW_LITE ${CMAKE_SOURCE_DIR}/../third_party/tensorflow_prebuilt/rpi_armv7l/lib/libtensorflow-lite.a)
	set(LIB_EDGETPU         ${CMAKE_SOURCE_DIR}/../third_party/edgetpu_runtime/libedgetpu/direct/armv7a/libedgetpu.so.1.0)
	set(LIBS pthread m dl)
elseif(${ARCH_TYPE} STREQUAL aarch64)
	message("[BUILD] aarch64")
	set(LIB_TENSORFLOW_LITE ${CMAKE_SOURCE_DIR}/../third_party/tensorflow_prebuilt/generic-aarch64_armv8-a/lib/libtensorflow-lite.a)
	set(LIB_EDGETPU         ${CMAKE_SOURCE_DIR}/../third_party/edgetpu_runtime/libedgetpu/direct/aarch64/libedgetpu.so.1.0)
	set(LIBS pthread m dl)
else()
	if(WIN32)
		message("[BUILD] windows (x64)")
		set(LIB_TENSORFLOW_LITE ${CMAKE_SOURCE_DIR}/../third_party/tensorflow_prebuilt/x64_windows/lib/tensorflowlite.dll.if.lib)
		file(COPY ${CMAKE_SOURCE_DIR}/../third_party/tensorflow_prebuilt/x64_windows/lib/tensorflowlite.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
		set(LIB_EDGETPU         ${CMAKE_SOURCE_DIR}/../third_party/edgetpu_runtime/libedgetpu/direct/x64_windows/edgetpu.dll.if.lib)
		set(LIBS )
	else()
		message("[BUILD] linux (x64)")
		set(LIB_TENSORFLOW_LITE ${CMAKE_SOURCE_DIR}/../third_party/tensorflow_prebuilt/x64_linux/lib/libtensorflow-lite.a)
		set(LIB_EDGETPU         ${CMAKE_SOURCE_DIR}/../third_party/edgetpu_runtime/libedgetpu/direct/k8/libedgetpu.so.1.0)
		set(LIBS pthread m dl)
	endif()
endif()

if(USE_EDGETPU)
	add_definitions(-DUSE_EDGETPU)
	if(WIN32)
		file(COPY ${CMAKE_SOURCE_DIR}/../third_party/edgetpu_runtime/libedgetpu/direct/x64_windows/edgetpu.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
	else()
		file(COPY ${LIB_EDGETPU} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
	endif()
else()
	set(LIB_EDGETPU)
endif()

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
target_include_directories(${ProjectName} PUBLIC ${CMAKE_SOURCE_DIR}/../third_party/tensorflow)
target_include_directories(${ProjectName} PUBLIC ${CMAKE_SOURCE_DIR}/../third_party/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include)
target_include_directories(${ProjectName} PUBLIC ${CMAKE_SOURCE_DIR}/../third_party/tensorflow/tensorflow/lite/tools/make/downloads/absl)
target_include_directories(${ProjectName} PUBLIC ${CMAKE_SOURCE_DIR}/../third_party/edgetpu_runtime)
target_include_directories(${ProjectName} PUBLIC ${CMAKE_SOURCE_DIR}/../third_party/edgetpu_runtime/libedgetpu)

target_link_libraries(${ProjectName} ${LIB_TENSORFLOW_LITE})
target_link_libraries(${ProjectName} ${LIB_EDGETPU})
target_link_libraries(${ProjectName} ${LIBS})

# Copy files
file(COPY ${CMAKE_SOURCE_DIR}/../resource DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
add_definitions(-DRESOURCE="resource")

