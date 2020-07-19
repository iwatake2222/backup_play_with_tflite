cmake_minimum_required(VERSION 3.0)

set(LibraryName "InferenceHelper")

set(TFLITE_DELEGATE_EDGETPU off CACHE BOOL "With Tflite Delegate EdgeTPU? [on/off]")

# Create library
add_library (${LibraryName} 
	InferenceHelper.h
	InferenceHelper.cpp
	InferenceHelperTensorflowLite.h
	InferenceHelperTensorflowLite.cpp
)

# For Tensorflow Lite
include(${CMAKE_CURRENT_LIST_DIR}/../third_party/cmakes/tensorflow.cmake)
target_include_directories(${LibraryName} PUBLIC ${TFLITE_INC})
target_link_libraries(${LibraryName} ${TFLITE_LIB})

# For Edge TPU
if(TFLITE_DELEGATE_EDGETPU)
	include(${CMAKE_SOURCE_DIR}/../third_party/cmakes/edgetpu.cmake)
	target_include_directories(${LibraryName} PUBLIC ${EDGETPI_INC})
	target_link_libraries(${LibraryName} ${EDGETPU_LIB})
	add_definitions(-DTFLITE_DELEGATE_EDGETPU)
endif()
