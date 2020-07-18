cmake_minimum_required(VERSION 3.0)

set(LibraryName "InferenceHelper")

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
