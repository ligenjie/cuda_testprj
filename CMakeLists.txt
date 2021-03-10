cmake_minimum_required(VERSION 3.5.11)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(cuda_testprj C CXX CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -static-libstdc++")
#set(CMAKE_CUDA_STANDARD 11)
find_package(CUDA REQUIRED)
#find_package(OpenCV REQUIRED)

SET(CUDA_PATH "/usr/local/cuda")
SET(CUDA_LIBS  ${CUDA_PATH}/lib64)
INCLUDE_DIRECTORIES(${CUDA_PATH}/include)
INCLUDE_DIRECTORIES(${PROJECT_NAME}/include)
link_directories(
    CUDA_LIBS
)
FILE(GLOB_RECURSE CUDA_LIBS ${CUDA_PATH}/lib64/*.so)
MESSAGE(STATUS "CUDA_LIBS = ${CUDA_LIBS}")


#CUDA_ADD_EXECUTABLE(${PROJECT_NAME} ${SRC_LIST})
cuda_add_executable(cuda_testprj main.cu chapter5/sharedMemoryTest.cu chapter5/sharedMemoryTest.cuh)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${CUDA_PATH})

set_target_properties(
        cuda_testprj
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

