cmake_minimum_required(VERSION 3.30)

project(cuml_go LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(cuml CONFIG REQUIRED)
find_package(raft CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

add_library(cumlgo SHARED
    cpp/random_forest/random_forest.cpp
    cpp/random_forest_cpu/random_forest_cpu.cpp
    cpp/kmeans/kmeans.cpp
    cpp/kmeans_cpu/kmeans_cpu.cpp
    cpp/knn/knn.cpp
    cpp/knn_cpu/knn_cpu.cpp
    cpp/common/cuda_utils.cpp
)

target_include_directories(cumlgo PRIVATE
    ${CMAKE_SOURCE_DIR}/cpp
    /usr/include/x86_64-linux-gnu
    ${CONDA_PREFIX}/include
)

target_link_libraries(cumlgo PRIVATE
    cuml::cuml
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    OpenMP::OpenMP_CXX
    /home/anupam/opt/onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so
)

target_compile_definitions(cumlgo PRIVATE
    ARMA_DONT_USE_WRAPPER
)