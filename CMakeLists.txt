cmake_minimum_required(VERSION 3.24)

project(cuml_go LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(cuml CONFIG REQUIRED)
find_package(raft CONFIG REQUIRED)
find_package(Treelite CONFIG REQUIRED)

set(CMAKE_SHARED_LINKER_FLAGS
    "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-as-needed")

add_library(cumlgo SHARED
    cpp/random_forest/random_forest.cpp
    cpp/common/cuda_utils.cpp
)

target_link_libraries(cumlgo
    PRIVATE
        cuml::cuml
)

add_executable(test_cuml
    cpp/main.cpp
)

target_link_libraries(test_cuml
    PRIVATE
        cumlgo
)
