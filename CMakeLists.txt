cmake_minimum_required(VERSION 3.15)
project(riona LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(riona
    src/main.cpp
    src/util.cpp
    src/arff_reader.cpp
    src/distance.cpp
    src/algorithms.cpp
    src/metrics.cpp
    src/output.cpp
)

target_include_directories(riona PRIVATE include)
