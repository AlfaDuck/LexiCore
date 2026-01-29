cmake_minimum_required(VERSION 4.1)
project(NLPTK)

set(CMAKE_CXX_STANDARD 20)

add_executable(NLPTK main.cpp
        src/io/file_reader.cpp
        src/preprocess/preprocessing.cpp
        src/similarity/cosine.cpp
        src/vectorize/bow.cpp)

target_include_directories(NLPTK PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)
