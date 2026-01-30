cmake_minimum_required(VERSION 4.1)
project(LexiCore)

set(CMAKE_CXX_STANDARD 20)

add_executable(LexiCore main.cpp
        src/io/file_reader.cpp
        src/preprocess/preprocessing.cpp
        src/similarity/cosine.cpp
        src/vectorize/bow.cpp
        src/vectorize/TF-IDF.cpp)

target_include_directories(LexiCore PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)
