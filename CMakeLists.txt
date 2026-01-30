cmake_minimum_required(VERSION 3.20)
project(LexiCore LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(lexicore)
target_include_directories(lexicore PUBLIC ${CMAKE_SOURCE_DIR}/include)

target_sources(lexicore PRIVATE
        src/io/file_reader.cpp
        src/preprocess/preprocessing.cpp
        src/similarity/cosine.cpp
        src/vectorize/bow.cpp
        src/vectorize/TF-IDF.cpp
        src/search/similarity_search.cpp
)

add_executable(lexicore_app main.cpp)
target_link_libraries(lexicore_app PRIVATE lexicore)

enable_testing()
add_subdirectory(tests)