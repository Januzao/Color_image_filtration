cmake_minimum_required(VERSION 3.30)
project(photo)

set(CMAKE_CXX_STANDARD 20)

find_package(PNG REQUIRED)
find_package(Threads REQUIRED)

add_executable(photo main.cpp
        photo.cpp
        photo.hpp
)
target_link_libraries(photo PNG::PNG Threads::Threads)
