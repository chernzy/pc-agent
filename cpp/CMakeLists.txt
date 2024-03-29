cmake_minimum_required(VERSION 3.0)
project(cpp)

set(CMAKE_CXX_STANDARD 11)

add_executable(cpp install.cpp)

target_link_libraries(cpp Advapi32)