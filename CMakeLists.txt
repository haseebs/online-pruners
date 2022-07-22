cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_executable(loading-test loading-test.cpp)
target_link_libraries(loading-test "${TORCH_LIBRARIES}")
set_property(TARGET loading-test PROPERTY CXX_STANDARD 14)
