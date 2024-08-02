cmake_minimum_required(VERSION 3.20)
project(chowdsp_fft)

set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

add_library(chowdsp_fft STATIC)
target_sources(chowdsp_fft
    PRIVATE
        chowdsp_fft.h
        chowdsp_fft.cpp
)
target_include_directories(chowdsp_fft PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(chowdsp_fft PRIVATE _USE_MATH_DEFINES=1)
