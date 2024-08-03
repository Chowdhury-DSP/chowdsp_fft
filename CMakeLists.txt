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

include(CheckCXXCompilerFlag)
if(WIN32)
    CHECK_CXX_COMPILER_FLAG("/arch:AVX2" COMPILER_OPT_ARCH_AVX_SUPPORTED)
    if(COMPILER_OPT_ARCH_AVX_SUPPORTED)
        message(STATUS "Compiler supports flags: /arch:AVX2")
        target_compile_options(chowdsp_fft PRIVATE /arch:AVX2)
    else()
        message(STATUS "Compiler DOES NOT supports flags: /arch:AVX2")
    endif()
else()
    CHECK_CXX_COMPILER_FLAG("-mavx -mfma" COMPILER_OPT_ARCH_AVX_SUPPORTED)
    if(COMPILER_OPT_ARCH_AVX_SUPPORTED)
        message(STATUS "Compiler supports flags: -mavx -mfma")
        target_compile_options(chowdsp_fft PRIVATE -mavx -mfma -Wno-unused-command-line-argument)
    else()
        message(STATUS "Compiler DOES NOT supports flags: -mavx -mfma")
    endif()
endif()
