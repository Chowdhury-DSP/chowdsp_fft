cmake_minimum_required(VERSION 3.20)
project(chowdsp_fft)

add_library(chowdsp_fft STATIC)
target_sources(chowdsp_fft
    PRIVATE
        chowdsp_fft.h
        chowdsp_fft.cpp
)
target_include_directories(chowdsp_fft PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(chowdsp_fft PRIVATE _USE_MATH_DEFINES=1)
target_compile_features(chowdsp_fft PRIVATE cxx_std_20)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("/arch:AVX2" COMPILER_OPT_ARCH_AVX_MSVC_SUPPORTED)
CHECK_CXX_COMPILER_FLAG("-mavx -mfma" COMPILER_OPT_ARCH_AVX_GCC_CLANG_SUPPORTED)
if(COMPILER_OPT_ARCH_AVX_MSVC_SUPPORTED)
    message(STATUS "chowdsp_fft -- Compiler supports flags: /arch:AVX2")
    add_library(chowdsp_fft_avx STATIC simd/chowdsp_fft_impl_avx.cpp)
    target_compile_options(chowdsp_fft_avx PRIVATE /arch:AVX2)
    target_compile_definitions(chowdsp_fft_avx PRIVATE _USE_MATH_DEFINES=1)
    target_compile_features(chowdsp_fft_avx PRIVATE cxx_std_20)
    target_link_libraries(chowdsp_fft PRIVATE chowdsp_fft_avx)
    target_compile_definitions(chowdsp_fft PRIVATE CHOWDSP_FFT_COMPILER_SUPPORTS_AVX=1)
else()
    if(COMPILER_OPT_ARCH_AVX_GCC_CLANG_SUPPORTED)
        message(STATUS "chowdsp_fft -- Compiler supports flags: -mavx2 -mfma")
        add_library(chowdsp_fft_avx STATIC simd/chowdsp_fft_impl_avx.cpp)
        target_compile_options(chowdsp_fft_avx PRIVATE -mavx2 -mfma -Wno-unused-command-line-argument)
        target_compile_features(chowdsp_fft_avx PRIVATE cxx_std_20)
        target_compile_definitions(chowdsp_fft_avx PRIVATE _USE_MATH_DEFINES=1)
        target_link_libraries(chowdsp_fft PRIVATE chowdsp_fft_avx)
        target_compile_definitions(chowdsp_fft PRIVATE CHOWDSP_FFT_COMPILER_SUPPORTS_AVX=1)
    else()
        message(STATUS "chowdsp_fft -- Compiler DOES NOT supports flags: -mavx2 -mfma")
        target_compile_definitions(chowdsp_fft PRIVATE CHOWDSP_FFT_COMPILER_SUPPORTS_AVX=0)
    endif()
endif()

if(CHOWDSP_FFT_TOTAL_DEBUG)
    message(AUTHOR_WARNING "chowdsp_fft -- Skipping debug optimization flags!")
else()
    if(MSVC)
        # message(STATUS "CMAKE_C_FLAGS: ${CMAKE_C_FLAGS}")
        # message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
        # message(STATUS "CMAKE_C_FLAGS_DEBUG: ${CMAKE_C_FLAGS_DEBUG}")
        # message(STATUS "CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")

        # The "RTC" flags are incompatible with any optimization flags
        # We disable RTC here, but I believe this does not affect any "higher" level CMake settings.
        STRING (REGEX REPLACE "/RTC(su|[1su])" "" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
        STRING (REGEX REPLACE "/RTC(su|[1su])" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
        # message(STATUS "CMAKE_C_FLAGS_DEBUG: ${CMAKE_C_FLAGS_DEBUG}")
        # message(STATUS "CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")

        set(DEBUG_OPT_FLAGS /O2 /Ob2)
    else()
        set(DEBUG_OPT_FLAGS -O2)
    endif()
    message(STATUS "Setting debug optimization flags: ${DEBUG_OPT_FLAGS}")

    target_compile_options(chowdsp_fft PRIVATE $<$<CONFIG:Debug>:${DEBUG_OPT_FLAGS}>)
    if(TARGET chowdsp_fft_avx)
        target_compile_options(chowdsp_fft_avx PRIVATE $<$<CONFIG:Debug>:${DEBUG_OPT_FLAGS}>)
    endif()
endif()

## JUCE compatibility
if(EXISTS "${JUCE_MODULES_DIR}")
    message(STATUS "chowdsp_fft -- Configuring chowdsp_fft_juce")
    juce_add_module(chowdsp_fft_juce ALIAS_NAMESPACE chowdsp)
    target_link_libraries(chowdsp_fft_juce INTERFACE chowdsp_fft)
endif()

if(CHOWDSP_FFT_TESTING)
    add_subdirectory(test)
    add_subdirectory(bench)
endif()
