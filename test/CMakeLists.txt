message(STATUS "chowdsp_fft -- Configuring tests")

include(CPM.cmake)
CPMAddPackage("gh:catchorg/Catch2@3.2.1")
include(${Catch2_SOURCE_DIR}/extras/Catch.cmake)
CPMAddPackage("bb:jpommier/pffft#fbc4058602803f40dc554b8a5d2bcc694c005f2f")

add_executable(test_chowdsp_fft_cpp test.cpp ${pffft_SOURCE_DIR}/pffft.c)
target_link_libraries(test_chowdsp_fft_cpp PRIVATE chowdsp_fft Catch2::Catch2WithMain)
target_include_directories(test_chowdsp_fft_cpp PRIVATE ${pffft_SOURCE_DIR})
target_compile_definitions(test_chowdsp_fft_cpp PRIVATE _USE_MATH_DEFINES=1)

if(CHOWDSP_FFT_COVERAGE)
    message(STATUS "chowdsp_fft -- Appending code coverage compiler flags: -g --coverage")
    target_compile_options(chowdsp_fft_avx PUBLIC -g --coverage)
    target_compile_options(chowdsp_fft PUBLIC -g --coverage)
    target_compile_options(test_chowdsp_fft_cpp PUBLIC -O0 -g --coverage)
    target_link_options(test_chowdsp_fft_cpp PUBLIC --coverage)
endif()

add_executable(test_chowdsp_fft_c test.c ${pffft_SOURCE_DIR}/pffft.c)
target_link_libraries(test_chowdsp_fft_c PRIVATE chowdsp_fft)
target_include_directories(test_chowdsp_fft_c PRIVATE ${pffft_SOURCE_DIR})
target_compile_definitions(test_chowdsp_fft_c PRIVATE _USE_MATH_DEFINES=1)
