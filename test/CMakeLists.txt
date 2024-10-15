message(STATUS "chowdsp_fft -- Configuring tests")

include(CPM.cmake)
CPMAddPackage("gh:catchorg/Catch2@3.2.1")
include(${Catch2_SOURCE_DIR}/extras/Catch.cmake)
CPMAddPackage("bb:jpommier/pffft#master")

add_executable(test_chowsdp_fft_cpp test.cpp ${pffft_SOURCE_DIR}/pffft.c)
target_link_libraries(test_chowsdp_fft_cpp PRIVATE chowdsp_fft Catch2::Catch2WithMain)
target_include_directories(test_chowsdp_fft_cpp PRIVATE ${pffft_SOURCE_DIR})
target_compile_definitions(test_chowsdp_fft_cpp PRIVATE _USE_MATH_DEFINES=1)
