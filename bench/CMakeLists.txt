message(STATUS "chowdsp_fft -- Configuring benchmarks ${CMAKE_BINARY_DIR}/_deps/pffft-src")

add_executable(bench_chowdsp_fft bench.cpp)
target_link_libraries(bench_chowdsp_fft PRIVATE chowdsp_fft)
target_include_directories(bench_chowdsp_fft PRIVATE ${CMAKE_BINARY_DIR}/_deps/pffft-src)
target_compile_definitions(bench_chowdsp_fft PRIVATE _USE_MATH_DEFINES=1)
