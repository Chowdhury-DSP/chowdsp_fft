#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>
#include <utility>

#include <chowdsp_fft.h>
#include <pffft.h>
#include <pffft.c>

// @TODO:
// These benchmark results are not very reliable at the moment.
// In particular the timings for small FFT sizes are way too noisy
// to get any meaningful info from. At the very least we should
// take an average from a bunch of runs. Maybe it's worth it
// to use a fully-fledged benchmarking framework like google/benchmark?

std::pair<float, float> bench_complex (int N)
{
    static constexpr int M = 50;

    auto* data = (float*) chowdsp::fft::aligned_malloc (sizeof (float) * N * 2);
    auto* data_ref = (float*) pffft_aligned_malloc (sizeof (float) * N * 2);
    auto* work_data = (float*) chowdsp::fft::aligned_malloc (sizeof (float) * N * 2);
    auto* work_data_ref = (float*) pffft_aligned_malloc (sizeof (float) * N * 2);

    for (int i = 0; i < N; ++i)
    {
        data[i * 2] = std::sin (3.14f * (100.0f / 48000.0f) * (float) i);
        data[i * 2 + 1] = std::cos (3.14f * (100.0f / 48000.0f) * (float) i);
    }
    std::copy (data, data + N * 2, data_ref);

    auto* fft_setup = chowdsp::fft::fft_new_setup (N, chowdsp::fft::FFT_COMPLEX);
    assert (fft_setup != nullptr);
    auto* pffft_setup = pffft_new_setup (N, PFFFT_COMPLEX);

    auto start = std::chrono::high_resolution_clock::now();
    for (int m = 0; m < M; ++m)
    {
        chowdsp::fft::fft_transform (fft_setup, data, data, work_data, chowdsp::fft::FFT_FORWARD);
        chowdsp::fft::fft_transform (fft_setup, data, data, work_data, chowdsp::fft::FFT_BACKWARD);
    }
    auto duration = std::chrono::high_resolution_clock::now() - start;
    auto test_duration_seconds = std::chrono::duration<float> (duration).count();
    std::cout << "    chowdsp_fft: " << test_duration_seconds << " seconds" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    for (int m = 0; m < M; ++m)
    {
        pffft_transform_ordered (pffft_setup, data_ref, data_ref, work_data_ref, PFFFT_FORWARD);
        pffft_transform_ordered (pffft_setup, data_ref, data_ref, work_data_ref, PFFFT_BACKWARD);
    }
    duration = std::chrono::high_resolution_clock::now() - start;
    auto ref_duration_seconds = std::chrono::duration<float> (duration).count();
    std::cout << "    pffft: " << ref_duration_seconds << " seconds" << std::endl;

    const auto speed_factor = ref_duration_seconds / test_duration_seconds;
    std::cout << "    speed: " << speed_factor << "x" << std::endl;

    chowdsp::fft::fft_destroy_setup (fft_setup);
    pffft_destroy_setup (pffft_setup);
    chowdsp::fft::aligned_free (data);
    pffft_aligned_free (data_ref);
    chowdsp::fft::aligned_free (work_data);
    pffft_aligned_free (work_data_ref);

    return { test_duration_seconds, ref_duration_seconds };
}

std::pair<float, float> bench_real (int N)
{
    static constexpr int M = 50;

    auto* data = (float*) chowdsp::fft::aligned_malloc (sizeof (float) * N);
    auto* data_ref = (float*) pffft_aligned_malloc (sizeof (float) * N);
    auto* work_data = (float*) chowdsp::fft::aligned_malloc (sizeof (float) * N);
    auto* work_data_ref = (float*) pffft_aligned_malloc (sizeof (float) * N);

    for (int i = 0; i < N; ++i)
    {
        data[i] = std::sin (3.14f * (100.0f / 48000.0f) * (float) i);
    }
    std::copy (data, data + N, data_ref);

    auto* fft_setup = chowdsp::fft::fft_new_setup (N, chowdsp::fft::FFT_REAL);
    assert (fft_setup != nullptr);
    auto* pffft_setup = pffft_new_setup (N, PFFFT_REAL);

    auto start = std::chrono::high_resolution_clock::now();
    for (int m = 0; m < M; ++m)
    {
        chowdsp::fft::fft_transform (fft_setup, data, data, work_data, chowdsp::fft::FFT_FORWARD);
        chowdsp::fft::fft_transform (fft_setup, data, data, work_data, chowdsp::fft::FFT_BACKWARD);
    }
    auto duration = std::chrono::high_resolution_clock::now() - start;
    auto test_duration_seconds = std::chrono::duration<float> (duration).count();
    std::cout << "    chowdsp_fft: " << test_duration_seconds << " seconds" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    for (int m = 0; m < M; ++m)
    {
        pffft_transform_ordered (pffft_setup, data_ref, data_ref, work_data_ref, PFFFT_FORWARD);
        pffft_transform_ordered (pffft_setup, data_ref, data_ref, work_data_ref, PFFFT_BACKWARD);
    }
    duration = std::chrono::high_resolution_clock::now() - start;
    auto ref_duration_seconds = std::chrono::duration<float> (duration).count();
    std::cout << "    pffft: " << ref_duration_seconds << " seconds" << std::endl;

    const auto speed_factor = ref_duration_seconds / test_duration_seconds;
    std::cout << "    speed: " << speed_factor << "x" << std::endl;

    chowdsp::fft::fft_destroy_setup (fft_setup);
    pffft_destroy_setup (pffft_setup);
    chowdsp::fft::aligned_free (data);
    pffft_aligned_free (data_ref);
    chowdsp::fft::aligned_free (work_data);
    pffft_aligned_free (work_data_ref);

    return { test_duration_seconds, ref_duration_seconds };
}

int main()
{
    std::vector<float> durations_cplx_test (15, 0.0f);
    std::vector<float> durations_cplx_ref (15, 0.0f);
    std::vector<float> durations_real_test (15, 0.0f);
    std::vector<float> durations_real_ref (15, 0.0f);
    for (size_t i = 5; i < 20; ++i)
    {
        const auto fft_size = 1 << i;
        std::cout << "Benchmarking Complex FFT with size: " << fft_size << std::endl;
        std::tie (durations_cplx_test[i-5], durations_cplx_ref[i-5]) = bench_complex (1 << i);
        std::cout << "Benchmarking Real FFT with size: " << fft_size << std::endl;
        std::tie (durations_real_test[i-5], durations_real_ref[i-5]) = bench_real (1 << i);
    }

    return 0;
}
