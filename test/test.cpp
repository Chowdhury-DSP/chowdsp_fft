#include <chowdsp_fft.h>
#include <pffft.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

void compare (const float* ref, const float* test, int N)
{
    const auto tol = 1.0e-6f * (float) N / 8.0f;
    for (int n = 0; n < N; ++n)
        REQUIRE (test[n] == Catch::Approx { ref[n] }.margin(tol));
}

void test_complex (int N, bool use_avx = false)
{
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

    auto* fft_setup = chowdsp::fft::fft_new_setup (N, chowdsp::fft::FFT_COMPLEX, use_avx);
    assert (fft_setup != nullptr);
    auto* pffft_setup = pffft_new_setup (N, PFFFT_COMPLEX);

    chowdsp::fft::fft_transform (fft_setup, data, data, work_data, chowdsp::fft::FFT_FORWARD);
    pffft_transform_ordered (pffft_setup, data_ref, data_ref, work_data_ref, PFFFT_FORWARD);

    compare (data_ref, data, N * 2);

    chowdsp::fft::fft_transform (fft_setup, data, data, work_data, chowdsp::fft::FFT_BACKWARD);
    pffft_transform_ordered (pffft_setup, data_ref, data_ref, work_data_ref, PFFFT_BACKWARD);

    const auto norm_gain = 1.0f / static_cast<float> (N);
    for (int n = 0; n < N * 2; ++n)
    {
        data[n] *= norm_gain;
        data_ref[n] *= norm_gain;
    }

    compare (data_ref, data, N * 2);

    chowdsp::fft::fft_destroy_setup (fft_setup);
    pffft_destroy_setup (pffft_setup);
    chowdsp::fft::aligned_free (data);
    pffft_aligned_free (data_ref);
    chowdsp::fft::aligned_free (work_data);
    pffft_aligned_free (work_data_ref);
}

void test_real (int N, bool use_avx = false)
{
    auto* data = (float*) chowdsp::fft::aligned_malloc (sizeof (float) * N);
    auto* data_ref = (float*) pffft_aligned_malloc (sizeof (float) * N);
    auto* work_data = (float*) chowdsp::fft::aligned_malloc (sizeof (float) * N);
    auto* work_data_ref = (float*) pffft_aligned_malloc (sizeof (float) * N);

    for (int i = 0; i < N; ++i)
    {
        data[i] = std::sin (3.14f * (100.0f / 48000.0f) * (float) i);
    }
    std::copy (data, data + N, data_ref);

    auto* fft_setup = chowdsp::fft::fft_new_setup (N, chowdsp::fft::FFT_REAL, use_avx);
    assert (fft_setup != nullptr);
    auto* pffft_setup = pffft_new_setup (N, PFFFT_REAL);

    chowdsp::fft::fft_transform (fft_setup, data, data, work_data, chowdsp::fft::FFT_FORWARD);
    pffft_transform_ordered (pffft_setup, data_ref, data_ref, work_data_ref, PFFFT_FORWARD);
    
    compare (data_ref, data, N);

    chowdsp::fft::fft_transform (fft_setup, data, data, work_data, chowdsp::fft::FFT_BACKWARD);
    pffft_transform_ordered (pffft_setup, data_ref, data_ref, work_data_ref, PFFFT_BACKWARD);

    const auto norm_gain = 1.0f / static_cast<float> (N);
    for (int n = 0; n < N; ++n)
    {
        data[n] *= norm_gain;
        data_ref[n] *= norm_gain;
    }

    compare (data_ref, data, N);

    chowdsp::fft::fft_destroy_setup (fft_setup);
    pffft_destroy_setup (pffft_setup);
    chowdsp::fft::aligned_free (data);
    pffft_aligned_free (data_ref);
    chowdsp::fft::aligned_free (work_data);
    pffft_aligned_free (work_data_ref);
}

TEST_CASE("FFT SSE/NEON")
{
    for (int i = 5; i < 20; ++i)
    {
        const auto fft_size = 1 << i;
        SECTION ("Testing Complex FFT with size: " + std::to_string (fft_size))
        {
            test_complex (fft_size);
        }

        SECTION ("Testing Real FFT with size: " + std::to_string (fft_size))
        {
            test_real (fft_size);
        }
    }
}

#if defined(__SSE2__)
TEST_CASE("FFT AVX")
{
    for (int i = 5; i < 20; ++i)
    {
        const auto fft_size = 1 << i;
        SECTION ("Testing Complex FFT with size: " + std::to_string (fft_size))
        {
            test_complex (fft_size, true);
        }

        SECTION ("Testing Real FFT with size: " + std::to_string (fft_size))
        {
            test_real (fft_size, true);
        }
    }
}
#endif
