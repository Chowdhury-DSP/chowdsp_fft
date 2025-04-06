#include <cmath>

#include <chowdsp_fft.h>
#include <pffft.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

void compare (const float* ref, const float* test, int N)
{
    const auto tol = 2.0e-7f * (float) N;
    for (int n = 0; n < N; ++n)
        REQUIRE (test[n] == Catch::Approx { ref[n] }.margin(tol));
}

void test_fft_complex (int N, bool use_avx = false)
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
    REQUIRE (fft_setup != nullptr);
    auto* pffft_setup = pffft_new_setup (N, PFFFT_COMPLEX);

    if (use_avx)
        REQUIRE (chowdsp::fft::fft_simd_width_bytes (fft_setup) == 32);
    else
        REQUIRE (chowdsp::fft::fft_simd_width_bytes (fft_setup) == 16);

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

void test_fft_real (int N, bool use_avx = false)
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
    REQUIRE (fft_setup != nullptr);
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

void test_convolution_complex (int N, bool use_avx = false)
{
    auto* sine1 = (float*) chowdsp::fft::aligned_malloc (sizeof (float) * N * 2);
    auto* sine1_ref = (float*) pffft_aligned_malloc (sizeof (float) * N * 2);
    auto* sine2 = (float*) chowdsp::fft::aligned_malloc (sizeof (float) * N * 2);
    auto* sine2_ref = (float*) pffft_aligned_malloc (sizeof (float) * N * 2);
    auto* out = (float*) chowdsp::fft::aligned_malloc (sizeof (float) * N * 2);
    auto* out_ref = (float*) pffft_aligned_malloc (sizeof (float) * N * 2);
    auto* work_data = (float*) chowdsp::fft::aligned_malloc (sizeof (float) * N * 2);
    auto* work_data_ref = (float*) pffft_aligned_malloc (sizeof (float) * N * 2);

    for (int i = 0; i < N; ++i)
    {
        sine1[i * 2] = std::sin (3.14f * (100.0f / 48000.0f) * (float) i);
        sine1[i * 2 + 1] = std::cos (3.14f * (100.0f / 48000.0f) * (float) i);
        sine2[i * 2] = std::sin (3.14f * (200.0f / 48000.0f) * (float) i);
        sine2[i * 2 + 1] = std::cos (3.14f * (200.0f / 48000.0f) * (float) i);
    }
    std::copy (sine1, sine1 + N * 2, sine1_ref);
    std::copy (sine2, sine2 + N * 2, sine2_ref);
    std::fill_n (out, N * 2, 0.0f);
    std::fill_n (out_ref, N * 2, 0.0f);
    const auto norm_gain = 1.0f / static_cast<float> (N);

    auto* pffft_setup = pffft_new_setup (N, PFFFT_COMPLEX);
    pffft_transform (pffft_setup, sine1_ref, sine1_ref, work_data_ref, PFFFT_FORWARD);
    pffft_transform (pffft_setup, sine2_ref, sine2_ref, work_data_ref, PFFFT_FORWARD);
    pffft_zconvolve_accumulate (pffft_setup, sine1_ref, sine2_ref, out_ref, norm_gain);
    pffft_transform (pffft_setup, out_ref, out_ref, work_data_ref, PFFFT_BACKWARD);

    auto* fft_setup = chowdsp::fft::fft_new_setup (N, chowdsp::fft::FFT_COMPLEX, use_avx);
    REQUIRE (fft_setup != nullptr);
    chowdsp::fft::fft_transform_unordered (fft_setup, sine1, sine1, work_data, chowdsp::fft::FFT_FORWARD);
    chowdsp::fft::fft_transform_unordered (fft_setup, sine2, sine2, work_data, chowdsp::fft::FFT_FORWARD);
    chowdsp::fft::fft_convolve_unordered (fft_setup, sine1, sine2, out, norm_gain);
    chowdsp::fft::fft_transform_unordered (fft_setup, out, out, work_data, chowdsp::fft::FFT_BACKWARD);

    compare (out_ref, out, N);

    chowdsp::fft::fft_destroy_setup (fft_setup);
    pffft_destroy_setup (pffft_setup);
    chowdsp::fft::aligned_free (sine1);
    pffft_aligned_free (sine1_ref);
    chowdsp::fft::aligned_free (sine2);
    pffft_aligned_free (sine2_ref);
    chowdsp::fft::aligned_free (out);
    pffft_aligned_free (out_ref);
    chowdsp::fft::aligned_free (work_data);
    pffft_aligned_free (work_data_ref);
}

void test_convolution_real (int N, bool use_avx = false)
{
    auto* sine1 = (float*) chowdsp::fft::aligned_malloc (sizeof (float) * N);
    auto* sine1_ref = (float*) pffft_aligned_malloc (sizeof (float) * N);
    auto* sine2 = (float*) chowdsp::fft::aligned_malloc (sizeof (float) * N);
    auto* sine2_ref = (float*) pffft_aligned_malloc (sizeof (float) * N);
    auto* out = (float*) chowdsp::fft::aligned_malloc (sizeof (float) * N);
    auto* out_ref = (float*) pffft_aligned_malloc (sizeof (float) * N);
    auto* work_data = (float*) chowdsp::fft::aligned_malloc (sizeof (float) * N);
    auto* work_data_ref = (float*) pffft_aligned_malloc (sizeof (float) * N);

    for (int i = 0; i < N; ++i)
    {
        sine1[i] = std::sin (3.14f * (100.0f / 48000.0f) * (float) i);
        sine2[i] = std::sin (3.14f * (200.0f / 48000.0f) * (float) i);
    }
    std::copy (sine1, sine1 + N, sine1_ref);
    std::copy (sine2, sine2 + N, sine2_ref);
    std::fill_n (out, N, 0.0f);
    std::fill_n (out_ref, N, 0.0f);
    const auto norm_gain = 1.0f / static_cast<float> (N);

    auto* pffft_setup = pffft_new_setup (N, PFFFT_REAL);
    pffft_transform (pffft_setup, sine1_ref, sine1_ref, work_data_ref, PFFFT_FORWARD);
    pffft_transform (pffft_setup, sine2_ref, sine2_ref, work_data_ref, PFFFT_FORWARD);
    pffft_zconvolve_accumulate (pffft_setup, sine1_ref, sine2_ref, out_ref, norm_gain);
    pffft_transform (pffft_setup, out_ref, out_ref, work_data_ref, PFFFT_BACKWARD);
    for (int i = 0; i < N; ++i)
        out_ref[i] += sine1_ref[i];

    auto* fft_setup = chowdsp::fft::fft_new_setup (N, chowdsp::fft::FFT_REAL, use_avx);
    REQUIRE (fft_setup != nullptr);
    chowdsp::fft::fft_transform_unordered (fft_setup, sine1, sine1, work_data, chowdsp::fft::FFT_FORWARD);
    chowdsp::fft::fft_transform_unordered (fft_setup, sine2, sine2, work_data, chowdsp::fft::FFT_FORWARD);
    chowdsp::fft::fft_convolve_unordered (fft_setup, sine1, sine2, out, norm_gain);
    chowdsp::fft::fft_transform_unordered (fft_setup, out, out, work_data, chowdsp::fft::FFT_BACKWARD);
    chowdsp::fft::fft_accumulate (fft_setup, out, sine1, out, N);

    compare (out_ref, out, N);

    chowdsp::fft::fft_destroy_setup (fft_setup);
    pffft_destroy_setup (pffft_setup);
    chowdsp::fft::aligned_free (sine1);
    pffft_aligned_free (sine1_ref);
    chowdsp::fft::aligned_free (sine2);
    pffft_aligned_free (sine2_ref);
    chowdsp::fft::aligned_free (out);
    pffft_aligned_free (out_ref);
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
            test_fft_complex (fft_size);
        }

        SECTION ("Testing Real FFT with size: " + std::to_string (fft_size))
        {
            test_fft_real (fft_size);
        }

        SECTION ("Testing Complex Convolution with size: " + std::to_string (fft_size))
        {
            test_convolution_complex (fft_size);
        }

        SECTION ("Testing Real Convolution with size: " + std::to_string (fft_size))
        {
            test_convolution_real (fft_size);
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
            test_fft_complex (fft_size, true);
        }

        SECTION ("Testing Real FFT with size: " + std::to_string (fft_size))
        {
            test_fft_real (fft_size, true);
        }

        SECTION ("Testing Complex Convolution with size: " + std::to_string (fft_size))
        {
            test_convolution_complex (fft_size, true);
        }

        SECTION ("Testing Real Convolution with size: " + std::to_string (fft_size))
        {
            test_convolution_real (fft_size, true);
        }
    }
}
#endif
