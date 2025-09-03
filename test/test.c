#include <chowdsp_fft.h>
#include <pffft.h>

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

void compare (const float* ref, const float* test, int N)
{
    const float tol = 1.0e-6f * (float) N / 8.0f;
    for (int n = 0; n < N; ++n)
        assert (fabsf (ref[n] - test[n]) < tol);
}

void test_complex (int N, bool use_avx, bool preallocate)
{
    float* data = (float*) aligned_malloc (sizeof (float) * N * 2);
    float* data_ref = (float*) pffft_aligned_malloc (sizeof (float) * N * 2);
    float* work_data = (float*) aligned_malloc (sizeof (float) * N * 2);
    float* work_data_ref = (float*) pffft_aligned_malloc (sizeof (float) * N * 2);

    for (int i = 0; i < N; ++i)
    {
        data[i * 2] = sinf (3.14f * (100.0f / 48000.0f) * (float) i);
        data[i * 2 + 1] = cosf (3.14f * (100.0f / 48000.0f) * (float) i);
    }
    memcpy (data_ref, data, N * 2 * sizeof (float));

    void* fft_setup;
    void* prealloc;
    if (preallocate)
    {
        size_t bytes_required = fft_bytes_required (N, FFT_COMPLEX, use_avx);
        prealloc = aligned_malloc (bytes_required);
        fft_setup = fft_new_setup_preallocated (N, FFT_COMPLEX, prealloc, use_avx);
    }
    else
    {

        fft_setup = fft_new_setup (N, FFT_COMPLEX, use_avx);
    }

    assert (fft_setup != NULL);
    PFFFT_Setup* pffft_setup = pffft_new_setup (N, PFFFT_COMPLEX);

    fft_transform (fft_setup, data, data, work_data, FFT_FORWARD);
    pffft_transform_ordered (pffft_setup, data_ref, data_ref, work_data_ref, PFFFT_FORWARD);

    compare (data_ref, data, N * 2);

    fft_transform (fft_setup, data, data, work_data, FFT_BACKWARD);
    pffft_transform_ordered (pffft_setup, data_ref, data_ref, work_data_ref, PFFFT_BACKWARD);

    const float norm_gain = 1.0f / (float) N;
    for (int n = 0; n < N * 2; ++n)
    {
        data[n] *= norm_gain;
        data_ref[n] *= norm_gain;
    }

    compare (data_ref, data, N * 2);

    if (preallocate)
        aligned_free (prealloc);
    else
        fft_destroy_setup (fft_setup);
    pffft_destroy_setup (pffft_setup);
    aligned_free (data);
    pffft_aligned_free (data_ref);
    aligned_free (work_data);
    pffft_aligned_free (work_data_ref);
}

void test_real (int N, bool use_avx, bool preallocate)
{
    float* data = (float*) aligned_malloc (sizeof (float) * N);
    float* data_ref = (float*) pffft_aligned_malloc (sizeof (float) * N);
    float* work_data = (float*) aligned_malloc (sizeof (float) * N);
    float* work_data_ref = (float*) pffft_aligned_malloc (sizeof (float) * N);

    for (int i = 0; i < N; ++i)
    {
        data[i] = sinf (3.14f * (100.0f / 48000.0f) * (float) i);
    }
    memcpy (data_ref, data, N * sizeof (float));

    void* fft_setup;
    void* prealloc;
    if (preallocate)
    {
        size_t bytes_required = fft_bytes_required (N, FFT_REAL, use_avx);
        prealloc = aligned_malloc (bytes_required);
        fft_setup = fft_new_setup_preallocated (N, FFT_REAL, prealloc, use_avx);
    }
    else
    {

        fft_setup = fft_new_setup (N, FFT_REAL, use_avx);
    }

    assert (fft_setup != NULL);
    PFFFT_Setup* pffft_setup = pffft_new_setup (N, PFFFT_REAL);

    fft_transform (fft_setup, data, data, work_data, FFT_FORWARD);
    pffft_transform_ordered (pffft_setup, data_ref, data_ref, work_data_ref, PFFFT_FORWARD);

    compare (data_ref, data, N);

    fft_transform (fft_setup, data, data, work_data, FFT_BACKWARD);
    pffft_transform_ordered (pffft_setup, data_ref, data_ref, work_data_ref, PFFFT_BACKWARD);

    const float norm_gain = 1.0f / (float) N;
    for (int n = 0; n < N; ++n)
    {
        data[n] *= norm_gain;
        data_ref[n] *= norm_gain;
    }

    compare (data_ref, data, N);

    if (preallocate)
        aligned_free (prealloc);
    else
        fft_destroy_setup (fft_setup);
    pffft_destroy_setup (pffft_setup);
    aligned_free (data);
    pffft_aligned_free (data_ref);
    aligned_free (work_data);
    pffft_aligned_free (work_data_ref);
}


int main()
{
    printf ("Running SSE/NEON Tests\n");
    for (int i = 5; i < 20; ++i)
    {
        const int fft_size = 1 << i;
        printf ("Testing Complex FFT with size: %d\n", fft_size);
        test_complex (fft_size, false, false);
        printf ("Testing Real FFT with size: %d\n", fft_size);
        test_real (fft_size, false, false);
    }

#if defined(__SSE2__)
    printf ("Running AVX Tests\n");
    for (int i = 5; i < 20; ++i)
    {
        const int fft_size = 1 << i;
        printf ("Testing Complex FFT with size: %d\n", fft_size);
        test_complex (fft_size, true, false);
        printf ("Testing Real FFT with size: %d\n", fft_size);
        test_real (fft_size, true, false);
    }
#endif

printf ("Running pre-allocated Tests\n");
for (int i = 5; i < 20; ++i)
{
    const int fft_size = 1 << i;
    printf ("Testing pre-allocated Complex FFT with size: %d\n", fft_size);
    test_complex (fft_size, false, true);
    printf ("Testing pre-allocated Real FFT with size: %d\n", fft_size);
    test_real (fft_size, false, true);
}

    printf ("Testing complete!\n");

    return 0;
}
