#include "chowdsp_fft.h"

#include <cassert>
#include <cmath>
#include <cstdlib>

namespace chowdsp::fft
{
static constexpr size_t MALLOC_V4SF_ALIGNMENT = 64; // with a 64-byte alignment, we are even aligned on L2 cache lines...
void* aligned_malloc (size_t nb_bytes)
{
    void *p, *p0 = malloc (nb_bytes + MALLOC_V4SF_ALIGNMENT);
    if (! p0)
        return nullptr;
    p = (void*) (((size_t) p0 + MALLOC_V4SF_ALIGNMENT) & (~((size_t) (MALLOC_V4SF_ALIGNMENT - 1))));
    *((void**) p - 1) = p0;
    return p;
}

void aligned_free (void* p)
{
    if (p)
        free (*((void**) p - 1));
}
} // namespace chowdsp::fft

#include "simd/chowdsp_fft_impl_common.cpp"
#if defined(__AVX2__)
// TODO
#elif defined(__SSE2__)
#include "simd/chowdsp_fft_impl_sse.cpp"
#elif defined(__ARM_NEON__)
#include "simd/chowdsp_fft_impl_neon.cpp"
#endif

namespace chowdsp::fft
{
void* fft_new_setup (int N, fft_transform_t transform)
{
#if defined(__AVX2__)
    // TODO
#elif defined(__SSE2__)
    return sse::fft_new_setup (N, transform);
#elif defined(__ARM_NEON__)
    return neon::fft_new_setup (N, transform);
#endif
}

void fft_destroy_setup (void* ptr)
{
#if defined(__AVX2__)
    // TODO
#elif defined(__SSE2__)
    return sse::fft_destroy_setup (reinterpret_cast<sse::FFT_Setup*> (ptr));
#elif defined(__ARM_NEON__)
    return neon::fft_destroy_setup (reinterpret_cast<neon::FFT_Setup*> (ptr));
#endif
}

void fft_transform (void* setup, const float* input, float* output, float* work, fft_direction_t direction)
{
#if defined(__AVX2__)
    // TODO
#elif defined(__SSE2__)
    return sse::pffft_transform_internal (reinterpret_cast<sse::FFT_Setup*> (setup),
                                          input,
                                          output,
                                          (__m128*) work,
                                          direction,
                                          1);
#elif defined(__ARM_NEON__)
    return neon::pffft_transform_internal (reinterpret_cast<neon::FFT_Setup*> (setup),
                                           input,
                                           output,
                                           (float32x4_t*) work,
                                           direction,
                                           1);
#endif
}
} // namespace chowdsp::fft
