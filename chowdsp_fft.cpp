/**
Copyright (c) 2024 Jatin Chowdhury ( jatin@chowdsp.com )
Copyright (c) 2013  Julien Pommier ( pommier@modartt.com )

Based on original fortran 77 code from FFTPACKv4 from NETLIB,
authored by Dr Paul Swarztrauber of NCAR, in 1985.

As confirmed by the NCAR fftpack software curators, the following
FFTPACKv5 license applies to FFTPACKv4 sources. My changes are
released under the same terms.

FFTPACK license:
http://www.cisl.ucar.edu/css/software/fftpack5/ftpk.html

Copyright (c) 2004 the University Corporation for Atmospheric
Research ("UCAR"). All rights reserved. Developed by NCAR's
Computational and Information Systems Laboratory, UCAR,
www.cisl.ucar.edu.

Redistribution and use of the Software in source and binary forms,
with or without modification, is permitted provided that the
following conditions are met:

- Neither the names of NCAR's Computational and Information Systems
  Laboratory, the University Corporation for Atmospheric Research,
  nor the names of its sponsors or contributors may be used to
  endorse or promote products derived from this Software without
  specific prior written permission.

- Redistributions of source code must retain the above copyright
  notices, this list of conditions, and the disclaimer below.

- Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions, and the disclaimer below in the
  documentation and/or other materials provided with the
  distribution.

THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING, BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES OR OTHER LIABILITY, WHETHER IN AN
ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE
SOFTWARE.
*/

#include "chowdsp_fft.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>

#if defined(_MSC_VER)
// Contains the definition of __cpuidex
#include <intrin.h>
#endif

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

#include "simd/chowdsp_fft_impl_common.hpp"

#if defined(__SSE2__) || defined(_M_AMD64) || defined(_M_X64)
#include "simd/chowdsp_fft_impl_sse.cpp"
#if CHOWDSP_FFT_COMPILER_SUPPORTS_AVX
namespace chowdsp::fft::avx
{
struct FFT_Setup;
FFT_Setup* fft_new_setup (int N, fft_transform_t transform);
void fft_destroy_setup (FFT_Setup* s);
void pffft_transform_internal (FFT_Setup* setup, const float* finput, float* foutput, void* scratch, fft_direction_t direction, int ordered);
void pffft_convolve_internal (FFT_Setup* setup, const float* a, const float* b, float* ab, float scaling);
void fft_accumulate_internal (const float* a, const float* b, float* ab, int N);
} // namespace chowdsp::fft::avx
static constexpr uintptr_t address_mask = ~static_cast<uintptr_t> (3);
static constexpr uintptr_t typeid_mask = static_cast<uintptr_t> (3);
#endif
#elif defined(__ARM_NEON__) || defined(_M_ARM64)
#include "simd/chowdsp_fft_impl_neon.cpp"
#endif

namespace chowdsp::fft
{
#if (defined(__SSE2__) || defined(_M_AMD64) || defined(_M_X64)) && CHOWDSP_FFT_COMPILER_SUPPORTS_AVX
// borrowed from XSIMD: https://github.com/xtensor-stack/xsimd/blob/master/include/xsimd/config/xsimd_cpuid.hpp#L124
static bool cpu_supports_avx()
{
    auto get_xcr0_low = []() noexcept
    {
        uint32_t xcr0;
#if defined(_MSC_VER) && _MSC_VER >= 1400
        xcr0 = (uint32_t) _xgetbv (0);
#elif defined(__GNUC__)
        __asm__ (
            "xorl %%ecx, %%ecx\n"
            "xgetbv\n"
            : "=a"(xcr0)
            :
#if defined(__i386__)
            : "ecx", "edx"
#else
            : "rcx", "rdx"
#endif
        );
#else /* _MSC_VER < 1400 */
#error "_MSC_VER < 1400 is not supported"
#endif /* _MSC_VER && _MSC_VER >= 1400 */
        return xcr0;
    };

    auto get_cpuid = [] (int reg[4], int level, int count = 0) noexcept
    {
#if defined(_MSC_VER)
        __cpuidex (reg, level, count);
#elif defined(__INTEL_COMPILER)
        __cpuid (reg, level);
#elif defined(__GNUC__) || defined(__clang__)
#if defined(__i386__) && defined(__PIC__)
        // %ebx may be the PIC register
        __asm__ ("xchg{l}\t{%%}ebx, %1\n\t"
                 "cpuid\n\t"
                 "xchg{l}\t{%%}ebx, %1\n\t"
                 : "=a"(reg[0]), "=r"(reg[1]), "=c"(reg[2]), "=d"(reg[3])
                 : "0"(level), "2"(count));

#else
        __asm__ ("cpuid\n\t"
                 : "=a"(reg[0]), "=b"(reg[1]), "=c"(reg[2]), "=d"(reg[3])
                 : "0"(level), "2"(count));
#endif
#else
#error "Unsupported configuration"
#endif
    };

    int regs1[4];

    get_cpuid (regs1, 0x1);

    // OS can explicitly disable the usage of SSE/AVX extensions
    // by setting an appropriate flag in CR0 register
    //
    // https://docs.kernel.org/admin-guide/hw-vuln/gather_data_sampling.html

    unsigned sse_state_os_enabled = 1;
    unsigned avx_state_os_enabled = 1;

    // OSXSAVE: A value of 1 indicates that the OS has set CR4.OSXSAVE[bit
    // 18] to enable XSETBV/XGETBV instructions to access XCR0 and
    // to support processor extended state management using
    // XSAVE/XRSTOR.
    bool osxsave = regs1[2] >> 27 & 1;
    if (osxsave)
    {
        uint32_t xcr0 = get_xcr0_low();

        sse_state_os_enabled = xcr0 >> 1 & 1;
        avx_state_os_enabled = xcr0 >> 2 & sse_state_os_enabled;
    }

    [[maybe_unused]] const auto sse2 = regs1[3] >> 26 & sse_state_os_enabled;
    [[maybe_unused]] const auto sse3 = regs1[2] >> 0 & sse_state_os_enabled;
    [[maybe_unused]] const auto ssse3 = regs1[2] >> 9 & sse_state_os_enabled;
    [[maybe_unused]] const auto sse4_1 = regs1[2] >> 19 & sse_state_os_enabled;
    [[maybe_unused]] const auto sse4_2 = regs1[2] >> 20 & sse_state_os_enabled;
    const auto fma3_sse42 = regs1[2] >> 12 & sse_state_os_enabled;

    const auto avx = regs1[2] >> 28 & avx_state_os_enabled;
    [[maybe_unused]] const auto fma3_avx = avx && fma3_sse42;

    int regs8[4];
    get_cpuid (regs8, 0x80000001);
    [[maybe_unused]] const auto fma4 = regs8[2] >> 16 & avx_state_os_enabled;

    // sse4a = regs[2] >> 6 & 1;

    // xop = regs[2] >> 11 & 1;

    int regs7[4];
    get_cpuid (regs7, 0x7);
    const auto avx2 = regs7[1] >> 5 & avx_state_os_enabled;

    int regs7a[4];
    get_cpuid (regs7a, 0x7, 0x1);
    [[maybe_unused]] const auto avxvnni = regs7a[0] >> 4 & avx_state_os_enabled;

    const auto fma3_avx2 = avx2 && fma3_sse42;

    return fma3_avx2;
}

void set_pointer_is_sse_setup (void*& ptr)
{
    // Sets the first bit of the pointer to 1 to mark that this is a pointer to an SSE setup
    ptr = reinterpret_cast<void*> ((reinterpret_cast<uintptr_t> (ptr) & address_mask)
                                   | (static_cast<uintptr_t> (1) & typeid_mask));
}

void* get_setup_pointer (void* ptr)
{
    return reinterpret_cast<void*> (reinterpret_cast<uintptr_t> (ptr) & address_mask);
}

bool check_is_pointer_sse_setup (void* ptr)
{
    // If the first bit of the pointer is 1, then this is an SSE setup
    return (reinterpret_cast<uintptr_t> (ptr) & typeid_mask) == 1;
}
#endif

void* fft_new_setup (int N, fft_transform_t transform, [[maybe_unused]] bool use_avx_if_available)
{
#if defined(__SSE2__) || defined(_M_AMD64) || defined(_M_X64)
#if CHOWDSP_FFT_COMPILER_SUPPORTS_AVX
    if (use_avx_if_available)
    {
        if (cpu_supports_avx())
        {
            auto* setup_ptr = avx::fft_new_setup (N, transform);
            if (setup_ptr != nullptr)
                return setup_ptr;
        }
    }
    void* ptr = sse::fft_new_setup (N, transform);
    set_pointer_is_sse_setup (ptr);
    return ptr;
#else
    return sse::fft_new_setup (N, transform);
#endif
#elif defined(__ARM_NEON__) || defined(_M_ARM64)
    return neon::fft_new_setup (N, transform);
#endif
}

void fft_destroy_setup (void* ptr)
{
#if defined(__SSE2__) || defined(_M_AMD64) || defined(_M_X64)
#if CHOWDSP_FFT_COMPILER_SUPPORTS_AVX
    if (check_is_pointer_sse_setup (ptr))
        sse::fft_destroy_setup (reinterpret_cast<sse::FFT_Setup*> (get_setup_pointer (ptr)));
    else
        avx::fft_destroy_setup (reinterpret_cast<avx::FFT_Setup*> (get_setup_pointer (ptr)));
#else
    sse::fft_destroy_setup (reinterpret_cast<sse::FFT_Setup*> (ptr));
#endif
#elif defined(__ARM_NEON__) || defined(_M_ARM64)
    neon::fft_destroy_setup (reinterpret_cast<neon::FFT_Setup*> (ptr));
#endif
}

int fft_simd_width_bytes (void* setup)
{
#if defined(__SSE2__) || defined(_M_AMD64) || defined(_M_X64)
#if CHOWDSP_FFT_COMPILER_SUPPORTS_AVX
    if (check_is_pointer_sse_setup (setup))
    {
        return 16;
    }
    else
    {
        return 32;
    }
#else
    return 16;
#endif
#elif defined(__ARM_NEON__) || defined(_M_ARM64)
    return 16;
#endif
}

void fft_transform (void* setup, const float* input, float* output, float* work, fft_direction_t direction)
{
#if defined(__SSE2__) || defined(_M_AMD64) || defined(_M_X64)
#if CHOWDSP_FFT_COMPILER_SUPPORTS_AVX
    if (check_is_pointer_sse_setup (setup))
    {
        sse::pffft_transform_internal (reinterpret_cast<sse::FFT_Setup*> (get_setup_pointer (setup)),
                                       input,
                                       output,
                                       (__m128*) work,
                                       direction,
                                       1);
    }
    else
    {
        avx::pffft_transform_internal (reinterpret_cast<avx::FFT_Setup*> (get_setup_pointer (setup)),
                                       input,
                                       output,
                                       work,
                                       direction,
                                       1);
    }
#else
    sse::pffft_transform_internal (reinterpret_cast<sse::FFT_Setup*> (setup),
                                   input,
                                   output,
                                   (__m128*) work,
                                   direction,
                                   1);
#endif
#elif defined(__ARM_NEON__) || defined(_M_ARM64)
    neon::pffft_transform_internal (reinterpret_cast<neon::FFT_Setup*> (setup),
                                    input,
                                    output,
                                    (float32x4_t*) work,
                                    direction,
                                    1);
#endif
}

void fft_transform_unordered (void* setup, const float* input, float* output, float* work, fft_direction_t direction)
{
#if defined(__SSE2__) || defined(_M_AMD64) || defined(_M_X64)
#if CHOWDSP_FFT_COMPILER_SUPPORTS_AVX
    if (check_is_pointer_sse_setup (setup))
    {
        sse::pffft_transform_internal (reinterpret_cast<sse::FFT_Setup*> (get_setup_pointer (setup)),
                                       input,
                                       output,
                                       (__m128*) work,
                                       direction,
                                       0);
    }
    else
    {
        avx::pffft_transform_internal (reinterpret_cast<avx::FFT_Setup*> (get_setup_pointer (setup)),
                                       input,
                                       output,
                                       work,
                                       direction,
                                       0);
    }
#else
    sse::pffft_transform_internal (reinterpret_cast<sse::FFT_Setup*> (setup),
                                   input,
                                   output,
                                   (__m128*) work,
                                   direction,
                                   0);
#endif
#elif defined(__ARM_NEON__) || defined(_M_ARM64)
    neon::pffft_transform_internal (reinterpret_cast<neon::FFT_Setup*> (setup),
                                    input,
                                    output,
                                    (float32x4_t*) work,
                                    direction,
                                    0);
#endif
}

void fft_convolve_unordered (void* setup, const float* a, const float* b, float* ab, float scaling)
{
#if defined(__SSE2__) || defined(_M_AMD64) || defined(_M_X64)
#if CHOWDSP_FFT_COMPILER_SUPPORTS_AVX
    if (check_is_pointer_sse_setup (setup))
    {
        sse::pffft_convolve_internal (reinterpret_cast<sse::FFT_Setup*> (get_setup_pointer (setup)),
                                      a,
                                      b,
                                      ab,
                                      scaling);
    }
    else
    {
        avx::pffft_convolve_internal (reinterpret_cast<avx::FFT_Setup*> (get_setup_pointer (setup)),
                                      a,
                                      b,
                                      ab,
                                      scaling);
    }
#else
    sse::pffft_convolve_internal (reinterpret_cast<sse::FFT_Setup*> (setup),
                                  a,
                                  b,
                                  ab,
                                  scaling);
#endif
#elif defined(__ARM_NEON__) || defined(_M_ARM64)
    neon::pffft_convolve_internal (reinterpret_cast<neon::FFT_Setup*> (setup),
                                   a,
                                   b,
                                   ab,
                                   scaling);
#endif
}

void fft_accumulate (void* setup, const float* a, const float* b, float* ab, int N)
{
#if defined(__SSE2__) || defined(_M_AMD64) || defined(_M_X64)
#if CHOWDSP_FFT_COMPILER_SUPPORTS_AVX
    if (check_is_pointer_sse_setup (setup))
    {
        sse::fft_accumulate_internal (a, b, ab, N);
    }
    else
    {
        avx::fft_accumulate_internal (a, b, ab, N);
    }
#else
    sse::fft_accumulate_internal (a, b, ab, N);
#endif
#elif defined(__ARM_NEON__) || defined(_M_ARM64)
    neon::fft_accumulate_internal (a, b, ab, N);
#endif
}
} // namespace chowdsp::fft
