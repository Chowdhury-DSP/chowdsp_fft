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

#include <arm_neon.h>
#include <tuple>

#ifdef _MSC_VER
#include <malloc.h>  // For alloca
#endif

namespace chowdsp::fft::neon
{
static constexpr size_t SIMD_SZ = 4;

struct FFT_Setup
{
    int N;
    int Ncvec; // nb of complex simd vectors (N/4 if PFFFT_COMPLEX, N/8 if PFFFT_REAL)
    int ifac[15];
    fft_transform_t transform;
    float32x4_t* data; // allocated room for twiddle coefs
    float* e; // points into 'data' , N/4*3 elements
    float* twiddle; // points into 'data', N/4 elements
};

static size_t fft_bytes_required (int N, fft_transform_t transform)
{
    const auto Ncvec = (transform == FFT_REAL ? N / 2 : N) / SIMD_SZ;
    const auto data_bytes = 2 * Ncvec * sizeof (float) * SIMD_SZ;
    return data_bytes + sizeof (FFT_Setup);
}

static FFT_Setup* fft_new_setup (int N, fft_transform_t transform, void* data)
{
    const auto Ncvec = (transform == FFT_REAL ? N / 2 : N) / SIMD_SZ;
    const auto data_bytes = 2 * Ncvec * sizeof (float) * SIMD_SZ;
    auto* s_data = (std::byte*) data;

    auto* s = (FFT_Setup*) (s_data + data_bytes);
    /* unfortunately, the fft size must be a multiple of 16 for complex FFTs
       and 32 for real FFTs -- a lot of stuff would need to be rewritten to
       handle other cases (or maybe just switch to a scalar fft, I don't know..) */
    if (transform == FFT_REAL)
    {
        assert ((N % (2 * SIMD_SZ * SIMD_SZ)) == 0 && N > 0);
    }
    if (transform == FFT_COMPLEX)
    {
        assert ((N % (SIMD_SZ * SIMD_SZ)) == 0 && N > 0);
    }
    //assert((N % 32) == 0);
    s->N = N;
    s->transform = transform;
    /* nb of complex simd vectors */
    s->Ncvec = Ncvec;
    s->data = (float32x4_t*) s_data;
    s->e = (float*) s->data;
    s->twiddle = (float*) (s->data + (2 * s->Ncvec * (SIMD_SZ - 1)) / SIMD_SZ);

    int k, m;
    for (k = 0; k < s->Ncvec; ++k)
    {
        int i = k / (int) SIMD_SZ;
        int j = k % (int) SIMD_SZ;
        for (m = 0; m < SIMD_SZ - 1; ++m)
        {
            const auto A = static_cast<float> (-2 * M_PI * (m + 1) * k / N);
            s->e[(2 * (i * 3 + m) + 0) * SIMD_SZ + j] = std::cos (A);
            s->e[(2 * (i * 3 + m) + 1) * SIMD_SZ + j] = std::sin (A);
        }
    }

    if (transform == FFT_REAL)
    {
        common::rffti1_ps (N / (int) SIMD_SZ, s->twiddle, s->ifac);
    }
    else
    {
        common::cffti1_ps (N / (int) SIMD_SZ, s->twiddle, s->ifac);
    }

    /* check that N is decomposable with allowed prime factors */
    for (k = 0, m = 1; k < s->ifac[1]; ++k)
    {
        m *= s->ifac[2 + k];
    }
    if (m != N / SIMD_SZ)
    {
        fft_destroy_setup (s);
        s = nullptr;
    }

    return s;
}

static void fft_destroy_setup (FFT_Setup* s)
{
    aligned_free (s->data);
}

//====================================================================
static inline auto interleave2 (float32x4_t in1, float32x4_t in2)
{
    const auto tmp = vzipq_f32 (in1, in2);
    return std::make_tuple (tmp.val[0], tmp.val[1]);
}

static inline auto uninterleave2 (float32x4_t in1, float32x4_t in2)
{
    const auto tmp = vuzpq_f32 (in1, in2);
    return std::make_tuple (tmp.val[0], tmp.val[1]);
}

static inline auto cplx_mul (float32x4_t ar, float32x4_t ai, float br, float bi)
{
    auto tmp = vmulq_n_f32 (ar, bi);
    ar = vmulq_n_f32 (ar, br);
    ar = vfmsq_n_f32 (ar, ai, bi);
    ai = vfmaq_n_f32 (tmp, ai, br);
    return std::make_tuple (ar, ai);
}

static inline auto cplx_mul_conj (float32x4_t ar, float32x4_t ai, float br, float bi)
{
    auto tmp = vmulq_n_f32 (ar, -bi);
    ar = vmulq_n_f32 (ar, br);
    ar = vfmaq_n_f32 (ar, ai, bi);
    ai = vfmaq_n_f32 (tmp, ai, br);
    return std::make_tuple (ar, ai);
}

static inline auto cplx_mul_v (float32x4_t ar, float32x4_t ai, float32x4_t br, float32x4_t bi)
{
    auto tmp = vmulq_f32 (ar, bi);
    ar = vmulq_f32 (ar, br);
    ar = vfmsq_f32 (ar, ai, bi);
    ai = vfmaq_f32 (tmp, ai, br);
    return std::make_tuple (ar, ai);
}

static inline auto cplx_mul_conj_v (float32x4_t ar, float32x4_t ai, float32x4_t br, float32x4_t bi)
{
    auto tmp = vmulq_f32 (ar, vnegq_f32(bi));
    ar = vmulq_f32 (ar, br);
    ar = vfmaq_f32 (ar, ai, bi);
    ai = vfmaq_f32 (tmp, ai, br);
    return std::make_tuple (ar, ai);
}

static inline void transpose4 (float32x4_t& x0, float32x4_t& x1, float32x4_t& x2, float32x4_t& x3)
{
    float32x4x2_t t0_ = vzipq_f32 (x0, x2);
    float32x4x2_t t1_ = vzipq_f32 (x1, x3);
    float32x4x2_t u0_ = vzipq_f32 (t0_.val[0], t1_.val[0]);
    float32x4x2_t u1_ = vzipq_f32 (t0_.val[1], t1_.val[1]);
    x0 = u0_.val[0];
    x1 = u0_.val[1];
    x2 = u1_.val[0];
    x3 = u1_.val[1];
}

//====================================================================
static void passf2_ps (int ido, int l1, const float32x4_t* cc, float32x4_t* ch, const float* wa1, float fsign)
{
    int k, i;
    int l1ido = l1 * ido;
    if (ido <= 2)
    {
        for (k = 0; k < l1ido; k += ido, ch += ido, cc += 2 * ido)
        {
            ch[0] = vaddq_f32 (cc[0], cc[ido + 0]);
            ch[l1ido] = vsubq_f32 (cc[0], cc[ido + 0]);
            ch[1] = vaddq_f32 (cc[1], cc[ido + 1]);
            ch[l1ido + 1] = vsubq_f32 (cc[1], cc[ido + 1]);
        }
    }
    else
    {
        for (k = 0; k < l1ido; k += ido, ch += ido, cc += 2 * ido)
        {
            for (i = 0; i < ido - 1; i += 2)
            {
                auto tr2 = vsubq_f32 (cc[i + 0], cc[i + ido + 0]);
                auto ti2 = vsubq_f32 (cc[i + 1], cc[i + ido + 1]);
                auto wr = wa1[i];
                auto wi = fsign * wa1[i + 1]; // @OPTIMIZE: fsign will always be +/- 1
                ch[i] = vaddq_f32 (cc[i + 0], cc[i + ido + 0]);
                ch[i + 1] = vaddq_f32 (cc[i + 1], cc[i + ido + 1]);
                std::tie (tr2, ti2) = cplx_mul (tr2, ti2, wr, wi);
                ch[i + l1ido] = tr2;
                ch[i + l1ido + 1] = ti2;
            }
        }
    }
}

static void passf3_ps (int ido, int l1, const float32x4_t* cc, float32x4_t* ch, const float* wa1, const float* wa2, float fsign)
{
    static constexpr float taur = -0.5f;
    float taui = 0.866025403784439f * fsign;
    int i, k;
    float32x4_t tr2, ti2, cr2, ci2, cr3, ci3, dr2, di2, dr3, di3;
    int l1ido = l1 * ido;
    float wr1, wi1, wr2, wi2;
    assert (ido > 2);
    for (k = 0; k < l1ido; k += ido, cc += 3 * ido, ch += ido)
    {
        for (i = 0; i < ido - 1; i += 2)
        {
            tr2 = vaddq_f32 (cc[i + ido], cc[i + 2 * ido]);
            cr2 = vfmaq_n_f32 (cc[i], tr2, taur);
            ch[i] = vaddq_f32 (cc[i], tr2);
            ti2 = vaddq_f32 (cc[i + ido + 1], cc[i + 2 * ido + 1]);
            ci2 = vfmaq_n_f32 (cc[i + 1], ti2, taur);
            ch[i + 1] = vaddq_f32 (cc[i + 1], ti2);
            cr3 = vmulq_n_f32 (vsubq_f32 (cc[i + ido], cc[i + 2 * ido]), taui);
            ci3 = vmulq_n_f32 (vsubq_f32 (cc[i + ido + 1], cc[i + 2 * ido + 1]), taui);
            dr2 = vsubq_f32 (cr2, ci3);
            dr3 = vaddq_f32 (cr2, ci3);
            di2 = vaddq_f32 (ci2, cr3);
            di3 = vsubq_f32 (ci2, cr3);
            wr1 = wa1[i];
            wi1 = fsign * wa1[i + 1];
            wr2 = wa2[i];
            wi2 = fsign * wa2[i + 1];
            std::tie (dr2, di2) = cplx_mul (dr2, di2, wr1, wi1);
            ch[i + l1ido] = dr2;
            ch[i + l1ido + 1] = di2;
            std::tie (dr3, di3) = cplx_mul (dr3, di3, wr2, wi2);
            ch[i + 2 * l1ido] = dr3;
            ch[i + 2 * l1ido + 1] = di3;
        }
    }
}

static void passf4_ps (int ido, int l1, const float32x4_t* cc, float32x4_t* ch, const float* wa1, const float* wa2, const float* wa3, float fsign)
{
    /* isign == -1 for forward transform and +1 for backward transform */

    int i, k;
    float32x4_t ci2, ci3, ci4, cr2, cr3, cr4, ti1, ti2, ti3, ti4, tr1, tr2, tr3, tr4;
    int l1ido = l1 * ido;
    if (ido == 2)
    {
        for (k = 0; k < l1ido; k += ido, ch += ido, cc += 4 * ido)
        {
            tr1 = vsubq_f32 (cc[0], cc[2 * ido + 0]);
            tr2 = vaddq_f32 (cc[0], cc[2 * ido + 0]);
            ti1 = vsubq_f32 (cc[1], cc[2 * ido + 1]);
            ti2 = vaddq_f32 (cc[1], cc[2 * ido + 1]);
            ti4 = vmulq_n_f32 (vsubq_f32 (cc[1 * ido + 0], cc[3 * ido + 0]), fsign);
            tr4 = vmulq_n_f32 (vsubq_f32 (cc[3 * ido + 1], cc[1 * ido + 1]), fsign);
            tr3 = vaddq_f32 (cc[ido + 0], cc[3 * ido + 0]);
            ti3 = vaddq_f32 (cc[ido + 1], cc[3 * ido + 1]);

            ch[0 * l1ido + 0] = vaddq_f32 (tr2, tr3);
            ch[0 * l1ido + 1] = vaddq_f32 (ti2, ti3);
            ch[1 * l1ido + 0] = vaddq_f32 (tr1, tr4);
            ch[1 * l1ido + 1] = vaddq_f32 (ti1, ti4);
            ch[2 * l1ido + 0] = vsubq_f32 (tr2, tr3);
            ch[2 * l1ido + 1] = vsubq_f32 (ti2, ti3);
            ch[3 * l1ido + 0] = vsubq_f32 (tr1, tr4);
            ch[3 * l1ido + 1] = vsubq_f32 (ti1, ti4);
        }
    }
    else
    {
        for (k = 0; k < l1ido; k += ido, ch += ido, cc += 4 * ido)
        {
            for (i = 0; i < ido - 1; i += 2)
            {
                float wr1, wi1, wr2, wi2, wr3, wi3;
                tr1 = vsubq_f32 (cc[i + 0], cc[i + 2 * ido + 0]);
                tr2 = vaddq_f32 (cc[i + 0], cc[i + 2 * ido + 0]);
                ti1 = vsubq_f32 (cc[i + 1], cc[i + 2 * ido + 1]);
                ti2 = vaddq_f32 (cc[i + 1], cc[i + 2 * ido + 1]);
                tr4 = vmulq_n_f32 (vsubq_f32 (cc[i + 3 * ido + 1], cc[i + 1 * ido + 1]), fsign);
                ti4 = vmulq_n_f32 (vsubq_f32 (cc[i + 1 * ido + 0], cc[i + 3 * ido + 0]), fsign);
                tr3 = vaddq_f32 (cc[i + ido + 0], cc[i + 3 * ido + 0]);
                ti3 = vaddq_f32 (cc[i + ido + 1], cc[i + 3 * ido + 1]);

                ch[i] = vaddq_f32 (tr2, tr3);
                cr3 = vsubq_f32 (tr2, tr3);
                ch[i + 1] = vaddq_f32 (ti2, ti3);
                ci3 = vsubq_f32 (ti2, ti3);

                cr2 = vaddq_f32 (tr1, tr4);
                cr4 = vsubq_f32 (tr1, tr4);
                ci2 = vaddq_f32 (ti1, ti4);
                ci4 = vsubq_f32 (ti1, ti4);
                wr1 = wa1[i];
                wi1 = fsign * wa1[i + 1];
                std::tie (cr2, ci2) = cplx_mul (cr2, ci2, wr1, wi1);
                wr2 = wa2[i];
                wi2 = fsign * wa2[i + 1];
                ch[i + l1ido] = cr2;
                ch[i + l1ido + 1] = ci2;

                std::tie (cr3, ci3) = cplx_mul (cr3, ci3, wr2, wi2);
                wr3 = wa3[i];
                wi3 = fsign * wa3[i + 1];
                ch[i + 2 * l1ido] = cr3;
                ch[i + 2 * l1ido + 1] = ci3;

                std::tie (cr4, ci4) = cplx_mul (cr4, ci4, wr3, wi3);
                ch[i + 3 * l1ido] = cr4;
                ch[i + 3 * l1ido + 1] = ci4;
            }
        }
    }
}

static void passf5_ps (int ido, int l1, const float32x4_t* cc, float32x4_t* ch, const float* wa1, const float* wa2, const float* wa3, const float* wa4, float fsign)
{
    static constexpr float tr11 = .309016994374947f;
    const float ti11 = .951056516295154f * fsign;
    static constexpr float tr12 = -.809016994374947f;
    const float ti12 = .587785252292473f * fsign;

    /* Local variables */
    int i, k;
    float32x4_t ci2, ci3, ci4, ci5, di3, di4, di5, di2, cr2, cr3, cr5, cr4, ti2, ti3,
        ti4, ti5, dr3, dr4, dr5, dr2, tr2, tr3, tr4, tr5;

    float wr1, wi1, wr2, wi2, wr3, wi3, wr4, wi4;

#define cc_ref(a_1, a_2) cc[(a_2 - 1) * ido + a_1 + 1]
#define ch_ref(a_1, a_3) ch[(a_3 - 1) * l1 * ido + a_1 + 1]

    assert (ido > 2);
    for (k = 0; k < l1; ++k, cc += 5 * ido, ch += ido)
    {
        for (i = 0; i < ido - 1; i += 2)
        {
            ti5 = vsubq_f32 (cc_ref (i, 2), cc_ref (i, 5));
            ti2 = vaddq_f32 (cc_ref (i, 2), cc_ref (i, 5));
            ti4 = vsubq_f32 (cc_ref (i, 3), cc_ref (i, 4));
            ti3 = vaddq_f32 (cc_ref (i, 3), cc_ref (i, 4));
            tr5 = vsubq_f32 (cc_ref (i - 1, 2), cc_ref (i - 1, 5));
            tr2 = vaddq_f32 (cc_ref (i - 1, 2), cc_ref (i - 1, 5));
            tr4 = vsubq_f32 (cc_ref (i - 1, 3), cc_ref (i - 1, 4));
            tr3 = vaddq_f32 (cc_ref (i - 1, 3), cc_ref (i - 1, 4));
            ch_ref (i - 1, 1) = vaddq_f32 (cc_ref (i - 1, 1), vaddq_f32 (tr2, tr3));
            ch_ref (i, 1) = vaddq_f32 (cc_ref (i, 1), vaddq_f32 (ti2, ti3));
            cr2 = vaddq_f32 (cc_ref (i - 1, 1), vaddq_f32 (vmulq_n_f32 (tr2, tr11), vmulq_n_f32 (tr3, tr12)));
            ci2 = vaddq_f32 (cc_ref (i, 1), vaddq_f32 (vmulq_n_f32 (ti2, tr11), vmulq_n_f32 (ti3, tr12)));
            cr3 = vaddq_f32 (cc_ref (i - 1, 1), vaddq_f32 (vmulq_n_f32 (tr2, tr12), vmulq_n_f32 (tr3, tr11)));
            ci3 = vaddq_f32 (cc_ref (i, 1), vaddq_f32 (vmulq_n_f32 (ti2, tr12), vmulq_n_f32 (ti3, tr11)));
            cr5 = vaddq_f32 (vmulq_n_f32 (tr5, ti11), vmulq_n_f32 (tr4, ti12));
            ci5 = vaddq_f32 (vmulq_n_f32 (ti5, ti11), vmulq_n_f32 (ti4, ti12));
            cr4 = vsubq_f32 (vmulq_n_f32 (tr5, ti12), vmulq_n_f32 (tr4, ti11));
            ci4 = vsubq_f32 (vmulq_n_f32 (ti5, ti12), vmulq_n_f32 (ti4, ti11));
            dr3 = vsubq_f32 (cr3, ci4);
            dr4 = vaddq_f32 (cr3, ci4);
            di3 = vaddq_f32 (ci3, cr4);
            di4 = vsubq_f32 (ci3, cr4);
            dr5 = vaddq_f32 (cr2, ci5);
            dr2 = vsubq_f32 (cr2, ci5);
            di5 = vsubq_f32 (ci2, cr5);
            di2 = vaddq_f32 (ci2, cr5);
            wr1 = wa1[i];
            wi1 = fsign * wa1[i + 1];
            wr2 = wa2[i];
            wi2 = fsign * wa2[i + 1];
            wr3 = wa3[i];
            wi3 = fsign * wa3[i + 1];
            wr4 = wa4[i];
            wi4 = fsign * wa4[i + 1];
            std::tie (dr2, di2) = cplx_mul (dr2, di2, wr1, wi1);
            ch_ref (i - 1, 2) = dr2;
            ch_ref (i, 2) = di2;
            std::tie (dr3, di3) = cplx_mul (dr3, di3, wr2, wi2);
            ch_ref (i - 1, 3) = dr3;
            ch_ref (i, 3) = di3;
            std::tie (dr4, di4) = cplx_mul (dr4, di4, wr3, wi3);
            ch_ref (i - 1, 4) = dr4;
            ch_ref (i, 4) = di4;
            std::tie (dr5, di5) = cplx_mul (dr5, di5, wr4, wi4);
            ch_ref (i - 1, 5) = dr5;
            ch_ref (i, 5) = di5;
        }
    }
#undef ch_ref
#undef cc_ref
}

static float32x4_t* cfftf1_ps (int n, const float32x4_t* input_readonly, float32x4_t* work1, float32x4_t* work2, const float* wa, const int* ifac, int isign)
{
    auto* in = (float32x4_t*) input_readonly;
    auto* out = (in == work2 ? work1 : work2);
    int nf = ifac[1], k1;
    int l1 = 1;
    int iw = 0;
    assert (in != out && work1 != work2);
    for (k1 = 2; k1 <= nf + 1; k1++)
    {
        int ip = ifac[k1];
        int l2 = ip * l1;
        int ido = n / l2;
        int idot = ido + ido;
        switch (ip)
        {
            case 5:
            {
                int ix2 = iw + idot;
                int ix3 = ix2 + idot;
                int ix4 = ix3 + idot;
                passf5_ps (idot, l1, in, out, &wa[iw], &wa[ix2], &wa[ix3], &wa[ix4], isign);
            }
            break;
            case 4:
            {
                int ix2 = iw + idot;
                int ix3 = ix2 + idot;
                passf4_ps (idot, l1, in, out, &wa[iw], &wa[ix2], &wa[ix3], isign);
            }
            break;
            case 3:
            {
                int ix2 = iw + idot;
                passf3_ps (idot, l1, in, out, &wa[iw], &wa[ix2], isign);
            }
            break;
            case 2:
            {
                passf2_ps (idot, l1, in, out, &wa[iw], isign);
            }
            break;
            default:
                assert (0);
        }
        l1 = l2;
        iw += (ip - 1) * idot;
        if (out == work2)
        {
            out = work1;
            in = work2;
        }
        else
        {
            out = work2;
            in = work1;
        }
    }

    return in; /* this is in fact the output .. */
}

static void pffft_cplx_finalize (int Ncvec, const float32x4_t* in, float32x4_t* out, const float32x4_t* e)
{
    int k, dk = Ncvec / (int) SIMD_SZ; // number of 4x4 matrix blocks
    float32x4_t r0, i0, r1, i1, r2, i2, r3, i3;
    float32x4_t sr0, dr0, sr1, dr1, si0, di0, si1, di1;
    assert (in != out);
    for (k = 0; k < dk; ++k)
    {
        r0 = in[8 * k + 0];
        i0 = in[8 * k + 1];
        r1 = in[8 * k + 2];
        i1 = in[8 * k + 3];
        r2 = in[8 * k + 4];
        i2 = in[8 * k + 5];
        r3 = in[8 * k + 6];
        i3 = in[8 * k + 7];
        transpose4 (r0, r1, r2, r3);
        transpose4 (i0, i1, i2, i3);
        std::tie (r1, i1) = cplx_mul_v (r1, i1, e[k * 6 + 0], e[k * 6 + 1]);
        std::tie (r2, i2) = cplx_mul_v (r2, i2, e[k * 6 + 2], e[k * 6 + 3]);
        std::tie (r3, i3) = cplx_mul_v (r3, i3, e[k * 6 + 4], e[k * 6 + 5]);

        sr0 = vaddq_f32 (r0, r2);
        dr0 = vsubq_f32 (r0, r2);
        sr1 = vaddq_f32 (r1, r3);
        dr1 = vsubq_f32 (r1, r3);
        si0 = vaddq_f32 (i0, i2);
        di0 = vsubq_f32 (i0, i2);
        si1 = vaddq_f32 (i1, i3);
        di1 = vsubq_f32 (i1, i3);

        /*
          transformation for each column is:

          [1   1   1   1   0   0   0   0]   [r0]
          [1   0  -1   0   0  -1   0   1]   [r1]
          [1  -1   1  -1   0   0   0   0]   [r2]
          [1   0  -1   0   0   1   0  -1]   [r3]
          [0   0   0   0   1   1   1   1] * [i0]
          [0   1   0  -1   1   0  -1   0]   [i1]
          [0   0   0   0   1  -1   1  -1]   [i2]
          [0  -1   0   1   1   0  -1   0]   [i3]
        */

        r0 = vaddq_f32 (sr0, sr1);
        i0 = vaddq_f32 (si0, si1);
        r1 = vaddq_f32 (dr0, di1);
        i1 = vsubq_f32 (di0, dr1);
        r2 = vsubq_f32 (sr0, sr1);
        i2 = vsubq_f32 (si0, si1);
        r3 = vsubq_f32 (dr0, di1);
        i3 = vaddq_f32 (di0, dr1);

        *out++ = r0;
        *out++ = i0;
        *out++ = r1;
        *out++ = i1;
        *out++ = r2;
        *out++ = i2;
        *out++ = r3;
        *out++ = i3;
    }
}

static void pffft_cplx_preprocess (int Ncvec, const float32x4_t* in, float32x4_t* out, const float32x4_t* e)
{
    int k, dk = Ncvec / (int) SIMD_SZ; // number of 4x4 matrix blocks
    float32x4_t r0, i0, r1, i1, r2, i2, r3, i3;
    float32x4_t sr0, dr0, sr1, dr1, si0, di0, si1, di1;
    assert (in != out);
    for (k = 0; k < dk; ++k)
    {
        r0 = in[8 * k + 0];
        i0 = in[8 * k + 1];
        r1 = in[8 * k + 2];
        i1 = in[8 * k + 3];
        r2 = in[8 * k + 4];
        i2 = in[8 * k + 5];
        r3 = in[8 * k + 6];
        i3 = in[8 * k + 7];

        sr0 = vaddq_f32 (r0, r2);
        dr0 = vsubq_f32 (r0, r2);
        sr1 = vaddq_f32 (r1, r3);
        dr1 = vsubq_f32 (r1, r3);
        si0 = vaddq_f32 (i0, i2);
        di0 = vsubq_f32 (i0, i2);
        si1 = vaddq_f32 (i1, i3);
        di1 = vsubq_f32 (i1, i3);

        r0 = vaddq_f32 (sr0, sr1);
        i0 = vaddq_f32 (si0, si1);
        r1 = vsubq_f32 (dr0, di1);
        i1 = vaddq_f32 (di0, dr1);
        r2 = vsubq_f32 (sr0, sr1);
        i2 = vsubq_f32 (si0, si1);
        r3 = vaddq_f32 (dr0, di1);
        i3 = vsubq_f32 (di0, dr1);

        std::tie (r1, i1) = cplx_mul_conj_v (r1, i1, e[k * 6 + 0], e[k * 6 + 1]);
        std::tie (r2, i2) = cplx_mul_conj_v (r2, i2, e[k * 6 + 2], e[k * 6 + 3]);
        std::tie (r3, i3) = cplx_mul_conj_v (r3, i3, e[k * 6 + 4], e[k * 6 + 5]);

        transpose4 (r0, r1, r2, r3);
        transpose4 (i0, i1, i2, i3);

        *out++ = r0;
        *out++ = i0;
        *out++ = r1;
        *out++ = i1;
        *out++ = r2;
        *out++ = i2;
        *out++ = r3;
        *out++ = i3;
    }
}

//====================================================================
static void radf2_ps (int ido, int l1, const float32x4_t* __restrict cc, float32x4_t* __restrict ch, const float* wa1)
{
    int i, k, l1ido = l1 * ido;
    for (k = 0; k < l1ido; k += ido)
    {
        auto a = cc[k], b = cc[k + l1ido];
        ch[2 * k] = vaddq_f32 (a, b);
        ch[2 * (k + ido) - 1] = vsubq_f32 (a, b);
    }
    if (ido < 2)
        return;
    if (ido != 2)
    {
        for (k = 0; k < l1ido; k += ido)
        {
            for (i = 2; i < ido; i += 2)
            {
                auto tr2 = cc[i - 1 + k + l1ido], ti2 = cc[i + k + l1ido];
                auto br = cc[i - 1 + k], bi = cc[i + k];
                std::tie (tr2, ti2) = cplx_mul_conj (tr2, ti2, wa1[i - 2], wa1[i - 1]);
                ch[i + 2 * k] = vaddq_f32 (bi, ti2);
                ch[2 * (k + ido) - i] = vsubq_f32 (ti2, bi);
                ch[i - 1 + 2 * k] = vaddq_f32 (br, tr2);
                ch[2 * (k + ido) - i - 1] = vsubq_f32 (br, tr2);
            }
        }
        if (ido % 2 == 1)
            return;
    }
    for (k = 0; k < l1ido; k += ido)
    {
        ch[2 * k + ido] = vnegq_f32 (cc[ido - 1 + k + l1ido]);
        ch[2 * k + ido - 1] = cc[k + ido - 1];
    }
}

static void radf3_ps (int ido, int l1, const float32x4_t* __restrict cc, float32x4_t* __restrict ch, const float* wa1, const float* wa2)
{
    static constexpr float taur = -0.5f;
    static constexpr float taui = 0.866025403784439f;
    int i, k, ic;
    float32x4_t ci2, di2, di3, cr2, dr2, dr3, ti2, ti3, tr2, tr3;
    for (k = 0; k < l1; k++)
    {
        cr2 = vaddq_f32 (cc[(k + l1) * ido], cc[(k + 2 * l1) * ido]);
        ch[3 * k * ido] = vaddq_f32 (cc[k * ido], cr2);
        ch[(3 * k + 2) * ido] = vmulq_n_f32 (vsubq_f32 (cc[(k + l1 * 2) * ido], cc[(k + l1) * ido]), taui);
        ch[ido - 1 + (3 * k + 1) * ido] = vaddq_f32 (cc[k * ido], vmulq_n_f32 (cr2, taur));
    }
    if (ido == 1)
        return;
    for (k = 0; k < l1; k++)
    {
        for (i = 2; i < ido; i += 2)
        {
            ic = ido - i;
            std::tie (dr2, di2) = cplx_mul_conj (cc[i - 1 + (k + l1) * ido], cc[i + (k + l1) * ido], wa1[i - 2], wa1[i - 1]);
            std::tie (dr3, di3) = cplx_mul_conj (cc[i - 1 + (k + l1 * 2) * ido], cc[i + (k + l1 * 2) * ido], wa2[i - 2], wa2[i - 1]);

            cr2 = vaddq_f32 (dr2, dr3);
            ci2 = vaddq_f32 (di2, di3);
            ch[i - 1 + 3 * k * ido] = vaddq_f32 (cc[i - 1 + k * ido], cr2);
            ch[i + 3 * k * ido] = vaddq_f32 (cc[i + k * ido], ci2);
            tr2 = vaddq_f32 (cc[i - 1 + k * ido], vmulq_n_f32 (cr2, taur));
            ti2 = vaddq_f32 (cc[i + k * ido], vmulq_n_f32 (ci2, taur));
            tr3 = vmulq_n_f32 (vsubq_f32 (di2, di3), taui);
            ti3 = vmulq_n_f32 (vsubq_f32 (dr3, dr2), taui);
            ch[i - 1 + (3 * k + 2) * ido] = vaddq_f32 (tr2, tr3);
            ch[ic - 1 + (3 * k + 1) * ido] = vsubq_f32 (tr2, tr3);
            ch[i + (3 * k + 2) * ido] = vaddq_f32 (ti2, ti3);
            ch[ic + (3 * k + 1) * ido] = vsubq_f32 (ti3, ti2);
        }
    }
}

static void radf4_ps (int ido, int l1, const float32x4_t* __restrict cc, float32x4_t* __restrict ch, const float* __restrict wa1, const float* __restrict wa2, const float* __restrict wa3)
{
    static constexpr float minus_hsqt2 = -0.7071067811865475f;
    int i, k, l1ido = l1 * ido;
    {
        const auto* __restrict cc_ = cc, * __restrict cc_end = cc + l1ido;
        auto* __restrict ch_ = ch;
        while (cc < cc_end)
        {
            // this loop represents between 25% and 40% of total radf4_ps cost !
            auto a0 = cc[0], a1 = cc[l1ido];
            auto a2 = cc[2 * l1ido], a3 = cc[3 * l1ido];
            auto tr1 = vaddq_f32 (a1, a3);
            auto tr2 = vaddq_f32 (a0, a2);
            ch[2 * ido - 1] = vsubq_f32 (a0, a2);
            ch[2 * ido] = vsubq_f32 (a3, a1);
            ch[0] = vaddq_f32 (tr1, tr2);
            ch[4 * ido - 1] = vsubq_f32 (tr2, tr1);
            cc += ido;
            ch += 4 * ido;
        }
        cc = cc_;
        ch = ch_;
    }
    if (ido < 2)
        return;
    if (ido != 2)
    {
        for (k = 0; k < l1ido; k += ido)
        {
            const auto* __restrict pc = (float32x4_t*) (cc + 1 + k);
            for (i = 2; i < ido; i += 2, pc += 2)
            {
                int ic = ido - i;
                float32x4_t cr2, ci2, cr3, ci3, cr4, ci4;
                float32x4_t tr1, ti1, tr2, ti2, tr3, ti3, tr4, ti4;

                std::tie (cr2, ci2) = cplx_mul_conj (pc[1 * l1ido + 0], pc[1 * l1ido + 1], wa1[i - 2], wa1[i - 1]);
                std::tie (cr3, ci3) = cplx_mul_conj (pc[2 * l1ido + 0], pc[2 * l1ido + 1], wa2[i - 2], wa2[i - 1]);
                std::tie (cr4, ci4) = cplx_mul_conj (pc[3 * l1ido + 0], pc[3 * l1ido + 1], wa3[i - 2], wa3[i - 1]);

                /* at this point, on SSE, five of "cr2 cr3 cr4 ci2 ci3 ci4" should be loaded in registers */

                tr1 = vaddq_f32 (cr2, cr4);
                tr4 = vsubq_f32 (cr4, cr2);
                tr2 = vaddq_f32 (pc[0], cr3);
                tr3 = vsubq_f32 (pc[0], cr3);
                ch[i - 1 + 4 * k] = vaddq_f32 (tr1, tr2);
                ch[ic - 1 + 4 * k + 3 * ido] = vsubq_f32 (tr2, tr1); // at this point tr1 and tr2 can be disposed
                ti1 = vaddq_f32 (ci2, ci4);
                ti4 = vsubq_f32 (ci2, ci4);
                ch[i - 1 + 4 * k + 2 * ido] = vaddq_f32 (ti4, tr3);
                ch[ic - 1 + 4 * k + 1 * ido] = vsubq_f32 (tr3, ti4); // dispose tr3, ti4
                ti2 = vaddq_f32 (pc[1], ci3);
                ti3 = vsubq_f32 (pc[1], ci3);
                ch[i + 4 * k] = vaddq_f32 (ti1, ti2);
                ch[ic + 4 * k + 3 * ido] = vsubq_f32 (ti1, ti2);
                ch[i + 4 * k + 2 * ido] = vaddq_f32 (tr4, ti3);
                ch[ic + 4 * k + 1 * ido] = vsubq_f32 (tr4, ti3);
            }
        }
        if (ido % 2 == 1)
            return;
    }
    for (k = 0; k < l1ido; k += ido)
    {
        auto a = cc[ido - 1 + k + l1ido], b = cc[ido - 1 + k + 3 * l1ido];
        auto c = cc[ido - 1 + k], d = cc[ido - 1 + k + 2 * l1ido];
        auto ti1 = vmulq_n_f32 (vaddq_f32 (a, b), minus_hsqt2);
        auto tr1 = vmulq_n_f32 (vsubq_f32 (b, a), minus_hsqt2);
        ch[ido - 1 + 4 * k] = vaddq_f32 (tr1, c);
        ch[ido - 1 + 4 * k + 2 * ido] = vsubq_f32 (c, tr1);
        ch[4 * k + 1 * ido] = vsubq_f32 (ti1, d);
        ch[4 * k + 3 * ido] = vaddq_f32 (ti1, d);
    }
}

static void radf5_ps (int ido, int l1, const float32x4_t* __restrict cc, float32x4_t* __restrict ch, const float* wa1, const float* wa2, const float* wa3, const float* wa4)
{
    static constexpr float tr11 = .309016994374947f;
    static constexpr float ti11 = .951056516295154f;
    static constexpr float tr12 = -.809016994374947f;
    static constexpr float ti12 = .587785252292473f;

    /* System generated locals */
    int cc_offset, ch_offset;

    /* Local variables */
    int i, k, ic;
    float32x4_t ci2, di2, ci4, ci5, di3, di4, di5, ci3, cr2, cr3, dr2, dr3, dr4, dr5,
        cr5, cr4, ti2, ti3, ti5, ti4, tr2, tr3, tr4, tr5;
    int idp2;

#define cc_ref(a_1, a_2, a_3) cc[((a_3) * l1 + (a_2)) * ido + a_1]
#define ch_ref(a_1, a_2, a_3) ch[((a_3) * 5 + (a_2)) * ido + a_1]

    /* Parameter adjustments */
    ch_offset = 1 + ido * 6;
    ch -= ch_offset;
    cc_offset = 1 + ido * (1 + l1);
    cc -= cc_offset;

    /* Function Body */
    for (k = 1; k <= l1; ++k)
    {
        cr2 = vaddq_f32 (cc_ref (1, k, 5), cc_ref (1, k, 2));
        ci5 = vsubq_f32 (cc_ref (1, k, 5), cc_ref (1, k, 2));
        cr3 = vaddq_f32 (cc_ref (1, k, 4), cc_ref (1, k, 3));
        ci4 = vsubq_f32 (cc_ref (1, k, 4), cc_ref (1, k, 3));
        ch_ref (1, 1, k) = vaddq_f32 (cc_ref (1, k, 1), vaddq_f32 (cr2, cr3));
        ch_ref (ido, 2, k) = vaddq_f32 (cc_ref (1, k, 1), vaddq_f32 (vmulq_n_f32 (cr2, tr11), vmulq_n_f32 (cr3, tr12)));
        ch_ref (1, 3, k) = vaddq_f32 (vmulq_n_f32 (ci5, ti11), vmulq_n_f32 (ci4, ti12));
        ch_ref (ido, 4, k) = vaddq_f32 (cc_ref (1, k, 1), vaddq_f32 (vmulq_n_f32 (cr2, tr12), vmulq_n_f32 (cr3, tr11)));
        ch_ref (1, 5, k) = vsubq_f32 (vmulq_n_f32 (ci5, ti12), vmulq_n_f32 (ci4, ti11));
    }
    if (ido == 1)
    {
        return;
    }
    idp2 = ido + 2;
    for (k = 1; k <= l1; ++k)
    {
        for (i = 3; i <= ido; i += 2)
        {
            ic = idp2 - i;
            std::tie (dr2, di2) = cplx_mul_conj (cc_ref (i - 1, k, 2), cc_ref (i, k, 2), wa1[i - 3], wa1[i - 2]);
            std::tie (dr3, di3) = cplx_mul_conj (cc_ref (i - 1, k, 3), cc_ref (i, k, 3), wa2[i - 3], wa2[i - 2]);
            std::tie (dr4, di4) = cplx_mul_conj (cc_ref (i - 1, k, 4), cc_ref (i, k, 4), wa3[i - 3], wa3[i - 2]);
            std::tie (dr5, di5) = cplx_mul_conj (cc_ref (i - 1, k, 5), cc_ref (i, k, 5), wa4[i - 3], wa4[i - 2]);
            cr2 = vaddq_f32 (dr2, dr5);
            ci5 = vsubq_f32 (dr5, dr2);
            cr5 = vsubq_f32 (di2, di5);
            ci2 = vaddq_f32 (di2, di5);
            cr3 = vaddq_f32 (dr3, dr4);
            ci4 = vsubq_f32 (dr4, dr3);
            cr4 = vsubq_f32 (di3, di4);
            ci3 = vaddq_f32 (di3, di4);
            ch_ref (i - 1, 1, k) = vaddq_f32 (cc_ref (i - 1, k, 1), vaddq_f32 (cr2, cr3));
            ch_ref (i, 1, k) = vsubq_f32 (cc_ref (i, k, 1), vaddq_f32 (ci2, ci3)); //
            tr2 = vaddq_f32 (cc_ref (i - 1, k, 1), vaddq_f32 (vmulq_n_f32 (cr2, tr11), vmulq_n_f32 (cr3, tr12)));
            ti2 = vsubq_f32 (cc_ref (i, k, 1), vaddq_f32 (vmulq_n_f32 (ci2, tr11), vmulq_n_f32 (ci3, tr12))); //
            tr3 = vaddq_f32 (cc_ref (i - 1, k, 1), vaddq_f32 (vmulq_n_f32 (cr2, tr12), vmulq_n_f32 (cr3, tr11)));
            ti3 = vsubq_f32 (cc_ref (i, k, 1), vaddq_f32 (vmulq_n_f32 (ci2, tr12), vmulq_n_f32 (ci3, tr11))); //
            tr5 = vaddq_f32 (vmulq_n_f32 (cr5, ti11), vmulq_n_f32 (cr4, ti12));
            ti5 = vaddq_f32 (vmulq_n_f32 (ci5, ti11), vmulq_n_f32 (ci4, ti12));
            tr4 = vsubq_f32 (vmulq_n_f32 (cr5, ti12), vmulq_n_f32 (cr4, ti11));
            ti4 = vsubq_f32 (vmulq_n_f32 (ci5, ti12), vmulq_n_f32 (ci4, ti11));
            ch_ref (i - 1, 3, k) = vsubq_f32 (tr2, tr5);
            ch_ref (ic - 1, 2, k) = vaddq_f32 (tr2, tr5);
            ch_ref (i, 3, k) = vaddq_f32 (ti2, ti5);
            ch_ref (ic, 2, k) = vsubq_f32 (ti5, ti2);
            ch_ref (i - 1, 5, k) = vsubq_f32 (tr3, tr4);
            ch_ref (ic - 1, 4, k) = vaddq_f32 (tr3, tr4);
            ch_ref (i, 5, k) = vaddq_f32 (ti3, ti4);
            ch_ref (ic, 4, k) = vsubq_f32 (ti4, ti3);
        }
    }
#undef cc_ref
#undef ch_ref
}

static float32x4_t* rfftf1_ps (int n, const float32x4_t* input_readonly, float32x4_t* work1, float32x4_t* work2, const float* wa, const int* ifac)
{
    auto* in = (float32x4_t*) input_readonly;
    auto* out = (in == work2 ? work1 : work2);
    int nf = ifac[1], k1;
    int l2 = n;
    int iw = n - 1;
    assert (in != out && work1 != work2);
    for (k1 = 1; k1 <= nf; ++k1)
    {
        int kh = nf - k1;
        int ip = ifac[kh + 2];
        int l1 = l2 / ip;
        int ido = n / l2;
        iw -= (ip - 1) * ido;
        switch (ip)
        {
            case 5:
            {
                int ix2 = iw + ido;
                int ix3 = ix2 + ido;
                int ix4 = ix3 + ido;
                radf5_ps (ido, l1, in, out, &wa[iw], &wa[ix2], &wa[ix3], &wa[ix4]);
            }
            break;
            case 4:
            {
                int ix2 = iw + ido;
                int ix3 = ix2 + ido;
                radf4_ps (ido, l1, in, out, &wa[iw], &wa[ix2], &wa[ix3]);
            }
            break;
            case 3:
            {
                int ix2 = iw + ido;
                radf3_ps (ido, l1, in, out, &wa[iw], &wa[ix2]);
            }
            break;
            case 2:
                radf2_ps (ido, l1, in, out, &wa[iw]);
                break;
            default:
                assert (0);
                break;
        }
        l2 = l1;
        if (out == work2)
        {
            out = work1;
            in = work2;
        }
        else
        {
            out = work2;
            in = work1;
        }
    }
    return in; /* this is in fact the output .. */
}

//====================================================================
static inline void pffft_real_finalize_4x4 (const float32x4_t* in0, const float32x4_t* in1, const float32x4_t* in, const float32x4_t* e, float32x4_t* out)
{
    float32x4_t r0, i0, r1, i1, r2, i2, r3, i3;
    float32x4_t sr0, dr0, sr1, dr1, si0, di0, si1, di1;
    r0 = *in0;
    i0 = *in1;
    r1 = *in++;
    i1 = *in++;
    r2 = *in++;
    i2 = *in++;
    r3 = *in++;
    i3 = *in++;
    transpose4 (r0, r1, r2, r3);
    transpose4 (i0, i1, i2, i3);

    /*
      transformation for each column is:

      [1   1   1   1   0   0   0   0]   [r0]
      [1   0  -1   0   0  -1   0   1]   [r1]
      [1   0  -1   0   0   1   0  -1]   [r2]
      [1  -1   1  -1   0   0   0   0]   [r3]
      [0   0   0   0   1   1   1   1] * [i0]
      [0  -1   0   1  -1   0   1   0]   [i1]
      [0  -1   0   1   1   0  -1   0]   [i2]
      [0   0   0   0  -1   1  -1   1]   [i3]
    */

    std::tie (r1, i1) = cplx_mul_v (r1, i1, e[0], e[1]);
    std::tie (r2, i2) = cplx_mul_v (r2, i2, e[2], e[3]);
    std::tie (r3, i3) = cplx_mul_v (r3, i3, e[4], e[5]);

    sr0 = vaddq_f32 (r0, r2);
    dr0 = vsubq_f32 (r0, r2);
    sr1 = vaddq_f32 (r1, r3);
    dr1 = vsubq_f32 (r3, r1);
    si0 = vaddq_f32 (i0, i2);
    di0 = vsubq_f32 (i0, i2);
    si1 = vaddq_f32 (i1, i3);
    di1 = vsubq_f32 (i3, i1);

    r0 = vaddq_f32 (sr0, sr1);
    r3 = vsubq_f32 (sr0, sr1);
    i0 = vaddq_f32 (si0, si1);
    i3 = vsubq_f32 (si1, si0);
    r1 = vaddq_f32 (dr0, di1);
    r2 = vsubq_f32 (dr0, di1);
    i1 = vsubq_f32 (dr1, di0);
    i2 = vaddq_f32 (dr1, di0);

    *out++ = r0;
    *out++ = i0;
    *out++ = r1;
    *out++ = i1;
    *out++ = r2;
    *out++ = i2;
    *out++ = r3;
    *out++ = i3;
}

static void pffft_real_finalize (int Ncvec, const float32x4_t* in, float32x4_t* out, const float32x4_t* e)
{
    int k, dk = Ncvec / (int) SIMD_SZ; // number of 4x4 matrix blocks
    /* fftpack order is f0r f1r f1i f2r f2i ... f(n-1)r f(n-1)i f(n)r */

    float32x4_t cr, ci, *uout = (float32x4_t*) out;
    float32x4_t save = in[7], zero = {};
    float xr0, xi0, xr1, xi1, xr2, xi2, xr3, xi3;
    static constexpr float s = M_SQRT2 / 2;

    cr = in[0];
    ci = in[Ncvec * 2 - 1];
    assert (in != out);
    pffft_real_finalize_4x4 (&zero, &zero, in + 1, e, out);

    /*
    [cr0 cr1 cr2 cr3 ci0 ci1 ci2 ci3]

    [Xr(1)]  ] [1   1   1   1   0   0   0   0]
    [Xr(N/4) ] [0   0   0   0   1   s   0  -s]
    [Xr(N/2) ] [1   0  -1   0   0   0   0   0]
    [Xr(3N/4)] [0   0   0   0   1  -s   0   s]
    [Xi(1)   ] [1  -1   1  -1   0   0   0   0]
    [Xi(N/4) ] [0   0   0   0   0  -s  -1  -s]
    [Xi(N/2) ] [0  -1   0   1   0   0   0   0]
    [Xi(3N/4)] [0   0   0   0   0  -s   1  -s]
  */

    xr0 = (((float*)&cr)[0] + ((float*)&cr)[2]) + (((float*)&cr)[1] + ((float*)&cr)[3]);
    ((float*)&uout[0])[0] = xr0;
    xi0 = (((float*)&cr)[0] + ((float*)&cr)[2]) - (((float*)&cr)[1] + ((float*)&cr)[3]);
    ((float*)&uout[1])[0] = xi0;
    xr2 = (((float*)&cr)[0] - ((float*)&cr)[2]);
    ((float*)&uout[4])[0] = xr2;
    xi2 = (((float*)&cr)[3] - ((float*)&cr)[1]);
    ((float*)&uout[5])[0] = xi2;
    xr1 = ((float*)&ci)[0] + s * (((float*)&ci)[1] - ((float*)&ci)[3]);
    ((float*)&uout[2])[0] = xr1;
    xi1 = -((float*)&ci)[2] - s * (((float*)&ci)[1] + ((float*)&ci)[3]);
    ((float*)&uout[3])[0] = xi1;
    xr3 = ((float*)&ci)[0] - s * (((float*)&ci)[1] - ((float*)&ci)[3]);
    ((float*)&uout[6])[0] = xr3;
    xi3 = ((float*)&ci)[2] - s * (((float*)&ci)[1] + ((float*)&ci)[3]);
    ((float*)&uout[7])[0] = xi3;

    for (k = 1; k < dk; ++k)
    {
        auto save_next = in[8 * k + 7];
        pffft_real_finalize_4x4 (&save, &in[8 * k + 0], in + 8 * k + 1, e + k * 6, out + k * 8);
        save = save_next;
    }
}

//====================================================================
static inline void pffft_real_preprocess_4x4 (const float32x4_t* in,
                                              const float32x4_t* e,
                                              float32x4_t* out,
                                              int first)
{
    float32x4_t r0 = in[0], i0 = in[1], r1 = in[2], i1 = in[3], r2 = in[4], i2 = in[5], r3 = in[6], i3 = in[7];
    /*
      transformation for each column is:

      [1   1   1   1   0   0   0   0]   [r0]
      [1   0   0  -1   0  -1  -1   0]   [r1]
      [1  -1  -1   1   0   0   0   0]   [r2]
      [1   0   0  -1   0   1   1   0]   [r3]
      [0   0   0   0   1  -1   1  -1] * [i0]
      [0  -1   1   0   1   0   0   1]   [i1]
      [0   0   0   0   1   1  -1  -1]   [i2]
      [0   1  -1   0   1   0   0   1]   [i3]
    */

    auto sr0 = vaddq_f32 (r0, r3), dr0 = vsubq_f32 (r0, r3);
    auto sr1 = vaddq_f32 (r1, r2), dr1 = vsubq_f32 (r1, r2);
    auto si0 = vaddq_f32 (i0, i3), di0 = vsubq_f32 (i0, i3);
    auto si1 = vaddq_f32 (i1, i2), di1 = vsubq_f32 (i1, i2);

    r0 = vaddq_f32 (sr0, sr1);
    r2 = vsubq_f32 (sr0, sr1);
    r1 = vsubq_f32 (dr0, si1);
    r3 = vaddq_f32 (dr0, si1);
    i0 = vsubq_f32 (di0, di1);
    i2 = vaddq_f32 (di0, di1);
    i1 = vsubq_f32 (si0, dr1);
    i3 = vaddq_f32 (si0, dr1);

    std::tie (r1, i1) = cplx_mul_conj_v (r1, i1, e[0], e[1]);
    std::tie (r2, i2) = cplx_mul_conj_v (r2, i2, e[2], e[3]);
    std::tie (r3, i3) = cplx_mul_conj_v (r3, i3, e[4], e[5]);

    transpose4 (r0, r1, r2, r3);
    transpose4 (i0, i1, i2, i3);

    if (! first)
    {
        *out++ = r0;
        *out++ = i0;
    }
    *out++ = r1;
    *out++ = i1;
    *out++ = r2;
    *out++ = i2;
    *out++ = r3;
    *out++ = i3;
}

static void pffft_real_preprocess (int Ncvec, const float32x4_t* in, float32x4_t* out, const float32x4_t* e)
{
    int k, dk = Ncvec / (int) SIMD_SZ; // number of 4x4 matrix blocks
    /* fftpack order is f0r f1r f1i f2r f2i ... f(n-1)r f(n-1)i f(n)r */

    float32x4_t Xr, Xi, *uout = (float32x4_t*) out;
    float cr0, ci0, cr1, ci1, cr2, ci2, cr3, ci3;
    static constexpr float s = M_SQRT2;
    assert (in != out);

    for (k = 0; k < 4; ++k)
    {
        ((float*)&Xr)[k] = ((float*) in)[8 * k];
        ((float*)&Xi)[k] = ((float*) in)[8 * k + 4];
    }

    pffft_real_preprocess_4x4 (in, e, out + 1, 1); // will write only 6 values

    /*
      [Xr0 Xr1 Xr2 Xr3 Xi0 Xi1 Xi2 Xi3]

      [cr0] [1   0   2   0   1   0   0   0]
      [cr1] [1   0   0   0  -1   0  -2   0]
      [cr2] [1   0  -2   0   1   0   0   0]
      [cr3] [1   0   0   0  -1   0   2   0]
      [ci0] [0   2   0   2   0   0   0   0]
      [ci1] [0   s   0  -s   0  -s   0  -s]
      [ci2] [0   0   0   0   0  -2   0   2]
      [ci3] [0  -s   0   s   0  -s   0  -s]
    */
    for (k = 1; k < dk; ++k)
    {
        pffft_real_preprocess_4x4 (in + 8 * k, e + k * 6, out - 1 + k * 8, 0);
    }

    cr0 = (((float*)&Xr)[0] + ((float*)&Xi)[0]) + 2 * ((float*)&Xr)[2];
    ((float*)&uout[0])[0] = cr0;
    cr1 = (((float*)&Xr)[0] - ((float*)&Xi)[0]) - 2 * ((float*)&Xi)[2];
    ((float*)&uout[0])[1] = cr1;
    cr2 = (((float*)&Xr)[0] + ((float*)&Xi)[0]) - 2 * ((float*)&Xr)[2];
    ((float*)&uout[0])[2] = cr2;
    cr3 = (((float*)&Xr)[0] - ((float*)&Xi)[0]) + 2 * ((float*)&Xi)[2];
    ((float*)&uout[0])[3] = cr3;
    ci0 = 2 * (((float*)&Xr)[1] + ((float*)&Xr)[3]);
    ((float*)&uout[2 * Ncvec - 1])[0] = ci0;
    ci1 = s * (((float*)&Xr)[1] - ((float*)&Xr)[3]) - s * (((float*)&Xi)[1] + ((float*)&Xi)[3]);
    ((float*)&uout[2 * Ncvec - 1])[1] = ci1;
    ci2 = 2 * (((float*)&Xi)[3] - ((float*)&Xi)[1]);
    ((float*)&uout[2 * Ncvec - 1])[2] = ci2;
    ci3 = -s * (((float*)&Xr)[1] - ((float*)&Xr)[3]) - s * (((float*)&Xi)[1] + ((float*)&Xi)[3]);
    ((float*)&uout[2 * Ncvec - 1])[3] = ci3;
}

//====================================================================
static void radb2_ps (int ido, int l1, const float32x4_t* cc, float32x4_t* ch, const float* wa1)
{
    static constexpr float minus_two = -2;
    int i, k, l1ido = l1 * ido;
    float32x4_t a, b, c, d, tr2, ti2;
    for (k = 0; k < l1ido; k += ido)
    {
        a = cc[2 * k];
        b = cc[2 * (k + ido) - 1];
        ch[k] = vaddq_f32 (a, b);
        ch[k + l1ido] = vsubq_f32 (a, b);
    }
    if (ido < 2)
        return;
    if (ido != 2)
    {
        for (k = 0; k < l1ido; k += ido)
        {
            for (i = 2; i < ido; i += 2)
            {
                a = cc[i - 1 + 2 * k];
                b = cc[2 * (k + ido) - i - 1];
                c = cc[i + 0 + 2 * k];
                d = cc[2 * (k + ido) - i + 0];
                ch[i - 1 + k] = vaddq_f32 (a, b);
                tr2 = vsubq_f32 (a, b);
                ch[i + 0 + k] = vsubq_f32 (c, d);
                ti2 = vaddq_f32 (c, d);
                std::tie (tr2, ti2) = cplx_mul (tr2, ti2, wa1[i - 2], wa1[i - 1]);
                ch[i - 1 + k + l1ido] = tr2;
                ch[i + 0 + k + l1ido] = ti2;
            }
        }
        if (ido % 2 == 1)
            return;
    }
    for (k = 0; k < l1ido; k += ido)
    {
        a = cc[2 * k + ido - 1];
        b = cc[2 * k + ido];
        ch[k + ido - 1] = vaddq_f32 (a, a);
        ch[k + ido - 1 + l1ido] = vmulq_n_f32 (b, minus_two);
    }
}

static void radb3_ps (int ido, int l1, const float32x4_t* __restrict cc, float32x4_t* __restrict ch, const float* wa1, const float* wa2)
{
    static constexpr float taur = -0.5f;
    static constexpr float taui = 0.866025403784439f;
    static constexpr float taui_2 = 0.866025403784439f * 2;
    int i, k, ic;
    float32x4_t ci2, ci3, di2, di3, cr2, cr3, dr2, dr3, ti2, tr2;
    for (k = 0; k < l1; k++)
    {
        tr2 = cc[ido - 1 + (3 * k + 1) * ido];
        tr2 = vaddq_f32 (tr2, tr2);
        cr2 = vmlaq_n_f32 (cc[3 * k * ido], tr2, taur);
        ch[k * ido] = vaddq_f32 (cc[3 * k * ido], tr2);
        ci3 = vmulq_n_f32 (cc[(3 * k + 2) * ido], taui_2);
        ch[(k + l1) * ido] = vsubq_f32 (cr2, ci3);
        ch[(k + 2 * l1) * ido] = vaddq_f32 (cr2, ci3);
    }
    if (ido == 1)
        return;
    for (k = 0; k < l1; k++)
    {
        for (i = 2; i < ido; i += 2)
        {
            ic = ido - i;
            tr2 = vaddq_f32 (cc[i - 1 + (3 * k + 2) * ido], cc[ic - 1 + (3 * k + 1) * ido]);
            cr2 = vmlaq_n_f32 (cc[i - 1 + 3 * k * ido], tr2, taur);
            ch[i - 1 + k * ido] = vaddq_f32 (cc[i - 1 + 3 * k * ido], tr2);
            ti2 = vsubq_f32 (cc[i + (3 * k + 2) * ido], cc[ic + (3 * k + 1) * ido]);
            ci2 = vmlaq_n_f32 (cc[i + 3 * k * ido], ti2, taur);
            ch[i + k * ido] = vaddq_f32 (cc[i + 3 * k * ido], ti2);
            cr3 = vmulq_n_f32 (vsubq_f32 (cc[i - 1 + (3 * k + 2) * ido], cc[ic - 1 + (3 * k + 1) * ido]), taui);
            ci3 = vmulq_n_f32 (vaddq_f32 (cc[i + (3 * k + 2) * ido], cc[ic + (3 * k + 1) * ido]), taui);
            dr2 = vsubq_f32 (cr2, ci3);
            dr3 = vaddq_f32 (cr2, ci3);
            di2 = vaddq_f32 (ci2, cr3);
            di3 = vsubq_f32 (ci2, cr3);
            std::tie (dr2, di2) = cplx_mul (dr2, di2, wa1[i - 2], wa1[i - 1]);
            ch[i - 1 + (k + l1) * ido] = dr2;
            ch[i + (k + l1) * ido] = di2;
            std::tie (dr3, di3) = cplx_mul (dr3, di3, wa2[i - 2], wa2[i - 1]);
            ch[i - 1 + (k + 2 * l1) * ido] = dr3;
            ch[i + (k + 2 * l1) * ido] = di3;
        }
    }
}

static void radb4_ps (int ido, int l1, const float32x4_t* __restrict cc, float32x4_t* __restrict ch, const float* __restrict wa1, const float* __restrict wa2, const float* __restrict wa3)
{
    static constexpr float minus_sqrt2 = -1.414213562373095f;
    int i, k, l1ido = l1 * ido;
    float32x4_t ci2, ci3, ci4, cr2, cr3, cr4, ti1, ti2, ti3, ti4, tr1, tr2, tr3, tr4;
    {
        const float32x4_t* __restrict cc_ = cc, * __restrict ch_end = ch + l1ido;
        float32x4_t* ch_ = ch;
        while (ch < ch_end)
        {
            auto a = cc[0], b = cc[4 * ido - 1];
            auto c = cc[2 * ido], d = cc[2 * ido - 1];
            tr3 = vaddq_f32 (d, d);
            tr2 = vaddq_f32 (a, b);
            tr1 = vsubq_f32 (a, b);
            tr4 = vaddq_f32 (c, c);
            ch[0 * l1ido] = vaddq_f32 (tr2, tr3);
            ch[2 * l1ido] = vsubq_f32 (tr2, tr3);
            ch[1 * l1ido] = vsubq_f32 (tr1, tr4);
            ch[3 * l1ido] = vaddq_f32 (tr1, tr4);

            cc += 4 * ido;
            ch += ido;
        }
        cc = cc_;
        ch = ch_;
    }
    if (ido < 2)
        return;
    if (ido != 2)
    {
        for (k = 0; k < l1ido; k += ido)
        {
            const auto* __restrict pc = (float32x4_t*) (cc - 1 + 4 * k);
            auto* __restrict ph = (float32x4_t*) (ch + k + 1);
            for (i = 2; i < ido; i += 2)
            {
                tr1 = vsubq_f32 (pc[i], pc[4 * ido - i]);
                tr2 = vaddq_f32 (pc[i], pc[4 * ido - i]);
                ti4 = vsubq_f32 (pc[2 * ido + i], pc[2 * ido - i]);
                tr3 = vaddq_f32 (pc[2 * ido + i], pc[2 * ido - i]);
                ph[0] = vaddq_f32 (tr2, tr3);
                cr3 = vsubq_f32 (tr2, tr3);

                ti3 = vsubq_f32 (pc[2 * ido + i + 1], pc[2 * ido - i + 1]);
                tr4 = vaddq_f32 (pc[2 * ido + i + 1], pc[2 * ido - i + 1]);
                cr2 = vsubq_f32 (tr1, tr4);
                cr4 = vaddq_f32 (tr1, tr4);

                ti1 = vaddq_f32 (pc[i + 1], pc[4 * ido - i + 1]);
                ti2 = vsubq_f32 (pc[i + 1], pc[4 * ido - i + 1]);

                ph[1] = vaddq_f32 (ti2, ti3);
                ph += l1ido;
                ci3 = vsubq_f32 (ti2, ti3);
                ci2 = vaddq_f32 (ti1, ti4);
                ci4 = vsubq_f32 (ti1, ti4);
                std::tie (cr2, ci2) = cplx_mul (cr2, ci2, wa1[i - 2], wa1[i - 1]);
                ph[0] = cr2;
                ph[1] = ci2;
                ph += l1ido;
                std::tie (cr3, ci3) = cplx_mul (cr3, ci3, wa2[i - 2], wa2[i - 1]);
                ph[0] = cr3;
                ph[1] = ci3;
                ph += l1ido;
                std::tie (cr4, ci4) = cplx_mul (cr4, ci4, wa3[i - 2], wa3[i - 1]);
                ph[0] = cr4;
                ph[1] = ci4;
                ph = ph - 3 * l1ido + 2;
            }
        }
        if (ido % 2 == 1)
            return;
    }
    for (k = 0; k < l1ido; k += ido)
    {
        int i0 = 4 * k + ido;
        auto c = cc[i0 - 1], d = cc[i0 + 2 * ido - 1];
        auto a = cc[i0 + 0], b = cc[i0 + 2 * ido + 0];
        tr1 = vsubq_f32 (c, d);
        tr2 = vaddq_f32 (c, d);
        ti1 = vaddq_f32 (b, a);
        ti2 = vsubq_f32 (b, a);
        ch[ido - 1 + k + 0 * l1ido] = vaddq_f32 (tr2, tr2);
        ch[ido - 1 + k + 1 * l1ido] = vmulq_n_f32 (vsubq_f32 (ti1, tr1), minus_sqrt2);
        ch[ido - 1 + k + 2 * l1ido] = vaddq_f32 (ti2, ti2);
        ch[ido - 1 + k + 3 * l1ido] = vmulq_n_f32 (vaddq_f32 (ti1, tr1), minus_sqrt2);
    }
}

static void radb5_ps (int ido, int l1, const float32x4_t* __restrict cc, float32x4_t* __restrict ch, const float* wa1, const float* wa2, const float* wa3, const float* wa4)
{
    static constexpr float tr11 = .309016994374947f;
    static constexpr float ti11 = .951056516295154f;
    static constexpr float tr12 = -.809016994374947f;
    static constexpr float ti12 = .587785252292473f;

    int cc_offset, ch_offset;

    /* Local variables */
    int i, k, ic;
    float32x4_t ci2, ci3, ci4, ci5, di3, di4, di5, di2, cr2, cr3, cr5, cr4, ti2, ti3,
        ti4, ti5, dr3, dr4, dr5, dr2, tr2, tr3, tr4, tr5;
    int idp2;

#define cc_ref(a_1, a_2, a_3) cc[((a_3) * 5 + (a_2)) * ido + a_1]
#define ch_ref(a_1, a_2, a_3) ch[((a_3) * l1 + (a_2)) * ido + a_1]

    /* Parameter adjustments */
    ch_offset = 1 + ido * (1 + l1);
    ch -= ch_offset;
    cc_offset = 1 + ido * 6;
    cc -= cc_offset;

    /* Function Body */
    for (k = 1; k <= l1; ++k)
    {
        ti5 = vaddq_f32 (cc_ref (1, 3, k), cc_ref (1, 3, k));
        ti4 = vaddq_f32 (cc_ref (1, 5, k), cc_ref (1, 5, k));
        tr2 = vaddq_f32 (cc_ref (ido, 2, k), cc_ref (ido, 2, k));
        tr3 = vaddq_f32 (cc_ref (ido, 4, k), cc_ref (ido, 4, k));
        ch_ref (1, k, 1) = vaddq_f32 (cc_ref (1, 1, k), vaddq_f32 (tr2, tr3));
        cr2 = vaddq_f32 (cc_ref (1, 1, k), vaddq_f32 (vmulq_n_f32 (tr2, tr11), vmulq_n_f32 (tr3, tr12)));
        cr3 = vaddq_f32 (cc_ref (1, 1, k), vaddq_f32 (vmulq_n_f32 (tr2, tr12), vmulq_n_f32 (tr3, tr11)));
        ci5 = vaddq_f32 (vmulq_n_f32 (ti5, ti11), vmulq_n_f32 (ti4, ti12));
        ci4 = vsubq_f32 (vmulq_n_f32 (ti5, ti12), vmulq_n_f32 (ti4, ti11));
        ch_ref (1, k, 2) = vsubq_f32 (cr2, ci5);
        ch_ref (1, k, 3) = vsubq_f32 (cr3, ci4);
        ch_ref (1, k, 4) = vaddq_f32 (cr3, ci4);
        ch_ref (1, k, 5) = vaddq_f32 (cr2, ci5);
    }
    if (ido == 1)
    {
        return;
    }
    idp2 = ido + 2;
    for (k = 1; k <= l1; ++k)
    {
        for (i = 3; i <= ido; i += 2)
        {
            ic = idp2 - i;
            ti5 = vaddq_f32 (cc_ref (i, 3, k), cc_ref (ic, 2, k));
            ti2 = vsubq_f32 (cc_ref (i, 3, k), cc_ref (ic, 2, k));
            ti4 = vaddq_f32 (cc_ref (i, 5, k), cc_ref (ic, 4, k));
            ti3 = vsubq_f32 (cc_ref (i, 5, k), cc_ref (ic, 4, k));
            tr5 = vsubq_f32 (cc_ref (i - 1, 3, k), cc_ref (ic - 1, 2, k));
            tr2 = vaddq_f32 (cc_ref (i - 1, 3, k), cc_ref (ic - 1, 2, k));
            tr4 = vsubq_f32 (cc_ref (i - 1, 5, k), cc_ref (ic - 1, 4, k));
            tr3 = vaddq_f32 (cc_ref (i - 1, 5, k), cc_ref (ic - 1, 4, k));
            ch_ref (i - 1, k, 1) = vaddq_f32 (cc_ref (i - 1, 1, k), vaddq_f32 (tr2, tr3));
            ch_ref (i, k, 1) = vaddq_f32 (cc_ref (i, 1, k), vaddq_f32 (ti2, ti3));
            cr2 = vaddq_f32 (cc_ref (i - 1, 1, k), vaddq_f32 (vmulq_n_f32 (tr2, tr11), vmulq_n_f32 (tr3, tr12)));
            ci2 = vaddq_f32 (cc_ref (i, 1, k), vaddq_f32 (vmulq_n_f32 (ti2, tr11), vmulq_n_f32 (ti3, tr12)));
            cr3 = vaddq_f32 (cc_ref (i - 1, 1, k), vaddq_f32 (vmulq_n_f32 (tr2, tr12), vmulq_n_f32 (tr3, tr11)));
            ci3 = vaddq_f32 (cc_ref (i, 1, k), vaddq_f32 (vmulq_n_f32 (ti2, tr12), vmulq_n_f32 (ti3, tr11)));
            cr5 = vaddq_f32 (vmulq_n_f32 (tr5, ti11), vmulq_n_f32 (tr4, ti12));
            ci5 = vaddq_f32 (vmulq_n_f32 (ti5, ti11), vmulq_n_f32 (ti4, ti12));
            cr4 = vsubq_f32 (vmulq_n_f32 (tr5, ti12), vmulq_n_f32 (tr4, ti11));
            ci4 = vsubq_f32 (vmulq_n_f32 (ti5, ti12), vmulq_n_f32 (ti4, ti11));
            dr3 = vsubq_f32 (cr3, ci4);
            dr4 = vaddq_f32 (cr3, ci4);
            di3 = vaddq_f32 (ci3, cr4);
            di4 = vsubq_f32 (ci3, cr4);
            dr5 = vaddq_f32 (cr2, ci5);
            dr2 = vsubq_f32 (cr2, ci5);
            di5 = vsubq_f32 (ci2, cr5);
            di2 = vaddq_f32 (ci2, cr5);
            std::tie (dr2, di2) = cplx_mul (dr2, di2, wa1[i - 3], wa1[i - 2]);
            std::tie (dr3, di3) = cplx_mul (dr3, di3, wa2[i - 3], wa2[i - 2]);
            std::tie (dr4, di4) = cplx_mul (dr4, di4, wa3[i - 3], wa3[i - 2]);
            std::tie (dr5, di5) = cplx_mul (dr5, di5, wa4[i - 3], wa4[i - 2]);

            ch_ref (i - 1, k, 2) = dr2;
            ch_ref (i, k, 2) = di2;
            ch_ref (i - 1, k, 3) = dr3;
            ch_ref (i, k, 3) = di3;
            ch_ref (i - 1, k, 4) = dr4;
            ch_ref (i, k, 4) = di4;
            ch_ref (i - 1, k, 5) = dr5;
            ch_ref (i, k, 5) = di5;
        }
    }
#undef cc_ref
#undef ch_ref
}

static float32x4_t* rfftb1_ps (int n, const float32x4_t* input_readonly, float32x4_t* work1, float32x4_t* work2, const float* wa, const int* ifac)
{
    auto* in = (float32x4_t*) input_readonly;
    auto* out = (in == work2 ? work1 : work2);
    int nf = ifac[1], k1;
    int l1 = 1;
    int iw = 0;
    assert (in != out);
    for (k1 = 1; k1 <= nf; k1++)
    {
        int ip = ifac[k1 + 1];
        int l2 = ip * l1;
        int ido = n / l2;
        switch (ip)
        {
            case 5:
            {
                int ix2 = iw + ido;
                int ix3 = ix2 + ido;
                int ix4 = ix3 + ido;
                radb5_ps (ido, l1, in, out, &wa[iw], &wa[ix2], &wa[ix3], &wa[ix4]);
            }
            break;
            case 4:
            {
                int ix2 = iw + ido;
                int ix3 = ix2 + ido;
                radb4_ps (ido, l1, in, out, &wa[iw], &wa[ix2], &wa[ix3]);
            }
            break;
            case 3:
            {
                int ix2 = iw + ido;
                radb3_ps (ido, l1, in, out, &wa[iw], &wa[ix2]);
            }
            break;
            case 2:
                radb2_ps (ido, l1, in, out, &wa[iw]);
                break;
            default:
                assert (0);
                break;
        }
        l1 = l2;
        iw += (ip - 1) * ido;

        if (out == work2)
        {
            out = work1;
            in = work2;
        }
        else
        {
            out = work2;
            in = work1;
        }
    }
    return in; /* this is in fact the output .. */
}

//====================================================================
/* [0 0 1 2 3 4 5 6 7 8] -> [0 8 7 6 5 4 3 2 1] */
static void reversed_copy (int N, const float32x4_t* in, int in_stride, float32x4_t* out)
{
    auto [g0, g1] = interleave2 (in[0], in[1]);
    in += in_stride;

    *--out = vcombine_f32 (vget_low_f32 (g1), vget_high_f32 (g0)); // [g0l, g0h], [g1l g1h] -> [g1l, g0h]
    int k;
    for (k = 1; k < N; ++k)
    {
        auto [h0, h1] = interleave2 (in[0], in[1]);
        in += in_stride;
        *--out = vcombine_f32 (vget_low_f32 (h0), vget_high_f32 (g1));
        *--out = vcombine_f32 (vget_low_f32 (h1), vget_high_f32 (h0));
        g1 = h1;
    }
    *--out = vcombine_f32 (vget_low_f32 (g0), vget_high_f32 (g1));
}

static void unreversed_copy (int N, const float32x4_t* in, float32x4_t* out, int out_stride)
{
    float32x4_t g0, g1, h0, h1;
    int k;
    g0 = g1 = in[0];
    ++in;
    for (k = 1; k < N; ++k)
    {
        h0 = *in++;
        h1 = *in++;
        g1 = vcombine_f32 (vget_low_f32 (h0), vget_high_f32 (g1));
        h0 = vcombine_f32 (vget_low_f32 (h1), vget_high_f32 (h0));
        std::tie (out[0], out[1]) = uninterleave2 (h0, g1);
        out += out_stride;
        g1 = h1;
    }
    h0 = *in++;
    h1 = g0;
    g1 = vcombine_f32 (vget_low_f32 (h0), vget_high_f32 (g1));
    h0 = vcombine_f32 (vget_low_f32 (h1), vget_high_f32 (h0));
    std::tie (out[0], out[1]) = uninterleave2 (h0, g1);
}

static void pffft_zreorder (FFT_Setup* setup, const float* in, float* out, fft_direction_t direction)
{
    int k, N = setup->N, Ncvec = setup->Ncvec;
    const auto* vin = (const float32x4_t*) in;
    auto* vout = (float32x4_t*) out;
    assert (in != out);
    if (setup->transform == FFT_REAL)
    {
        int dk = N / 32;
        if (direction == FFT_FORWARD)
        {
            for (k = 0; k < dk; ++k)
            {
                std::tie (vout[2 * (0 * dk + k) + 0], vout[2 * (0 * dk + k) + 1]) = interleave2 (vin[k * 8 + 0], vin[k * 8 + 1]);
                std::tie (vout[2 * (2 * dk + k) + 0], vout[2 * (2 * dk + k) + 1]) = interleave2 (vin[k * 8 + 4], vin[k * 8 + 5]);
            }
            reversed_copy (dk, vin + 2, 8, (float32x4_t*) (out + N / 2));
            reversed_copy (dk, vin + 6, 8, (float32x4_t*) (out + N));
        }
        else
        {
            for (k = 0; k < dk; ++k)
            {
                std::tie (vout[k * 8 + 0], vout[k * 8 + 1]) = uninterleave2 (vin[2 * (0 * dk + k) + 0], vin[2 * (0 * dk + k) + 1]);
                std::tie (vout[k * 8 + 4], vout[k * 8 + 5]) = uninterleave2 (vin[2 * (2 * dk + k) + 0], vin[2 * (2 * dk + k) + 1]);
            }
            unreversed_copy (dk, (float32x4_t*) (in + N / 4), (float32x4_t*) (out + N - 6 * SIMD_SZ), -8);
            unreversed_copy (dk, (float32x4_t*) (in + 3 * N / 4), (float32x4_t*) (out + N - 2 * SIMD_SZ), -8);
        }
    }
    else
    {
        if (direction == FFT_FORWARD)
        {
            for (k = 0; k < Ncvec; ++k)
            {
                int kk = (k / 4) + (k % 4) * (Ncvec / 4);
                std::tie (vout[kk * 2], vout[kk * 2 + 1]) = interleave2 (vin[k * 2], vin[k * 2 + 1]);
            }
        }
        else
        {
            for (k = 0; k < Ncvec; ++k)
            {
                int kk = (k / 4) + (k % 4) * (Ncvec / 4);
                std::tie (vout[k * 2], vout[k * 2 + 1]) = uninterleave2 (vin[kk * 2], vin[kk * 2 + 1]);
            }
        }
    }
}

//====================================================================
void pffft_transform_internal (FFT_Setup* setup, const float* finput, float* foutput, float32x4_t* scratch, fft_direction_t direction, int ordered)
{
    int k, Ncvec = setup->Ncvec;
    int nf_odd = (setup->ifac[1] & 1);

    // temporary buffer is allocated on the stack if the scratch pointer is NULL
    int stack_allocate = (scratch == nullptr ? Ncvec * 2 : 1);
    auto* scratch_on_stack = (float32x4_t*) alloca (stack_allocate * sizeof (float32x4_t));

    const auto* vinput = (const float32x4_t*) finput;
    auto* voutput = (float32x4_t*) foutput;
    float32x4_t* buff[2] = { voutput, scratch ? scratch : scratch_on_stack };
    int ib = (nf_odd ^ ordered ? 1 : 0);

    if (direction == FFT_FORWARD)
    {
        ib = ! ib;
        if (setup->transform == FFT_REAL)
        {
            ib = (rfftf1_ps (Ncvec * 2, vinput, buff[ib], buff[! ib], setup->twiddle, &setup->ifac[0]) == buff[0] ? 0 : 1);
            pffft_real_finalize (Ncvec, buff[ib], buff[! ib], (float32x4_t*) setup->e);
        }
        else
        {
            float32x4_t* tmp = buff[ib];
            for (k = 0; k < Ncvec; ++k)
            {
                std::tie (tmp[k * 2], tmp[k * 2 + 1]) = uninterleave2 (vinput[k * 2], vinput[k * 2 + 1]);
            }
            ib = (cfftf1_ps (Ncvec, buff[ib], buff[! ib], buff[ib], setup->twiddle, &setup->ifac[0], -1) == buff[0] ? 0 : 1);
            pffft_cplx_finalize (Ncvec, buff[ib], buff[! ib], (float32x4_t*) setup->e);
        }
        if (ordered)
        {
            pffft_zreorder (setup, (float*) buff[! ib], (float*) buff[ib], FFT_FORWARD);
        }
        else
            ib = ! ib;
    }
    else
    {
        if (vinput == buff[ib])
        {
            ib = ! ib; // may happen when finput == foutput
        }
        if (ordered)
        {
            pffft_zreorder (setup, (float*) vinput, (float*) buff[ib], FFT_BACKWARD);
            vinput = buff[ib];
            ib = ! ib;
        }
        if (setup->transform == FFT_REAL)
        {
            pffft_real_preprocess (Ncvec, vinput, buff[ib], (float32x4_t*) setup->e);
            ib = (rfftb1_ps (Ncvec * 2, buff[ib], buff[0], buff[1], setup->twiddle, &setup->ifac[0]) == buff[0] ? 0 : 1);
        }
        else
        {
            pffft_cplx_preprocess (Ncvec, vinput, buff[ib], (float32x4_t*) setup->e);
            ib = (cfftf1_ps (Ncvec, buff[ib], buff[0], buff[1], setup->twiddle, &setup->ifac[0], +1) == buff[0] ? 0 : 1);
            for (k = 0; k < Ncvec; ++k)
            {
                std::tie (buff[ib][k * 2], buff[ib][k * 2 + 1]) = interleave2 (buff[ib][k * 2], buff[ib][k * 2 + 1]);
            }
        }
    }

    if (buff[ib] != voutput)
    {
        /* extra copy required -- this situation should only happen when finput == foutput */
        assert (finput == foutput);
        for (k = 0; k < Ncvec; ++k)
        {
            float32x4_t a = buff[ib][2 * k], b = buff[ib][2 * k + 1];
            voutput[2 * k] = a;
            voutput[2 * k + 1] = b;
        }
        ib = ! ib;
    }
    assert (buff[ib] == voutput);
}

void pffft_convolve_internal (FFT_Setup* setup, const float* a, const float* b, float* ab, float scaling)
{
    int Ncvec = setup->Ncvec;
    auto* va = (const float32x4_t*) a;
    auto* vb = (const float32x4_t*) b;
    auto* vab = (float32x4_t*) ab;

    float ar0, ai0, br0, bi0, abr0, abi0;
    const auto vscal = vld1q_dup_f32 (&scaling);
    int i;

    ar0 = reinterpret_cast<const float*> (&va[0])[0];
    ai0 = reinterpret_cast<const float*> (&va[1])[0];
    br0 = reinterpret_cast<const float*> (&vb[0])[0];
    bi0 = reinterpret_cast<const float*> (&vb[1])[0];
    abr0 = reinterpret_cast<const float*> (&vab[0])[0];
    abi0 = reinterpret_cast<const float*> (&vab[1])[0];

    for (i = 0; i < Ncvec; i += 2)
    {
        float32x4_t ar, ai, br, bi;
        ar = va[2 * i + 0];
        ai = va[2 * i + 1];
        br = vb[2 * i + 0];
        bi = vb[2 * i + 1];
        std::tie (ar, ai) = cplx_mul_v (ar, ai, br, bi);
        vab[2 * i + 0] = vmlaq_f32 (vab[2 * i + 0], ar, vscal);
        vab[2 * i + 1] = vmlaq_f32 (vab[2 * i + 1], ai, vscal);
        ar = va[2 * i + 2];
        ai = va[2 * i + 3];
        br = vb[2 * i + 2];
        bi = vb[2 * i + 3];
        std::tie (ar, ai) = cplx_mul_v (ar, ai, br, bi);
        vab[2 * i + 2] = vmlaq_f32 (vab[2 * i + 2], ar, vscal);
        vab[2 * i + 3] = vmlaq_f32 (vab[2 * i + 3], ai, vscal);
    }

    if (setup->transform == FFT_REAL)
    {
        reinterpret_cast<float*> (&vab[0])[0] = abr0 + ar0 * br0 * scaling;
        reinterpret_cast<float*> (&vab[1])[0] = abi0 + ai0 * bi0 * scaling;
    }
}

void fft_accumulate_internal (const float* a, const float* b, float* ab, int N)
{
    assert (N % (SIMD_SZ * 2) == 0);
    const auto Ncvec = N / (int) SIMD_SZ;
    auto* va = (const float32x4_t*) a;
    auto* vb = (const float32x4_t*) b;
    auto* vab = (float32x4_t*) ab;

    for (int i = 0; i < Ncvec; i += 2)
    {
        vab[i] = vaddq_f32 (va[i], vb[i]);
        vab[i + 1] = vaddq_f32 (va[i + 1], vb[i + 1]);
    }
}
} // namespace chowdsp::fft::neon
