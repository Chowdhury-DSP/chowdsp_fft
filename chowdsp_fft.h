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

#pragma once

#ifdef __cplusplus
#include <cstddef>

extern "C"
{
namespace chowdsp::fft
{
#else
#include <stddef.h>
#include <stdbool.h>
#endif

/* direction of the transform */
typedef enum
{
    FFT_FORWARD,
    FFT_BACKWARD
} fft_direction_t;

/* type of transform */
typedef enum
{
    FFT_REAL,
    FFT_COMPLEX
} fft_transform_t;

/*
  prepare for performing transforms of size N -- the returned
  FFT_Setup structure is read-only so it can safely be shared by
  multiple concurrent threads.
*/
void* fft_new_setup (int N, fft_transform_t transform, bool use_avx_if_available
#ifdef __cplusplus
                                                       = true
#endif
);
void fft_destroy_setup (void*);

/*
   Perform a Fourier transform , The z-domain data is stored as
   interleaved complex numbers.

   Transforms are not scaled: PFFFT_BACKWARD(PFFFT_FORWARD(x)) = N*x.
   Typically you will want to scale the backward transform by 1/N.

   The 'work' pointer should point to an area of N (2*N for complex
   fft) floats, properly aligned. If 'work' is NULL, then stack will
   be used instead (this is probably the best strategy for small
   FFTs, say for N < 16384).

   input and output may alias.
*/
void fft_transform (void* setup, const float* input, float* output, float* work, fft_direction_t direction);

void* aligned_malloc (size_t nb_bytes);
void aligned_free (void*);

#ifdef __cplusplus
}
} // namespace chowdsp::fft
#endif
