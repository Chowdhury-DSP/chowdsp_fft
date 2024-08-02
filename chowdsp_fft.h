#pragma once

#include <cstddef>

namespace chowdsp::fft
{
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
void* fft_new_setup (int N, fft_transform_t transform);
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
} // namespace chowdsp::fft
