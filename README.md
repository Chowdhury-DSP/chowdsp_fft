# chowdsp_fft

`chowdsp_fft` is a fork of the [`pffft`](https://bitbucket.org/jpommier/pffft/src/master/)
library for computing Fast Fourier Transforms. This fork adds a couple
of optimizations, most importantly the ability to use AVX intrinsics
for single-precision FFTs on compatible platforms.

## Using `chowdsp_fft` with AVX

The provided CMake configuration will compile `chowdsp_fft` as static
library containing both the SSE and AVX implementations (so long as
the compiler supports compiling with AVX). From there, when the
user creates an FFT "setup" object, an option is provided to use
the AVX implementation if it's available on the host system. When
this option is enabled, several things happen:
- When compiling for a platform that doesn't support AVX (e.g. an ARM CPU), the option is ignored
- When running on a compatible platform, the library will first check if the host CPU supports AVX intrinsics. If not, the library will fall back to the SSE implementation.
- Next the library will check if the requested FFT size is large enough for the AVX implementation to work. Smaller FFT sizes will also fall back to the SSE implementation.
