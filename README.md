# chowdsp_fft

[![Test](https://github.com/Chowdhury-DSP/chowdsp_fft/actions/workflows/test.yml/badge.svg)](https://github.com/Chowdhury-DSP/chowdsp_fft/actions/workflows/test.yml)
[![Bench](https://github.com/Chowdhury-DSP/chowdsp_fft/actions/workflows/bench.yml/badge.svg)](https://github.com/Chowdhury-DSP/chowdsp_fft/actions/workflows/bench.yml)
[![codecov](https://codecov.io/gh/Chowdhury-DSP/chowdsp_fft/graph/badge.svg?token=A5BJ6CS859)](https://codecov.io/gh/Chowdhury-DSP/chowdsp_fft)

`chowdsp_fft` is a fork of the [`pffft`](https://bitbucket.org/jpommier/pffft/src/master/)
library for computing Fast Fourier Transforms. This fork adds a couple
of optimizations, most importantly the ability to use AVX intrinsics
for single-precision FFTs on compatible platforms. The library also
contains some methods which may be useful for computing convolutions
using frequency-domain multiplication.

## Disclaimer

This library has only been tested for power-of-2 FFT sizes.
My understanding is that `pffft` supports some additional FFT
sizes. I imagine many of those sizes will work with this
implementation as well, but please test first and report back!

## Using `chowdsp_fft` with AVX SIMD Intrinsics

The provided CMake configuration will compile `chowdsp_fft` as static
library containing both the SSE and AVX implementations (so long as
the compiler supports compiling with AVX). From there, when the
user creates an FFT "setup" object, an option is provided to use
the AVX implementation if it's available on the host system. When
this option is enabled, several things happen:
- When compiling for a platform that doesn't support AVX (e.g. an ARM CPU), the option is ignored.
- When running on a compatible platform, the library will first check if the host CPU supports AVX intrinsics. If not, the library will fall back to the SSE implementation.
- Next the library will check if the requested FFT size is large enough for the AVX implementation to work. Smaller FFT sizes will also fall back to the SSE implementation.

## Using with JUCE

If you're using this library as part of a CMake project that
uses JUCE, the CMake configuration for this library will also
generate a JUCE module that you can link to `chowdsp::chowdsp_fft_juce`,
which allows the FFT implementations to be used by `juce::dsp::FFT`.

In order to avoid symbols clashing, you must make sure that the
`juce_dsp` module is not being compiled alongside `chowdsp_fft_juce`.
If you are in control of all the modules being used, this should be
pretty easy to do, but if you're using third-party modules that require
`juce_dsp` then I'm not sure what you should do. Since this module will
compile the `juce_dsp` code for you, you may want to manually define
`JUCE_MODULE_AVAILABLE_juce_dsp=1` so that other modules in your system
are aware.

## License

```
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
```
