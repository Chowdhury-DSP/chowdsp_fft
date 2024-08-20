# chowdsp_fft

`chowdsp_fft` is a fork of the [`pffft`](https://bitbucket.org/jpommier/pffft/src/master/)
library for computing Fast Fourier Transforms. This fork adds a couple
of optimizations, most importantly the ability to use AVX intrinsics
for single-precision FFTs on compatible platforms.

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

I guess this code is under the same license as `pffft`... does
anyone know what that license is?
