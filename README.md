# chowdsp_fft — radix-5 correctness patch

This repository is a fork of [Chowdhury-DSP/chowdsp_fft](https://github.com/Chowdhury-DSP/chowdsp_fft).

It contains a small correctness fix for the forward real radix-5 transform path, affecting FFT sizes that contain repeated factors of 5 (`5^2` or higher).

## The bug

The SIMD radix-5 forward implementation used the operands of a complex conjugate multiply in the wrong order.

The affected operation should be equivalent to:

```text
twiddle * conj(data)
```

but the ChowDSP implementation evaluated the equivalent of:

```text
data * conj(twiddle)
```

These expressions have the same real component, but opposite-sign imaginary components.

For FFT sizes containing only one factor of 5, the affected twiddle path is not exercised. With two or more factors of 5, the erroneous path is reached.

Examples of affected sizes included:

```text
800
1600
2400
3200
6400
...
```

## Fix

The correction is applied to:

- `simd/chowdsp_fft_impl_sse.cpp`
- `simd/chowdsp_fft_impl_avx.cpp`
- `simd/chowdsp_fft_impl_neon.cpp`

The SSE and AVX2/FMA implementations have been runtime-tested.

The NEON implementation receives the equivalent algebraic correction, but has **not yet been runtime-tested on ARM**.

No changes were required to the common FFT setup/factorisation code.

## Validation

The patched x86 implementation was tested across **131 valid real PFFFT transform sizes from 256 through 131072**.

Each size was checked using:

- ChowDSP automatic backend
- ChowDSP forced-SSE backend
- maintained `marton78/pffft`
- double-precision PocketFFT as an independent numerical reference

The test suite independently validates:

- forward FFT relative-L2 error
- forward FFT normalized maximum-bin error
- forward → inverse round-trip error
- inverse FFT relative-L2 error using an independently generated spectrum
- inverse FFT normalized maximum error

### Result

```text
ChowDSP automatic: 131 / 131 PASS
ChowDSP forced SSE: 131 / 131 PASS
marton78 PFFFT:     131 / 131 PASS
```

All previously failing sizes containing `5^2` or greater pass after the patch.

Typical worst-case numerical errors across the complete sweep remain in the expected single-precision range, approximately `2e-7` relative-L2 and below `1e-6` for the tested maximum/round-trip metrics.

## Performance

The patched ChowDSP AVX2/FMA backend retains its performance advantage over the SSE implementation.

On the test system, across sizes where the AVX2/FMA path was available, the geometric-mean speedup over forced SSE was approximately:

```text
1.87x
```

For common power-of-two sizes, measured forward+inverse pair timings showed roughly `1.5x–2.0x` improvement over the SSE path.

Performance varies by CPU, compiler, FFT size, cache state, and build configuration.

## Scope

This patch is intentionally narrow. It corrects the repeated-radix-5 forward-transform issue and does not attempt to redesign the FFT implementation.

The validation described above covers the tested real-transform x86 paths. It should not be interpreted as exhaustive validation of every API mode, compiler, architecture, alignment configuration, or platform.

## Upstream

Original project:

https://github.com/Chowdhury-DSP/chowdsp_fft

Maintained PFFFT implementation used for cross-checking:

https://github.com/marton78/pffft

PocketFFT was used as an independent numerical reference.

## License

This fork retains the original project copyright notices and license terms.

See the source files and upstream repository for the applicable license.
