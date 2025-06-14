#include "chowdsp_fft_juce.h"

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_dsp/juce_dsp.h>
#undef JUCE_DSP_H_INCLUDED // HACK!
#include <juce_dsp/juce_dsp.cpp> // NOLINT

// reference: https://github.com/soundradix/JUCE/blob/ad6afe48faecc02e21a774d1bf643cb30852cd4b/modules/juce_dsp/frequency/juce_FFT.cpp#L946

namespace juce::dsp
{

class ChowDSP_FFT : public FFT::Instance
{
public:
    static constexpr auto priority = 7;

    static ChowDSP_FFT* create (const int order)
    {
        if (order < 5)
        {
            // Not supported according to PFFFT's docs:
            //
            // > supports only transforms for inputs of length N of the form
            // > N=(2^a)*(3^b)*(5^c), a >= 5, b >=0, c >= 0
            jassertfalse;
            return nullptr;
        }
        return new ChowDSP_FFT (order);
    }

    void perform (const Complex<float>* input, Complex<float>* output, bool inverse) const noexcept override
    {
        memcpy (inout_buffer, input, sizeof (float) << (order + 1));

        chowdsp::fft::fft_transform (setup_complex,
                                     inout_buffer,
                                     inout_buffer,
                                     work_buffer,
                                     inverse ? chowdsp::fft::FFT_BACKWARD : chowdsp::fft::FFT_FORWARD);

        memcpy (output, inout_buffer, sizeof (float) << (order + 1));

        if (inverse)
            FloatVectorOperations::multiply ((float*) output, 1.0f / float (1 << order), 1 << (order + 1));
    }

    void performRealOnlyForwardTransform (float* inoutData, bool ignoreNegativeFreqs) const noexcept override
    {
        memcpy (inout_buffer, inoutData, sizeof (float) << (order));
        chowdsp::fft::fft_transform (setup_real,
                                     inout_buffer,
                                     inout_buffer,
                                     work_buffer,
                                     chowdsp::fft::FFT_FORWARD);
        memcpy (inoutData, inout_buffer, sizeof (float) << (order));

        const int nyquist = 1 << (order - 1);
        inoutData[2 * nyquist] = inoutData[1];
        inoutData[2 * nyquist + 1] = 0.0f;
        inoutData[1] = 0.0f;
        if (! ignoreNegativeFreqs)
        {
            // Silly anti-feature to produce redundant negative freqs!
            auto out = (Complex<float>*) inoutData;
            for (int i = 1; i < nyquist; ++i)
                out[nyquist + i] = std::conj (out[nyquist - i]);
        }
    }

    void performRealOnlyInverseTransform (float* inoutData) const noexcept override
    {
        JUCE_BEGIN_IGNORE_WARNINGS_MSVC (4334)
        inoutData[1] = inoutData[1 << order];
        JUCE_END_IGNORE_WARNINGS_MSVC

        memcpy (inout_buffer, inoutData, sizeof (float) << (order));
        chowdsp::fft::fft_transform (setup_real,
                                     inout_buffer,
                                     inout_buffer,
                                     work_buffer,
                                     chowdsp::fft::FFT_BACKWARD);
        memcpy (inoutData, inout_buffer, sizeof (float) << (order));

        FloatVectorOperations::multiply (inoutData, 1.0f / float (1 << order), 1 << order);
    }

    ~ChowDSP_FFT() override
    {
        chowdsp::fft::fft_destroy_setup (setup_real);
        chowdsp::fft::fft_destroy_setup (setup_complex);
        chowdsp::fft::aligned_free (inout_buffer);
        chowdsp::fft::aligned_free (work_buffer);
    }

private:
    explicit ChowDSP_FFT (int order_) : order (order_)
    {
        setup_real = chowdsp::fft::fft_new_setup (1 << order, chowdsp::fft::FFT_REAL);
        jassert (setup_real != nullptr);

        setup_complex = chowdsp::fft::fft_new_setup (1 << order, chowdsp::fft::FFT_COMPLEX);
        jassert (setup_complex != nullptr);

        inout_buffer = (float*) chowdsp::fft::aligned_malloc (sizeof (float) << (order + 1));
        jassert (inout_buffer != nullptr);

        work_buffer = (float*) chowdsp::fft::aligned_malloc (sizeof (float) << (order + 1));
        jassert (work_buffer != nullptr);
    }

    void* setup_real = nullptr;
    void* setup_complex = nullptr;
    float* inout_buffer = nullptr;
    float* work_buffer = nullptr;
    int order {};
};

FFT::EngineImpl<ChowDSP_FFT> chowdsp_fft_impl;
} // namespace juce::dsp
