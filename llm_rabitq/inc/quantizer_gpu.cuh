/*
 * Minimal header for standalone quantization support.
 * Only exposes get_const_scaling_factors_fully_gpu (used by StandaloneQuantizerGPU).
 * Stripped from the full DataQuantizerGPU class (no indexing code).
 */

#ifndef RABITQ_STANDALONE_QUANTIZER_GPU_CUH
#define RABITQ_STANDALONE_QUANTIZER_GPU_CUH

#include <cstddef>

class DataQuantizerGPU {
public:
    static float get_const_scaling_factors_fully_gpu(size_t dim, size_t ex_bits);
};

#endif
