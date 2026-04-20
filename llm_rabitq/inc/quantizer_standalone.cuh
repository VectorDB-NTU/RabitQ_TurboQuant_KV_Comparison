//
// Standalone GPU quantizer for RaBitQ.
//
// Mirrors the CPU RaBitQ-Library API:
//   quantize_scalar → codes + (delta, vl) per vector
//   quantize_full   → codes + (f_add, f_rescale, f_error) per vector
//
// Both have two overloads: with centroid and without (zero centroid default).
// Operates on batches of N vectors. Rotation is handled internally.
//

#ifndef GBITQ_QUANTIZER_STANDALONE_CUH
#define GBITQ_QUANTIZER_STANDALONE_CUH

#include <cstdint>
#include <cstddef>
#include "rotator_gpu.cuh"

class StandaloneQuantizerGPU {
private:
    uint32_t dim_;            // Original dimension
    uint32_t padded_dim_;     // Padded to multiple of 64
    size_t total_bits_;       // Total bits per dimension (1 sign + ex_bits)
    size_t ex_bits_;          // Extended bits = total_bits - 1
    RotatorGPU rotator_;
    float const_scaling_factor_;  // Precomputed rescale factor for fast path
    bool use_fast_quantize_;      // Use const_scaling_factor vs per-vector search

    // Internal: rotate + subtract centroid → residuals on device
    // d_data: N × dim (unpadded), d_centroid: dim (unpadded)
    // d_residual: N × padded_dim (output, allocated by caller)
    void compute_residuals(const float* d_data, const float* d_centroid,
                           size_t N, float* d_residual) const;

public:
    /// @param dim        Original vector dimension
    /// @param total_bits Bits per dimension (1..9). ex_bits = total_bits - 1.
    /// @param rota_type  Rotator type (Matrix or FhtKac)
    /// @param fast       Use precomputed const_scaling_factor (fast) or per-vector search (slow)
    StandaloneQuantizerGPU(uint32_t dim, size_t total_bits,
                           RotatorType rota_type = RotatorType::FhtKac,
                           bool fast = true);

    ~StandaloneQuantizerGPU() = default;

    uint32_t padded_dim() const { return padded_dim_; }
    size_t total_bits() const { return total_bits_; }
    size_t ex_bits() const { return ex_bits_; }
    const RotatorGPU& rotator() const { return rotator_; }

    // =========================================================================
    // quantize_scalar: produces per-dimension total_code + (delta, vl) per vector
    //
    // Reconstruction: reconstructed[i] = total_code[i] * delta + vl
    //
    // d_data:       [in]  N × dim floats on device (unpadded, row-major)
    // d_centroid:   [in]  dim floats on device (or nullptr for zero centroid)
    // N:            [in]  number of vectors
    // d_total_code: [out] N × padded_dim on device (uint8_t for 1-8 bits, uint16_t for 9 bits)
    // d_delta:      [out] N floats on device
    // d_vl:         [out] N floats on device
    // =========================================================================

    // uint16_t overloads (supports all bit widths 1-9)
    void quantize_scalar(const float* d_data, size_t N,
                         uint16_t* d_total_code, float* d_delta, float* d_vl) const;
    void quantize_scalar(const float* d_data, const float* d_centroid, size_t N,
                         uint16_t* d_total_code, float* d_delta, float* d_vl) const;

    // uint8_t overloads (efficient for 1-8 bits; asserts total_bits <= 8)
    void quantize_scalar(const float* d_data, size_t N,
                         uint8_t* d_total_code, float* d_delta, float* d_vl) const;
    void quantize_scalar(const float* d_data, const float* d_centroid, size_t N,
                         uint8_t* d_total_code, float* d_delta, float* d_vl) const;

    // =========================================================================
    // quantize_full: produces per-dimension total_code + (f_add, f_rescale, f_error) per vector
    //
    // Distance estimation factors for approximate nearest neighbor search.
    //
    // d_data:       [in]  N × dim floats on device (unpadded, row-major)
    // d_centroid:   [in]  dim floats on device (or nullptr for zero centroid)
    // N:            [in]  number of vectors
    // d_total_code: [out] N × padded_dim on device (uint8_t for 1-8 bits, uint16_t for 9 bits)
    // d_factors:    [out] N × 3 floats on device (f_add, f_rescale, f_error per vector)
    // =========================================================================

    // uint16_t overloads
    void quantize_full(const float* d_data, size_t N,
                       uint16_t* d_total_code, float* d_factors) const;
    void quantize_full(const float* d_data, const float* d_centroid, size_t N,
                       uint16_t* d_total_code, float* d_factors) const;

    // uint8_t overloads (efficient for 1-8 bits; asserts total_bits <= 8)
    void quantize_full(const float* d_data, size_t N,
                       uint8_t* d_total_code, float* d_factors) const;
    void quantize_full(const float* d_data, const float* d_centroid, size_t N,
                       uint8_t* d_total_code, float* d_factors) const;
};

// ============================================================================
// Free functions for direct quantization on pre-computed residuals (no rotation).
// Useful for verification against CPU implementations.
// ============================================================================

/// Quantize pre-computed residuals → codes + delta + vl.
/// d_residuals must be N × padded_dim on device (already rotated & centroid-subtracted).
/// delta_mode: 0=RECONSTRUCTION, 1=UNBIASED, 2=PLAIN
/// Uses fused warp-cooperative rescale + quantize kernel.
void standalone_quantize_fused_on_residuals(
    const float* d_residuals, size_t N, size_t padded_dim,
    size_t ex_bits, float const_scaling_factor, bool use_fast,
    uint16_t* d_total_code, float* d_delta, float* d_vl, int delta_mode = 0);
void standalone_quantize_fused_on_residuals(
    const float* d_residuals, size_t N, size_t padded_dim,
    size_t ex_bits, float const_scaling_factor, bool use_fast,
    uint8_t* d_total_code, float* d_delta, float* d_vl, int delta_mode = 0);

/// Quantize pre-computed residuals → codes + full factors (f_add, f_rescale, f_error).
/// d_centroid: padded_dim floats (rotated centroid, or zero for no centroid).
void standalone_quantize_full_on_residuals(
    const float* d_residuals, const float* d_centroid,
    size_t N, size_t padded_dim,
    size_t ex_bits, float const_scaling_factor, bool use_fast,
    uint16_t* d_total_code, float* d_factors);

#endif // GBITQ_QUANTIZER_STANDALONE_CUH
