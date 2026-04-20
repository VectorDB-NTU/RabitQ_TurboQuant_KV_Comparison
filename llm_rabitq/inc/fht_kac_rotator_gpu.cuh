//
// Created by Stardust on 4/2/26.
// GPU FHT + Kac's Walk Rotator — alternative to RotatorGPU (matrix multiply).
//

#ifndef GBITQ_FHT_KAC_ROTATOR_GPU_CUH
#define GBITQ_FHT_KAC_ROTATOR_GPU_CUH

#include <cstdint>
#include <fstream>
#include "defines.hpp"
#include "utils/utils_cuda.cuh"

/// FhtKacRotatorGPU implements random rotation via 4 rounds of
/// (random sign flip -> Fast Hadamard Transform [-> Kac's walk]).
///
/// Compared to RotatorGPU (cuBLAS sgemm, O(N*D^2)):
///   - Memory: O(D) instead of O(D^2)
///   - Compute: O(N*D*log D) instead of O(N*D^2)
///   - No cuBLAS dependency
///
/// The algorithm matches the CPU FhtKacRotator in RaBitQ-Library.
/// Supports both power-of-2 and non-power-of-2 padded dimensions.
class FhtKacRotatorGPU {
private:
    size_t dim_;          // Original dimension
    size_t padded_dim_;   // round_up_to_multiple(dim, 64)
    size_t trunc_dim_;    // 1 << floor_log2(dim), largest power-of-2 <= dim
    float fac_;           // 1.0f / sqrt(trunc_dim)
    int log_N_;           // log2(trunc_dim), for FHT kernel dispatch
    uint8_t* d_flip_;     // Device pointer: 4 * padded_dim/8 bytes of random sign bits

public:
    /// Construct with random sign-flip bits.
    /// @param dim Original vector dimension. Padded dimension = round_up_to_multiple(dim, 64).
    explicit FhtKacRotatorGPU(uint32_t dim);

    /// Default constructor (uninitialized).
    explicit FhtKacRotatorGPU() : dim_(0), padded_dim_(0), trunc_dim_(0),
                                   fac_(0), log_N_(0), d_flip_(nullptr) {}

    ~FhtKacRotatorGPU();

    FhtKacRotatorGPU& operator=(const FhtKacRotatorGPU& other);

    /// @return Padded dimension.
    size_t size() const;

    /// Load sign-flip bits from file (4 * padded_dim/8 bytes).
    void load(std::ifstream& input);

    /// Save sign-flip bits to file.
    void save(std::ofstream& output) const;

    /// Rotate N vectors of padded_dim floats.
    /// @param d_A     Input:  N x padded_dim matrix on device (row-major).
    /// @param d_RAND_A Output: N x padded_dim matrix on device (row-major).
    /// @param N       Number of vectors.
    void rotate(const float* d_A, float* d_RAND_A, size_t N) const;
};

#endif // GBITQ_FHT_KAC_ROTATOR_GPU_CUH
