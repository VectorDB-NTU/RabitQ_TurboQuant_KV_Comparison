//
// Created by Stardust on 3/24/25.
//

#ifndef EXRABITQ_ROTATOR_GPU_CUH
#define EXRABITQ_ROTATOR_GPU_CUH

#include <cstdint>
#include <fstream>
#include <cublas_v2.h>
#include "defines.hpp"
#include "utils/utils_cuda.cuh"

/// Rotator type tag — persisted in the save file so load() can reconstruct
/// the correct implementation.
enum class RotatorType : uint8_t {
    Matrix  = 0,   // Full D×D rotation matrix via cuBLAS sgemm, O(N*D^2)
    FhtKac  = 1,   // Fast Hadamard Transform + Kac's walk, O(N*D*logD)
};

/// Unified GPU rotator supporting both matrix multiplication and FHT+Kac.
///
/// The rotator type is chosen at construction time and persisted on save/load.
/// All callers use the same interface regardless of the underlying implementation.
class RotatorGPU {
private:
    RotatorType type_;
    size_t D;              // Padded dimension (multiple of 64)

    // --- Matrix rotator members ---
    float* d_P = nullptr;           // Device: D×D rotation matrix
    cublasHandle_t m_handle{};

    // --- FhtKac rotator members ---
    size_t dim_ = 0;                // Original (unpadded) dimension
    size_t trunc_dim_ = 0;          // 1 << floor_log2(dim), largest power-of-2 <= dim
    float fac_ = 0;                 // 1/sqrt(trunc_dim)
    int log_N_ = 0;                 // log2(trunc_dim)
    uint8_t* d_flip_ = nullptr;     // Device: 4 * D/8 bytes of random sign bits

    // Internal helpers
    void init_matrix(uint32_t dim);
    void init_fht_kac(uint32_t dim);
    void free_resources();

public:
    /// Construct a rotator of the given type.
    /// @param dim  Original vector dimension. Padded to multiple of 64.
    /// @param type Rotator implementation to use.
    explicit RotatorGPU(uint32_t dim, RotatorType type = RotatorType::FhtKac);

    /// Default constructor (uninitialized).
    RotatorGPU() : type_(RotatorType::FhtKac), D(0) {}

    ~RotatorGPU();

    RotatorGPU& operator=(const RotatorGPU& other);

    /// @return Padded dimension.
    size_t size() const;

    /// @return The rotator type.
    RotatorType rotator_type() const { return type_; }

    /// Save rotator to file. Format: [uint8_t type_tag] [type-specific data].
    void save(std::ofstream& output) const;

    /// Load rotator from file. Reads the type tag, then loads the appropriate data.
    /// The rotator must already be constructed with the correct dimension (via
    /// constructor or assignment) before calling load().
    void load(std::ifstream& input);

    /// Rotate N vectors of D floats.
    /// @param d_A      Input:  N × D matrix on device (row-major).
    /// @param d_RAND_A Output: N × D matrix on device (row-major).
    /// @param N        Number of vectors.
    void rotate(const float* d_A, float* d_RAND_A, size_t N) const;
};

#endif //EXRABITQ_ROTATOR_GPU_CUH
