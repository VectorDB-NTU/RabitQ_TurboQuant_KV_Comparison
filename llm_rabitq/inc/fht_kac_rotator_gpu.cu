//
// Created by Stardust on 4/2/26.
// GPU FHT + Kac's Walk Rotator — implementation.
//

#include "fht_kac_rotator_gpu.cuh"
#include "fht_cuda.cuh"
#include <random>
#include <cmath>

// ============================================================================
// Helper functions
// ============================================================================

static inline size_t floor_log2(size_t x) {
    size_t r = 0;
    while (x >>= 1) { ++r; }
    return r;
}

static inline size_t round_up_to_multiple(size_t val, size_t mult) {
    return ((val + mult - 1) / mult) * mult;
}

// ============================================================================
// FhtKacRotatorGPU implementation
// ============================================================================

FhtKacRotatorGPU::FhtKacRotatorGPU(uint32_t dim) {
    dim_ = dim;
    padded_dim_ = round_up_to_multiple(dim, 64);

    size_t bottom_log = floor_log2(dim);
    trunc_dim_ = 1ULL << bottom_log;
    log_N_ = static_cast<int>(bottom_log);
    fac_ = 1.0f / std::sqrt(static_cast<float>(trunc_dim_));

    // Generate random sign-flip bits on host
    size_t flip_bytes = 4 * padded_dim_ / 8;  // 4 rounds, padded_dim bits per round
    std::vector<uint8_t> h_flip(flip_bytes);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto& b : h_flip) {
        b = static_cast<uint8_t>(dist(gen));
    }

    CUDA_CHECK(cudaMalloc(&d_flip_, flip_bytes));
    CUDA_CHECK(cudaMemcpy(d_flip_, h_flip.data(), flip_bytes, cudaMemcpyHostToDevice));
}

FhtKacRotatorGPU::~FhtKacRotatorGPU() {
    if (d_flip_) {
        cudaFree(d_flip_);
    }
}

FhtKacRotatorGPU& FhtKacRotatorGPU::operator=(const FhtKacRotatorGPU& other) {
    if (this != &other) {
        dim_ = other.dim_;
        padded_dim_ = other.padded_dim_;
        trunc_dim_ = other.trunc_dim_;
        fac_ = other.fac_;
        log_N_ = other.log_N_;

        if (d_flip_) { cudaFree(d_flip_); }

        size_t flip_bytes = 4 * padded_dim_ / 8;
        CUDA_CHECK(cudaMalloc(&d_flip_, flip_bytes));
        CUDA_CHECK(cudaMemcpy(d_flip_, other.d_flip_, flip_bytes, cudaMemcpyDeviceToDevice));
    }
    return *this;
}

size_t FhtKacRotatorGPU::size() const {
    return padded_dim_;
}

void FhtKacRotatorGPU::load(std::ifstream& input) {
    size_t flip_bytes = 4 * padded_dim_ / 8;
    std::vector<uint8_t> h_flip(flip_bytes);
    input.read(reinterpret_cast<char*>(h_flip.data()), static_cast<long>(flip_bytes));
    CUDA_CHECK(cudaMemcpy(d_flip_, h_flip.data(), flip_bytes, cudaMemcpyHostToDevice));
}

void FhtKacRotatorGPU::save(std::ofstream& output) const {
    size_t flip_bytes = 4 * padded_dim_ / 8;
    std::vector<uint8_t> h_flip(flip_bytes);
    CUDA_CHECK(cudaMemcpy(h_flip.data(), d_flip_, flip_bytes, cudaMemcpyDeviceToHost));
    output.write(reinterpret_cast<const char*>(h_flip.data()), static_cast<long>(flip_bytes));
}

void FhtKacRotatorGPU::rotate(const float* d_A, float* d_RAND_A, size_t N) const {
    cudaStream_t stream = 0;

    if (trunc_dim_ == padded_dim_) {
        // ============================================================
        // Power-of-2 path: single fused kernel
        // All 4 rounds of (sign_flip → FHT) run in registers.
        // Scale deferred to end: total_scale = fac^4.
        // ============================================================
        float total_scale = fac_ * fac_ * fac_ * fac_;
        fht::dispatch_fused_rotate(d_A, d_RAND_A, d_flip_,
                                    static_cast<int>(N), log_N_,
                                    total_scale, stream);
    } else {
        // ============================================================
        // Non-power-of-2 path: single fused kernel
        // All 4 rounds of (sign_flip → FHT → kacs_walk) via shared memory.
        // fac applied per-round (can't defer — kacs_walk mixes scales).
        // Only the final 0.25 is deferred.
        // ============================================================
        fht::dispatch_fused_rotate_nonpow2(d_A, d_RAND_A, d_flip_,
                                            static_cast<int>(N), log_N_,
                                            static_cast<int>(padded_dim_),
                                            fac_, 0.25f, stream);
    }
}
