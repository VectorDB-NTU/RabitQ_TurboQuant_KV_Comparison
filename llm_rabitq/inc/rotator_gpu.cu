//
// Created by Stardust on 3/24/25.
//

#include "rotator_gpu.cuh"
#include "fht_cuda.cuh"
#include "third/Eigen/Dense"
#include <random>
#include <cmath>

// ============================================================================
// Helpers
// ============================================================================

static inline size_t rd_up(size_t dim, size_t mult) {
    return ((dim + mult - 1) / mult) * mult;
}

static inline size_t floor_log2(size_t x) {
    size_t r = 0;
    while (x >>= 1) { ++r; }
    return r;
}

// ============================================================================
// Matrix rotator internals
// ============================================================================

void RotatorGPU::init_matrix(uint32_t dim) {
    D = rd_up(dim, 64);

#ifdef DEBUG_BATCH_CONSTRUCT
    srand(1);
#endif
    FloatRowMat RAND = random_gaussian_matrix<float>(D, D);
    Eigen::HouseholderQR<FloatRowMat> qr(RAND);
    FloatRowMat Q = qr.householderQ();
    FloatRowMat P = Q.transpose();

    float* hostP = new float[D * D];
    std::memcpy(hostP, P.data(), sizeof(float) * D * D);
    CUDA_CHECK(cudaMalloc(&d_P, sizeof(float) * D * D));
    CUDA_CHECK(cudaMemcpy(d_P, hostP, sizeof(float) * D * D, cudaMemcpyHostToDevice));
    delete[] hostP;

    cublasCreate(&m_handle);
}

// ============================================================================
// FhtKac rotator internals
// ============================================================================

void RotatorGPU::init_fht_kac(uint32_t dim) {
    dim_ = dim;
    D = rd_up(dim, 64);

    size_t bottom_log = floor_log2(dim);
    trunc_dim_ = 1ULL << bottom_log;
    log_N_ = static_cast<int>(bottom_log);
    fac_ = 1.0f / std::sqrt(static_cast<float>(trunc_dim_));

    size_t flip_bytes = 4 * D / 8;
    std::vector<uint8_t> h_flip(flip_bytes);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto& b : h_flip)
        b = static_cast<uint8_t>(dist(gen));

    CUDA_CHECK(cudaMalloc(&d_flip_, flip_bytes));
    CUDA_CHECK(cudaMemcpy(d_flip_, h_flip.data(), flip_bytes, cudaMemcpyHostToDevice));
}

// ============================================================================
// Resource cleanup
// ============================================================================

void RotatorGPU::free_resources() {
    if (d_P) { cudaFree(d_P); d_P = nullptr; }
    if (d_flip_) { cudaFree(d_flip_); d_flip_ = nullptr; }
    if (type_ == RotatorType::Matrix) {
        cublasDestroy(m_handle);
        m_handle = {};
    }
}

// ============================================================================
// Public interface
// ============================================================================

RotatorGPU::RotatorGPU(uint32_t dim, RotatorType type) : type_(type) {
    if (type == RotatorType::Matrix) {
        init_matrix(dim);
    } else {
        init_fht_kac(dim);
    }
}

RotatorGPU::~RotatorGPU() {
    free_resources();
}

RotatorGPU& RotatorGPU::operator=(const RotatorGPU& other) {
    if (this == &other) return *this;

    free_resources();
    type_ = other.type_;
    D = other.D;

    if (type_ == RotatorType::Matrix) {
        CUDA_CHECK(cudaMalloc(&d_P, sizeof(float) * D * D));
        CUDA_CHECK(cudaMemcpy(d_P, other.d_P, sizeof(float) * D * D, cudaMemcpyDeviceToDevice));
        cublasCreate(&m_handle);
    } else {
        dim_ = other.dim_;
        trunc_dim_ = other.trunc_dim_;
        fac_ = other.fac_;
        log_N_ = other.log_N_;

        size_t flip_bytes = 4 * D / 8;
        CUDA_CHECK(cudaMalloc(&d_flip_, flip_bytes));
        CUDA_CHECK(cudaMemcpy(d_flip_, other.d_flip_, flip_bytes, cudaMemcpyDeviceToDevice));
    }
    return *this;
}

size_t RotatorGPU::size() const {
    return D;
}

// ============================================================================
// Save / Load — file format: [uint8_t type_tag] [type-specific data]
// ============================================================================

void RotatorGPU::save(std::ofstream& output) const {
    uint8_t tag = static_cast<uint8_t>(type_);
    output.write(reinterpret_cast<const char*>(&tag), sizeof(tag));

    if (type_ == RotatorType::Matrix) {
        float* hostP = new float[D * D];
        CUDA_CHECK(cudaMemcpy(hostP, d_P, sizeof(float) * D * D, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < D * D; ++i)
            output.write(reinterpret_cast<char*>(&hostP[i]), sizeof(float));
        delete[] hostP;
    } else {
        size_t flip_bytes = 4 * D / 8;
        std::vector<uint8_t> h_flip(flip_bytes);
        CUDA_CHECK(cudaMemcpy(h_flip.data(), d_flip_, flip_bytes, cudaMemcpyDeviceToHost));
        output.write(reinterpret_cast<const char*>(h_flip.data()), static_cast<long>(flip_bytes));
    }
}

void RotatorGPU::load(std::ifstream& input) {
    uint8_t tag;
    input.read(reinterpret_cast<char*>(&tag), sizeof(tag));
    RotatorType file_type = static_cast<RotatorType>(tag);

    // If the loaded type differs from current, reinitialize
    if (file_type != type_) {
        free_resources();
        type_ = file_type;
        // Re-allocate for the new type (D is already set from constructor)
        if (type_ == RotatorType::Matrix) {
            CUDA_CHECK(cudaMalloc(&d_P, sizeof(float) * D * D));
            cublasCreate(&m_handle);
        } else {
            // Recompute FhtKac derived fields from D
            // dim_ may not be exact (we only have D), but for load this is fine
            // since all FHT parameters derive from D (padded_dim)
            dim_ = D;  // approximate — padded_dim == D after load
            size_t bottom_log = floor_log2(D);
            trunc_dim_ = 1ULL << bottom_log;
            log_N_ = static_cast<int>(bottom_log);
            fac_ = 1.0f / std::sqrt(static_cast<float>(trunc_dim_));

            size_t flip_bytes = 4 * D / 8;
            CUDA_CHECK(cudaMalloc(&d_flip_, flip_bytes));
        }
    }

    if (type_ == RotatorType::Matrix) {
        float* hostP = new float[D * D];
        for (size_t i = 0; i < D * D; ++i)
            input.read(reinterpret_cast<char*>(&hostP[i]), sizeof(float));
        CUDA_CHECK(cudaMemcpy(d_P, hostP, sizeof(float) * D * D, cudaMemcpyHostToDevice));
        delete[] hostP;
    } else {
        size_t flip_bytes = 4 * D / 8;
        std::vector<uint8_t> h_flip(flip_bytes);
        input.read(reinterpret_cast<char*>(h_flip.data()), static_cast<long>(flip_bytes));
        CUDA_CHECK(cudaMemcpy(d_flip_, h_flip.data(), flip_bytes, cudaMemcpyHostToDevice));
    }
}

// ============================================================================
// Rotate
// ============================================================================

void RotatorGPU::rotate(const float* d_A, float* d_RAND_A, size_t N) const {
    if (type_ == RotatorType::Matrix) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasStatus_t status = cublasSgemm(m_handle,
                                            CUBLAS_OP_N, CUBLAS_OP_N,
                                            D, N, D,
                                            &alpha,
                                            d_P, D,
                                            d_A, D,
                                            &beta,
                                            d_RAND_A, D);
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "cuBLAS sgemm failed" << std::endl;
            exit(EXIT_FAILURE);
        }
    } else {
        cudaStream_t stream = 0;
        if (trunc_dim_ == D) {
            float total_scale = fac_ * fac_ * fac_ * fac_;
            fht::dispatch_fused_rotate(d_A, d_RAND_A, d_flip_,
                                        static_cast<int>(N), log_N_,
                                        total_scale, stream);
        } else {
            fht::dispatch_fused_rotate_nonpow2(d_A, d_RAND_A, d_flip_,
                                                static_cast<int>(N), log_N_,
                                                static_cast<int>(D),
                                                fac_, 0.25f, stream);
        }
    }
}
