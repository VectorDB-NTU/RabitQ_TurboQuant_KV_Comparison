/*
 * Minimal extract from quantizer_gpu_fast.cu for standalone quantization.
 * Only contains: get_const_scaling_factors_fully_gpu and its dependencies.
 * All indexing-related kernels removed.
 */

#include <curand_kernel.h>
#include <cub/device/device_reduce.cuh>
#include "quantizer_gpu.cuh"
#include "utils/utils_cuda.cuh"

// ============================================================================
// Warp/block reduction helpers
// ============================================================================

__inline__ __device__ float warpReduceSumdup(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

__inline__ __device__ float blockReduceSumdup(float v) {
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    v = warpReduceSumdup(v);
    if (lane == 0) shared[wid] = v;
    __syncthreads();

    float out = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.f;
    if (wid == 0) out = warpReduceSumdup(out);
    return out;
}

// ============================================================================
// Constant for rescale factor search range
// ============================================================================

extern __constant__ float d_kTightStart_opt[9] = {
    0.0f, 0.15f, 0.20f, 0.52f, 0.59f, 0.71f, 0.75f, 0.77f, 0.81f,
};

// ============================================================================
// compute_best_rescale_parallel: finds optimal rescale factor t for a vector
// ============================================================================

__device__ float compute_best_rescale_parallel(
        float* s_xp_norm, int D, int EX_BITS,
        float* reuse_space, int BlockSize) {
    int tid = threadIdx.x;
    constexpr float kEps = 1e-5f;
    constexpr int kNEnum = 10;

    // Step 1: Find max value
    float local_max = 0.0f;
    for (int i = tid; i < D; i += BlockSize)
        local_max = fmaxf(local_max, fabsf(s_xp_norm[i]));

    float *s_reduce = reuse_space;
    s_reduce[tid] = local_max;
    __syncthreads();
    for (int stride = BlockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            s_reduce[tid] = fmaxf(s_reduce[tid], s_reduce[tid + stride]);
        __syncthreads();
    }

    __shared__ float max_o;
    if (tid == 0) max_o = s_reduce[0];
    __syncthreads();
    if (max_o < kEps) return 1.0f;

    // Step 2: Coarse grid search
    float t_end = static_cast<float>((1 << EX_BITS) - 1 + kNEnum) / max_o;
    float t_start = t_end * d_kTightStart_opt[EX_BITS];

    const int COARSE_SAMPLES = 64;
    float best_coarse_ip = 0.0f;
    float best_coarse_t = t_start;

    for (int i = tid; i < COARSE_SAMPLES; i += BlockSize) {
        float t = t_start + (t_end - t_start) * i / (COARSE_SAMPLES - 1);
        float numerator = 0.0f;
        float sqr_denominator = static_cast<float>(D) * 0.25f;
        for (int j = 0; j < D; j++) {
            float val = fabsf(s_xp_norm[j]);
            int quantized = min(static_cast<int>((t * val) + kEps), (1 << EX_BITS) - 1);
            numerator += (quantized + 0.5f) * val;
            sqr_denominator += quantized * quantized + quantized;
        }
        float ip = numerator / sqrtf(sqr_denominator);
        if (ip > best_coarse_ip) { best_coarse_ip = ip; best_coarse_t = t; }
    }

    float *s_coarse_ip = reuse_space + BlockSize;
    float *s_coarse_t = s_coarse_ip + BlockSize;
    s_coarse_ip[tid] = best_coarse_ip;
    s_coarse_t[tid] = best_coarse_t;
    __syncthreads();
    for (int stride = BlockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_coarse_ip[tid + stride] > s_coarse_ip[tid]) {
                s_coarse_ip[tid] = s_coarse_ip[tid + stride];
                s_coarse_t[tid] = s_coarse_t[tid + stride];
            }
        }
        __syncthreads();
    }

    // Step 3: Fine search
    float center_t = s_coarse_t[0];
    float range = (t_end - t_start) / COARSE_SAMPLES;
    float fine_start = fmaxf(t_start, center_t - range);
    float fine_end = fminf(t_end, center_t + range);

    const int FINE_SAMPLES = 32;
    float best_fine_ip = 0.0f;
    float best_fine_t = center_t;

    for (int i = tid; i < FINE_SAMPLES; i += BlockSize) {
        float t = fine_start + (fine_end - fine_start) * i / (FINE_SAMPLES - 1);
        float numerator = 0.0f;
        float sqr_denominator = static_cast<float>(D) * 0.25f;
        for (int j = 0; j < D; j++) {
            float val = fabsf(s_xp_norm[j]);
            int quantized = min(static_cast<int>((t * val) + kEps), (1 << EX_BITS) - 1);
            numerator += (quantized + 0.5f) * val;
            sqr_denominator += quantized * quantized + quantized;
        }
        float ip = numerator / sqrtf(sqr_denominator);
        if (ip > best_fine_ip) { best_fine_ip = ip; best_fine_t = t; }
    }

    float* s_fine_ip = s_coarse_ip;
    float* s_fine_t = s_coarse_t;
    s_fine_ip[tid] = best_fine_ip;
    s_fine_t[tid] = best_fine_t;
    __syncthreads();
    for (int stride = BlockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_fine_ip[tid + stride] > s_fine_ip[tid]) {
                s_fine_ip[tid] = s_fine_ip[tid + stride];
                s_fine_t[tid] = s_fine_t[tid + stride];
            }
        }
        __syncthreads();
    }

    return s_fine_t[0];
}

// ============================================================================
// fully_fused_kernel: generates random vector, normalizes, finds best rescale
// ============================================================================

__global__ void fully_fused_kernel(
    float* __restrict__ output_factors,
    const int rows, const int cols, const int ex_bits,
    unsigned long long seed) {
    const int row_id = blockIdx.x;
    if (row_id >= rows) return;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    extern __shared__ float shared_mem[];
    float* row_data = shared_mem;
    float* reuse_space = &row_data[cols];

    curandState rng_state;
    curand_init(seed, row_id * block_size + tid, 0, &rng_state);

    for (int i = tid; i < cols; i += block_size)
        row_data[i] = curand_normal(&rng_state);
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += block_size) {
        float val = row_data[i];
        local_sum += val * val;
    }
    float norm_squared = blockReduceSumdup(local_sum);

    __shared__ float inv_norm;
    if (tid == 0) inv_norm = rsqrtf(norm_squared);
    __syncthreads();

    for (int i = tid; i < cols; i += block_size)
        row_data[i] = fabsf(row_data[i] * inv_norm);
    __syncthreads();

    float rescale_factor = compute_best_rescale_parallel(
        row_data, cols, ex_bits, reuse_space, block_size);

    if (tid == 0) output_factors[row_id] = rescale_factor;
}

// ============================================================================
// get_const_scaling_factors_fully_gpu
// ============================================================================

float DataQuantizerGPU::get_const_scaling_factors_fully_gpu(size_t dim, size_t ex_bits) {
    constexpr long kConstNum = 100;

    float* d_factors;
    float* d_sum;
    CUDA_CHECK(cudaMalloc(&d_factors, kConstNum * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));

    int block_size = 256;
    if (dim <= 512) block_size = 128;
    if (dim >= 1536) block_size = 512;

    size_t shared_mem_size = (dim + 3 * block_size) * sizeof(float);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (shared_mem_size > prop.sharedMemPerBlock) {
        block_size = 128;
        shared_mem_size = (dim + 3 * block_size) * sizeof(float);
    }

#ifdef DEBUG_BATCH_CONSTRUCT
    unsigned long long seed = 42;
#else
    unsigned long long seed = time(nullptr);
#endif

    fully_fused_kernel<<<kConstNum, block_size, shared_mem_size>>>(
        d_factors, kConstNum, dim, ex_bits, seed);
    CUDA_CHECK(cudaGetLastError());

    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(nullptr, temp_storage_bytes, d_factors, d_sum, kConstNum);

    void* d_temp_storage = nullptr;
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_factors, d_sum, kConstNum);
    CUDA_CHECK(cudaGetLastError());

    float sum;
    CUDA_CHECK(cudaMemcpy(&sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_factors));
    CUDA_CHECK(cudaFree(d_sum));
    CUDA_CHECK(cudaFree(d_temp_storage));

    return sum / kConstNum;
}
