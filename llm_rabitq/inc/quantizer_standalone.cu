//
// Standalone GPU quantizer — implementation.
//

#include "quantizer_standalone.cuh"
#include "quantizer_gpu.cuh"   // for DataQuantizerGPU::get_const_scaling_factors_fully_gpu
#include <cmath>

// ============================================================================
// Helpers
// ============================================================================

static inline uint32_t rd_up(uint32_t x, uint32_t n) {
    return ((x + n - 1u) / n) * n;
}

// ============================================================================
// GPU Kernels
// ============================================================================

/// Pad N vectors from dim to padded_dim (zero-fill), then subtract centroid.
__global__ void sa_pad_and_subtract_kernel(
    const float* __restrict__ d_data,
    const float* __restrict__ d_centroid,  // padded_dim, already padded
    float* __restrict__ d_out,
    int N, int dim, int padded_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * padded_dim) return;

    int vec = idx / padded_dim;
    int d   = idx % padded_dim;

    float val = (d < dim) ? d_data[vec * dim + d] : 0.0f;
    d_out[idx] = val - d_centroid[d];
}

// ============================================================================
// Forward declaration of compute_best_rescale_parallel (defined in quantizer_gpu_fast.cu)
// We need it for the non-fast quantization kernel.
// ============================================================================

// Tight-start constants in device constant memory (must match quantizer_gpu_fast.cu)
extern __device__ __constant__ float d_kTightStart_opt[9];

__device__ float compute_best_rescale_parallel(
    float* s_xp_norm, int D, int EX_BITS, float* reuse_space, int BlockSize);

// ============================================================================
// Single-pass quantization kernel — templated on code type
// ============================================================================

/// Fast path: uses precomputed const_scaling_factor.
/// Non-fast path: uses compute_best_rescale_parallel per vector.
template<typename CodeT>
__global__ void sa_quantize_total_code_kernel(
    const float* __restrict__ d_residual,   // N × padded_dim
    CodeT* __restrict__ d_total_code,       // N × padded_dim
    int N, int padded_dim, int ex_bits,
    float const_scaling_factor,
    bool use_fast)
{
    extern __shared__ char smem[];
    float* s_partial = reinterpret_cast<float*>(smem);
    // For non-fast path, compute_best_rescale_parallel needs 3×blockDim.x reuse space.
    // Layout: [s_partial: blockDim.x] [s_abs_norm: padded_dim] [reuse: 3*blockDim.x]
    float* s_abs_norm = s_partial + blockDim.x;
    float* s_reuse    = s_abs_norm + padded_dim;

    int vec_id = blockIdx.x;
    if (vec_id >= N) return;

    const float* res = d_residual + (size_t)vec_id * padded_dim;
    CodeT* code = d_total_code + (size_t)vec_id * padded_dim;

    // Step 1: compute norm of residual (block-level reduction)
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < padded_dim; i += blockDim.x) {
        float v = res[i];
        local_sum += v * v;
    }
    s_partial[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < (unsigned)s)
            s_partial[threadIdx.x] += s_partial[threadIdx.x + s];
        __syncthreads();
    }
    float inv_norm = rsqrtf(s_partial[0] + 1e-30f);

    // Compute abs_norm for each dimension (needed for both paths)
    for (int i = threadIdx.x; i < padded_dim; i += blockDim.x) {
        s_abs_norm[i] = fabsf(res[i]) * inv_norm;
    }
    __syncthreads();

    // Step 2: determine scaling factor
    float t;
    if (use_fast) {
        t = const_scaling_factor;
    } else {
        // Per-vector optimal rescale via parallel search
        t = compute_best_rescale_parallel(s_abs_norm, padded_dim, ex_bits,
                                          s_reuse, blockDim.x);
    }

    // Step 3: quantize each dimension
    int mask = (1 << ex_bits) - 1;
    int max_code = mask;

    for (int i = threadIdx.x; i < padded_dim; i += blockDim.x) {
        float r = res[i];

        // Quantize absolute normalized value
        int ex_code = __float2int_rd(t * s_abs_norm[i] + 1e-5f);
        if (ex_code > max_code) ex_code = max_code;

        // Sign reversion: flip bits for negative residuals
        if (r < 0.0f) ex_code = (~ex_code) & mask;

        // Merge sign bit as MSB
        int sign = (r >= 0.0f) ? 1 : 0;
        code[i] = static_cast<CodeT>(ex_code + (sign << ex_bits));
    }
}

// ============================================================================
// v2.1: Signed quantization kernel
//
// Quantize the signed normalized value directly:
//   signed_code = floor(t * val + copysign(eps, val))
//   total_code  = signed_code + 2^ex_bits
//
// Note: produces ±1 differences vs v1/CPU on exact-integer boundaries for
// negative values (bit-flip encoding ≠ simple signed shift). Slightly faster
// than v2.2 due to no branch.
// ============================================================================

template<typename CodeT>
__global__ void sa_quantize_signed_kernel(
    const float* __restrict__ d_residual,
    CodeT* __restrict__ d_total_code,
    int N, int padded_dim, int ex_bits,
    float const_scaling_factor,
    bool use_fast)
{
    extern __shared__ char smem[];
    float* s_partial = reinterpret_cast<float*>(smem);
    float* s_abs_norm = s_partial + blockDim.x;
    float* s_reuse    = s_abs_norm + padded_dim;

    int vec_id = blockIdx.x;
    if (vec_id >= N) return;

    const float* res = d_residual + (size_t)vec_id * padded_dim;
    CodeT* code = d_total_code + (size_t)vec_id * padded_dim;

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < padded_dim; i += blockDim.x) {
        float v = res[i];
        local_sum += v * v;
    }
    s_partial[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < (unsigned)s)
            s_partial[threadIdx.x] += s_partial[threadIdx.x + s];
        __syncthreads();
    }
    float inv_norm = rsqrtf(s_partial[0] + 1e-30f);

    if (!use_fast) {
        for (int i = threadIdx.x; i < padded_dim; i += blockDim.x)
            s_abs_norm[i] = fabsf(res[i]) * inv_norm;
        __syncthreads();
    }

    float t;
    if (use_fast) {
        t = const_scaling_factor;
    } else {
        t = compute_best_rescale_parallel(s_abs_norm, padded_dim, ex_bits,
                                          s_reuse, blockDim.x);
    }

    // Direct signed quantization: floor(t*val + copysign(eps,val)) + offset
    int offset = 1 << ex_bits;
    int max_total = (1 << (ex_bits + 1)) - 1;
    constexpr float kEps = 1e-5f;

    for (int i = threadIdx.x; i < padded_dim; i += blockDim.x) {
        float val = res[i] * inv_norm;
        float eps_signed = copysignf(kEps, val);
        int total = __float2int_rd(t * val + eps_signed) + offset;
        if (total < 0) total = 0;
        if (total > max_total) total = max_total;
        code[i] = static_cast<CodeT>(total);
    }
}

// ============================================================================
// v2.2: Abs+branch quantization kernel (exact match with v1/CPU bit-flip encoding)
// ============================================================================

template<typename CodeT>
__global__ void sa_quantize_total_code_direct_kernel(
    const float* __restrict__ d_residual,
    CodeT* __restrict__ d_total_code,
    int N, int padded_dim, int ex_bits,
    float const_scaling_factor,
    bool use_fast)
{
    extern __shared__ char smem[];
    float* s_partial = reinterpret_cast<float*>(smem);
    float* s_abs_norm = s_partial + blockDim.x;
    float* s_reuse    = s_abs_norm + padded_dim;

    int vec_id = blockIdx.x;
    if (vec_id >= N) return;

    const float* res = d_residual + (size_t)vec_id * padded_dim;
    CodeT* code = d_total_code + (size_t)vec_id * padded_dim;

    // Step 1: compute norm (block reduction)
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < padded_dim; i += blockDim.x) {
        float v = res[i];
        local_sum += v * v;
    }
    s_partial[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < (unsigned)s)
            s_partial[threadIdx.x] += s_partial[threadIdx.x + s];
        __syncthreads();
    }
    float inv_norm = rsqrtf(s_partial[0] + 1e-30f);

    // For non-fast path: need abs_norm in smem for compute_best_rescale_parallel
    if (!use_fast) {
        for (int i = threadIdx.x; i < padded_dim; i += blockDim.x)
            s_abs_norm[i] = fabsf(res[i]) * inv_norm;
        __syncthreads();
    }

    // Step 2: determine scaling factor
    float t;
    if (use_fast) {
        t = const_scaling_factor;
    } else {
        t = compute_best_rescale_parallel(s_abs_norm, padded_dim, ex_bits,
                                          s_reuse, blockDim.x);
    }

    // Step 3: quantize using abs magnitude, then apply sign via offset.
    //
    // The CPU bit-flip encoding maps:
    //   positive (magnitude k) → k + 2^ex_bits
    //   negative (magnitude k) → (2^ex_bits - 1) - k
    //
    // This is NOT a simple signed shift, so we quantize |val|, then branch:
    int mask = (1 << ex_bits) - 1;
    int offset = 1 << ex_bits;
    constexpr float kEps = 1e-5f;

    for (int i = threadIdx.x; i < padded_dim; i += blockDim.x) {
        float r = res[i];
        float abs_val = fabsf(r) * inv_norm;

        int k = __float2int_rd(t * abs_val + kEps);
        if (k > mask) k = mask;

        int total = (r >= 0.0f) ? (k + offset) : (mask - k);
        code[i] = static_cast<CodeT>(total);
    }
}

// ============================================================================
// v2.3: Split-kernel approach for non-fast path
//
// Kernel A (64 threads): compute norm + abs_norm + best_rescale → t, inv_norm per vector
// Kernel B (256 threads): read t, inv_norm, quantize (no search code → high occupancy)
// ============================================================================

/// Warp-level sum reduction
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

/// Warp-level max reduction (returns max value and corresponding t in lane 0)
__device__ __forceinline__ void warp_reduce_max_with_t(float& ip, float& t) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_ip = __shfl_down_sync(0xffffffff, ip, offset);
        float other_t  = __shfl_down_sync(0xffffffff, t, offset);
        if (other_ip > ip) { ip = other_ip; t = other_t; }
    }
}

/// Evaluate one rescale sample: compute IP metric for a given t value.
/// Each warp cooperatively evaluates one sample (32 threads share the D-element loop).
/// abs_norm should be in shared memory for fast repeated access.
__device__ __forceinline__ float evaluate_rescale_sample(
    const float* __restrict__ s_abs_norm, int D, int ex_bits, float t, int lane_id)
{
    constexpr float kEps = 1e-5f;
    int max_code = (1 << ex_bits) - 1;
    float numerator = 0.0f;
    float sqr_denom = (lane_id == 0) ? static_cast<float>(D) * 0.25f : 0.0f;

    for (int j = lane_id; j < D; j += 32) {
        float val = s_abs_norm[j];
        int quantized = min(__float2int_rd(t * val + kEps), max_code);
        numerator += (quantized + 0.5f) * val;
        sqr_denom += quantized * quantized + quantized;
    }

    numerator = warp_reduce_sum(numerator);
    sqr_denom = warp_reduce_sum(sqr_denom);

    return numerator / sqrtf(sqr_denom);
}

/// Kernel A: compute per-vector rescale factor and inverse norm.
///
/// Warp-cooperative: each warp evaluates one sample point, 32 threads share the
/// D-element inner loop. 256 threads = 8 warps → 8 samples evaluated in parallel.
/// Coarse (64 samples) in 8 iterations, fine (32 samples) in 4 iterations.
///
/// Smem: only kBlockSize floats for norm reduction + 2*kNWarps floats for sample results.
template<int kBlockSize = 256>
__global__ void sa_compute_rescale_kernel(
    const float* __restrict__ d_residual,
    float* __restrict__ d_t_per_vec,
    float* __restrict__ d_inv_norm_per_vec,
    int N, int padded_dim, int ex_bits)
{
    constexpr int kNWarps = kBlockSize / 32;
    constexpr float kEps = 1e-5f;
    constexpr int kNEnum = 10;
    constexpr int COARSE_SAMPLES = 64;
    constexpr int FINE_SAMPLES = 64;

    // Shared memory layout:
    //   [0, kBlockSize)          : s_reduce (for norm/max reductions)
    //   [kBlockSize, kBlockSize + padded_dim) : s_abs_norm (cached in smem for fast repeated access)
    //   [kBlockSize + padded_dim, + 2*kNWarps) : s_warp_ip, s_warp_t
    extern __shared__ char smem[];
    float* s_reduce  = reinterpret_cast<float*>(smem);
    float* s_abs_norm = s_reduce + kBlockSize;
    float* s_warp_ip = s_abs_norm + padded_dim;
    float* s_warp_t  = s_warp_ip + kNWarps;

    int vec_id = blockIdx.x;
    if (vec_id >= N) return;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    const float* res = d_residual + (size_t)vec_id * padded_dim;

    // Step 1: Compute norm (block-level reduction)
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < padded_dim; i += kBlockSize)
        local_sum += res[i] * res[i];
    s_reduce[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = kBlockSize / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) s_reduce[threadIdx.x] += s_reduce[threadIdx.x + s];
        __syncthreads();
    }
    float inv_norm = rsqrtf(s_reduce[0] + 1e-30f);
    if (threadIdx.x == 0) d_inv_norm_per_vec[vec_id] = inv_norm;

    // Step 2: Load abs_norm into shared memory + find max
    float local_max = 0.0f;
    for (int i = threadIdx.x; i < padded_dim; i += kBlockSize) {
        float val = fabsf(res[i]) * inv_norm;
        s_abs_norm[i] = val;
        local_max = fmaxf(local_max, val);
    }
    s_reduce[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = kBlockSize / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) s_reduce[threadIdx.x] = fmaxf(s_reduce[threadIdx.x], s_reduce[threadIdx.x + s]);
        __syncthreads();
    }
    float max_o = s_reduce[0];

    if (max_o < kEps) {
        if (threadIdx.x == 0) d_t_per_vec[vec_id] = 1.0f;
        return;
    }

    float t_end = static_cast<float>((1 << ex_bits) - 1 + kNEnum) / max_o;
    float t_start = t_end * d_kTightStart_opt[ex_bits];

    // Step 3: Coarse grid search — each warp evaluates one sample
    float best_coarse_ip = 0.0f;
    float best_coarse_t = t_start;

    for (int base = 0; base < COARSE_SAMPLES; base += kNWarps) {
        int sample_idx = base + warp_id;
        float t = (sample_idx < COARSE_SAMPLES)
            ? t_start + (t_end - t_start) * sample_idx / (COARSE_SAMPLES - 1)
            : t_start;

        float ip = (sample_idx < COARSE_SAMPLES)
            ? evaluate_rescale_sample(s_abs_norm, padded_dim, ex_bits, t, lane_id)
            : 0.0f;

        // ip and t are valid in lane 0 of each warp
        if (lane_id == 0) {
            if (ip > best_coarse_ip) { best_coarse_ip = ip; best_coarse_t = t; }
        }
    }

    // Cross-warp reduction to find global best coarse sample
    if (lane_id == 0) {
        s_warp_ip[warp_id] = best_coarse_ip;
        s_warp_t[warp_id] = best_coarse_t;
    }
    __syncthreads();

    // Cross-warp reduction: warp 0 reduces kNWarps results via shared memory
    // All lanes in warp 0 must participate in __shfl_down_sync (full mask).
    // Lanes >= kNWarps get dummy values that won't win the max comparison.
    if (warp_id == 0) {
        float ip = (lane_id < kNWarps) ? s_warp_ip[lane_id] : -1.0f;
        float t  = (lane_id < kNWarps) ? s_warp_t[lane_id]  : 0.0f;
        for (int s = kNWarps / 2; s > 0; s >>= 1) {
            float other_ip = __shfl_down_sync(0xffffffff, ip, s);
            float other_t  = __shfl_down_sync(0xffffffff, t, s);
            if (other_ip > ip) { ip = other_ip; t = other_t; }
        }
        if (lane_id == 0) { s_warp_ip[0] = ip; s_warp_t[0] = t; }
    }
    __syncthreads();

    float center_t = s_warp_t[0];
    float range = (t_end - t_start) / COARSE_SAMPLES;
    float fine_start = fmaxf(t_start, center_t - range);
    float fine_end   = fminf(t_end,   center_t + range);

    // Step 4: Fine grid search — same warp-cooperative approach
    float best_fine_ip = 0.0f;
    float best_fine_t = center_t;

    for (int base = 0; base < FINE_SAMPLES; base += kNWarps) {
        int sample_idx = base + warp_id;
        float t = (sample_idx < FINE_SAMPLES)
            ? fine_start + (fine_end - fine_start) * sample_idx / (FINE_SAMPLES - 1)
            : center_t;

        float ip = (sample_idx < FINE_SAMPLES)
            ? evaluate_rescale_sample(s_abs_norm, padded_dim, ex_bits, t, lane_id)
            : 0.0f;

        if (lane_id == 0) {
            if (ip > best_fine_ip) { best_fine_ip = ip; best_fine_t = t; }
        }
    }

    // Cross-warp reduction for fine (same fix: all lanes in warp 0 participate)
    if (lane_id == 0) {
        s_warp_ip[warp_id] = best_fine_ip;
        s_warp_t[warp_id] = best_fine_t;
    }
    __syncthreads();

    if (warp_id == 0) {
        float ip = (lane_id < kNWarps) ? s_warp_ip[lane_id] : -1.0f;
        float t  = (lane_id < kNWarps) ? s_warp_t[lane_id]  : 0.0f;
        for (int s = kNWarps / 2; s > 0; s >>= 1) {
            float other_ip = __shfl_down_sync(0xffffffff, ip, s);
            float other_t  = __shfl_down_sync(0xffffffff, t, s);
            if (other_ip > ip) { ip = other_ip; t = other_t; }
        }
        if (lane_id == 0) d_t_per_vec[vec_id] = t;
    }
}

/// Kernel B: quantize using pre-computed t and inv_norm (v2.2 abs+branch logic).
/// No search code → fewer registers → higher occupancy.
template<typename CodeT>
__global__ void sa_quantize_with_t_kernel(
    const float* __restrict__ d_residual,
    const float* __restrict__ d_t_per_vec,
    const float* __restrict__ d_inv_norm_per_vec,
    CodeT* __restrict__ d_total_code,
    int N, int padded_dim, int ex_bits)
{
    int vec_id = blockIdx.x;
    if (vec_id >= N) return;

    const float* res = d_residual + (size_t)vec_id * padded_dim;
    CodeT* code = d_total_code + (size_t)vec_id * padded_dim;

    float t = d_t_per_vec[vec_id];
    float inv_norm = d_inv_norm_per_vec[vec_id];

    int mask = (1 << ex_bits) - 1;
    int offset = 1 << ex_bits;
    constexpr float kEps = 1e-5f;

    for (int i = threadIdx.x; i < padded_dim; i += blockDim.x) {
        float r = res[i];
        float abs_val = fabsf(r) * inv_norm;

        int k = __float2int_rd(t * abs_val + kEps);
        if (k > mask) k = mask;

        int total = (r >= 0.0f) ? (k + offset) : (mask - k);
        code[i] = static_cast<CodeT>(total);
    }
}

// ============================================================================
// v2.4: Fused rescale search + quantize kernel (non-fast path)
//
// Combines warp-cooperative rescale search and quantization into one kernel.
// No intermediate global memory for t/inv_norm. Fast path uses const_scaling_factor.
// ============================================================================

template<typename CodeT, int kBlockSize = 256>
__global__ void sa_quantize_fused_kernel(
    const float* __restrict__ d_residual,
    CodeT* __restrict__ d_total_code,
    int N, int padded_dim, int ex_bits,
    float const_scaling_factor,
    bool use_fast)
{
    constexpr int kNWarps = kBlockSize / 32;
    constexpr float kEps = 1e-5f;
    constexpr int kNEnum = 10;
    constexpr int COARSE_SAMPLES = 64;
    constexpr int FINE_SAMPLES = 64;

    // Shared memory layout:
    //   [0, kBlockSize)                     : s_reduce
    //   [kBlockSize, kBlockSize + padded_dim): s_abs_norm
    //   [kBlockSize + padded_dim, +2*kNWarps): s_warp_ip, s_warp_t
    extern __shared__ char smem[];
    float* s_reduce   = reinterpret_cast<float*>(smem);
    float* s_abs_norm = s_reduce + kBlockSize;
    float* s_warp_ip  = s_abs_norm + padded_dim;
    float* s_warp_t   = s_warp_ip + kNWarps;

    int vec_id = blockIdx.x;
    if (vec_id >= N) return;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    const float* res = d_residual + (size_t)vec_id * padded_dim;
    CodeT* code = d_total_code + (size_t)vec_id * padded_dim;

    // Step 1: Compute norm
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < padded_dim; i += kBlockSize)
        local_sum += res[i] * res[i];
    s_reduce[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = kBlockSize / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) s_reduce[threadIdx.x] += s_reduce[threadIdx.x + s];
        __syncthreads();
    }
    float inv_norm = rsqrtf(s_reduce[0] + 1e-30f);

    float t;

    if (use_fast) {
        t = const_scaling_factor;
    } else {
        // Step 2: Load abs_norm into shared memory + find max
        float local_max = 0.0f;
        for (int i = threadIdx.x; i < padded_dim; i += kBlockSize) {
            float val = fabsf(res[i]) * inv_norm;
            s_abs_norm[i] = val;
            local_max = fmaxf(local_max, val);
        }
        s_reduce[threadIdx.x] = local_max;
        __syncthreads();
        for (int s = kBlockSize / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) s_reduce[threadIdx.x] = fmaxf(s_reduce[threadIdx.x], s_reduce[threadIdx.x + s]);
            __syncthreads();
        }
        float max_o = s_reduce[0];

        if (max_o < kEps) { t = 1.0f; }
        else {
            float t_end = static_cast<float>((1 << ex_bits) - 1 + kNEnum) / max_o;
            float t_start = t_end * d_kTightStart_opt[ex_bits];

            // Step 3: Coarse grid search (warp-cooperative)
            float best_coarse_ip = 0.0f, best_coarse_t = t_start;
            for (int base = 0; base < COARSE_SAMPLES; base += kNWarps) {
                int si = base + warp_id;
                float tc = (si < COARSE_SAMPLES)
                    ? t_start + (t_end - t_start) * si / (COARSE_SAMPLES - 1) : t_start;
                float ip = (si < COARSE_SAMPLES)
                    ? evaluate_rescale_sample(s_abs_norm, padded_dim, ex_bits, tc, lane_id) : 0.0f;
                if (lane_id == 0 && ip > best_coarse_ip) { best_coarse_ip = ip; best_coarse_t = tc; }
            }

            if (lane_id == 0) { s_warp_ip[warp_id] = best_coarse_ip; s_warp_t[warp_id] = best_coarse_t; }
            __syncthreads();
            if (warp_id == 0) {
                float ip = (lane_id < kNWarps) ? s_warp_ip[lane_id] : -1.0f;
                float tc = (lane_id < kNWarps) ? s_warp_t[lane_id]  : 0.0f;
                for (int s = kNWarps / 2; s > 0; s >>= 1) {
                    float oi = __shfl_down_sync(0xffffffff, ip, s);
                    float ot = __shfl_down_sync(0xffffffff, tc, s);
                    if (oi > ip) { ip = oi; tc = ot; }
                }
                if (lane_id == 0) { s_warp_ip[0] = ip; s_warp_t[0] = tc; }
            }
            __syncthreads();

            float center_t = s_warp_t[0];
            float range = (t_end - t_start) / COARSE_SAMPLES;
            float fine_start = fmaxf(t_start, center_t - range);
            float fine_end   = fminf(t_end,   center_t + range);

            // Step 4: Fine grid search
            float best_fine_ip = 0.0f, best_fine_t = center_t;
            for (int base = 0; base < FINE_SAMPLES; base += kNWarps) {
                int si = base + warp_id;
                float tf = (si < FINE_SAMPLES)
                    ? fine_start + (fine_end - fine_start) * si / (FINE_SAMPLES - 1) : center_t;
                float ip = (si < FINE_SAMPLES)
                    ? evaluate_rescale_sample(s_abs_norm, padded_dim, ex_bits, tf, lane_id) : 0.0f;
                if (lane_id == 0 && ip > best_fine_ip) { best_fine_ip = ip; best_fine_t = tf; }
            }

            if (lane_id == 0) { s_warp_ip[warp_id] = best_fine_ip; s_warp_t[warp_id] = best_fine_t; }
            __syncthreads();
            if (warp_id == 0) {
                float ip = (lane_id < kNWarps) ? s_warp_ip[lane_id] : -1.0f;
                float tf = (lane_id < kNWarps) ? s_warp_t[lane_id]  : 0.0f;
                for (int s = kNWarps / 2; s > 0; s >>= 1) {
                    float oi = __shfl_down_sync(0xffffffff, ip, s);
                    float ot = __shfl_down_sync(0xffffffff, tf, s);
                    if (oi > ip) { ip = oi; tf = ot; }
                }
                if (lane_id == 0) s_warp_t[0] = tf;
            }
            __syncthreads();
            t = s_warp_t[0];
        }
    }

    // Step 5: Quantize (abs+branch, same as v2.2)
    int mask = (1 << ex_bits) - 1;
    int offset = 1 << ex_bits;
    for (int i = threadIdx.x; i < padded_dim; i += kBlockSize) {
        float r = res[i];
        float abs_val = fabsf(r) * inv_norm;
        int k = __float2int_rd(t * abs_val + kEps);
        if (k > mask) k = mask;
        int total = (r >= 0.0f) ? (k + offset) : (mask - k);
        code[i] = static_cast<CodeT>(total);
    }
}

// ============================================================================
// Scalar factor kernel — templated on code type
// ============================================================================

template<typename CodeT>
__global__ void sa_compute_delta_vl_kernel(
    const float* __restrict__ d_residual,
    const CodeT* __restrict__ d_total_code,
    float* __restrict__ d_delta,
    float* __restrict__ d_vl,
    int N, int padded_dim, int ex_bits,
    int delta_mode)  // 0=RECONSTRUCTION, 1=UNBIASED, 2=PLAIN
{
    extern __shared__ char smem[];
    float* s_buf = reinterpret_cast<float*>(smem);

    int vec_id = blockIdx.x;
    if (vec_id >= N) return;

    const float* res = d_residual + (size_t)vec_id * padded_dim;
    const CodeT* code = d_total_code + (size_t)vec_id * padded_dim;

    float cb = -((float)(1 << ex_bits) - 0.5f);

    float local_res_sq = 0.0f, local_ucb_sq = 0.0f, local_dot = 0.0f;
    for (int i = threadIdx.x; i < padded_dim; i += blockDim.x) {
        float r = res[i];
        float u = (float)code[i] + cb;
        local_res_sq += r * r;
        local_ucb_sq += u * u;
        local_dot    += r * u;
    }

    float* s_res = s_buf;
    float* s_ucb = s_buf + blockDim.x;
    float* s_dot = s_buf + 2 * blockDim.x;
    s_res[threadIdx.x] = local_res_sq;
    s_ucb[threadIdx.x] = local_ucb_sq;
    s_dot[threadIdx.x] = local_dot;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < (unsigned)s) {
            s_res[threadIdx.x] += s_res[threadIdx.x + s];
            s_ucb[threadIdx.x] += s_ucb[threadIdx.x + s];
            s_dot[threadIdx.x] += s_dot[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float norm_res = sqrtf(s_res[0]);
        float norm_ucb = sqrtf(s_ucb[0]);
        float cos_sim  = s_dot[0] / (norm_res * norm_ucb + 1e-30f);

        // 0=RECONSTRUCTION: delta = norm / norm_quan * cos
        // 1=UNBIASED:       delta = norm / norm_quan / cos
        // 2=PLAIN:           delta = norm / norm_quan
        float ratio = norm_res / (norm_ucb + 1e-30f);
        float delta;
        if (delta_mode == 1)
            delta = ratio / (cos_sim + 1e-30f);
        else if (delta_mode == 2)
            delta = ratio;
        else
            delta = ratio * cos_sim;
        d_delta[vec_id] = delta;
        d_vl[vec_id]    = delta * cb;
    }
}

// ============================================================================
// Full factor kernel — templated on code type
// ============================================================================

template<typename CodeT>
__global__ void sa_compute_full_factors_kernel(
    const float* __restrict__ d_residual,
    const float* __restrict__ d_centroid,   // padded_dim (rotated centroid)
    const CodeT* __restrict__ d_total_code,
    float* __restrict__ d_factors,          // N × 3
    int N, int padded_dim, int ex_bits)
{
    extern __shared__ char smem[];
    float* s_buf = reinterpret_cast<float*>(smem);  // 4 × blockDim.x

    int vec_id = blockIdx.x;
    if (vec_id >= N) return;

    const float* res  = d_residual + (size_t)vec_id * padded_dim;
    const CodeT* code = d_total_code + (size_t)vec_id * padded_dim;

    float cb = -((float)(1 << ex_bits) - 0.5f);
    constexpr float kEpsilon = 1.9f;

    float local_l2 = 0.0f, local_ip_res = 0.0f, local_ip_cent = 0.0f, local_xu_sq = 0.0f;
    for (int i = threadIdx.x; i < padded_dim; i += blockDim.x) {
        float r = res[i];
        float c = d_centroid[i];
        float xu_cb = (float)code[i] + cb;

        local_l2      += r * r;
        local_ip_res  += r * xu_cb;
        local_ip_cent += c * xu_cb;
        local_xu_sq   += xu_cb * xu_cb;
    }

    float* s0 = s_buf;
    float* s1 = s_buf + blockDim.x;
    float* s2 = s_buf + 2 * blockDim.x;
    float* s3 = s_buf + 3 * blockDim.x;
    s0[threadIdx.x] = local_l2;
    s1[threadIdx.x] = local_ip_res;
    s2[threadIdx.x] = local_ip_cent;
    s3[threadIdx.x] = local_xu_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < (unsigned)s) {
            s0[threadIdx.x] += s0[threadIdx.x + s];
            s1[threadIdx.x] += s1[threadIdx.x + s];
            s2[threadIdx.x] += s2[threadIdx.x + s];
            s3[threadIdx.x] += s3[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float l2_sq        = s0[0];
        float ip_resi_xucb = s1[0];
        float ip_cent_xucb = s2[0];
        float xu_sq         = s3[0];

        float l2_norm = sqrtf(l2_sq);
        float denom   = ip_resi_xucb + 1e-30f;

        float f_add     = l2_sq + 2.0f * l2_sq * (ip_cent_xucb / denom);
        float f_rescale = -2.0f * l2_sq / denom;

        float ratio = (l2_sq * xu_sq) / (denom * denom);
        float inner = fmaxf(0.0f, (ratio - 1.0f) / ((float)padded_dim - 1.0f));
        float f_error = 2.0f * l2_norm * kEpsilon * sqrtf(inner);

        d_factors[vec_id * 3 + 0] = f_add;
        d_factors[vec_id * 3 + 1] = f_rescale;
        d_factors[vec_id * 3 + 2] = f_error;
    }
}

// ============================================================================
// StandaloneQuantizerGPU — constructor
// ============================================================================

StandaloneQuantizerGPU::StandaloneQuantizerGPU(uint32_t dim, size_t total_bits,
                                               RotatorType rota_type, bool fast)
    : dim_(dim), padded_dim_(rd_up(dim, 64)),
      total_bits_(total_bits), ex_bits_(total_bits - 1),
      rotator_(dim, rota_type),
      const_scaling_factor_(0.0f),
      use_fast_quantize_(fast)
{
    if (use_fast_quantize_ && ex_bits_ > 0) {
        // Use existing GPU-accelerated const scaling factor computation
        const_scaling_factor_ = DataQuantizerGPU::get_const_scaling_factors_fully_gpu(
            padded_dim_, ex_bits_);
    }
}

// ============================================================================
// Internal: compute residuals (pad → rotate → subtract centroid)
// ============================================================================

/// Pad kernel: copy N × dim → N × padded_dim with zero-fill (contiguous input).
__global__ void sa_pad_kernel(const float* __restrict__ d_src,
                               float* __restrict__ d_dst,
                               int N, int dim, int padded_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * padded_dim) return;
    int row = idx / padded_dim;
    int col = idx % padded_dim;
    d_dst[idx] = (col < dim) ? d_src[row * dim + col] : 0.0f;
}

void StandaloneQuantizerGPU::compute_residuals(const float* d_data,
                                                const float* d_centroid,
                                                size_t N, float* d_residual) const {
    size_t padded_bytes = (N + 1) * padded_dim_ * sizeof(float);
    bool needs_pad = (dim_ != padded_dim_);
    constexpr int block = 256;

    // Step 1: Pad data[0..N-1] and centroid into a (N+1) × padded_dim buffer
    float* d_buf = nullptr;  // (N+1) × padded_dim: data rows + centroid row
    CUDA_CHECK(cudaMalloc(&d_buf, padded_bytes));

    if (needs_pad) {
        // Pad N data vectors + 1 centroid row
        int total = static_cast<int>((N + 1) * padded_dim_);
        int grid = (total + block - 1) / block;

        // First pad N data rows
        sa_pad_kernel<<<(static_cast<int>(N * padded_dim_) + block - 1) / block, block>>>(
            d_data, d_buf,
            static_cast<int>(N), static_cast<int>(dim_), static_cast<int>(padded_dim_));
        CUDA_CHECK(cudaGetLastError());

        // Pad centroid into the (N+1)-th row
        if (d_centroid) {
            float* d_cent_row = d_buf + N * padded_dim_;
            CUDA_CHECK(cudaMemset(d_cent_row, 0, padded_dim_ * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_cent_row, d_centroid,
                                   dim_ * sizeof(float), cudaMemcpyDeviceToDevice));
        } else {
            CUDA_CHECK(cudaMemset(d_buf + N * padded_dim_, 0, padded_dim_ * sizeof(float)));
        }
    } else {
        // No padding needed — copy data directly, append centroid row
        CUDA_CHECK(cudaMemcpy(d_buf, d_data, N * padded_dim_ * sizeof(float), cudaMemcpyDeviceToDevice));
        if (d_centroid) {
            CUDA_CHECK(cudaMemcpy(d_buf + N * padded_dim_, d_centroid,
                                   padded_dim_ * sizeof(float), cudaMemcpyDeviceToDevice));
        } else {
            CUDA_CHECK(cudaMemset(d_buf + N * padded_dim_, 0, padded_dim_ * sizeof(float)));
        }
    }

    // Step 2: Rotate all (N+1) vectors at once
    float* d_rotated = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rotated, padded_bytes));
    rotator_.rotate(d_buf, d_rotated, N + 1);

    // Step 3: Subtract rotated centroid from each rotated data vector → d_residual
    float* d_rotated_cent = d_rotated + N * padded_dim_;
    {
        int total = static_cast<int>(N * padded_dim_);
        int grid = (total + block - 1) / block;
        sa_pad_and_subtract_kernel<<<grid, block>>>(
            d_rotated, d_rotated_cent, d_residual,
            static_cast<int>(N), static_cast<int>(padded_dim_), static_cast<int>(padded_dim_));
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaFree(d_rotated));
}

// ============================================================================
// Internal: launch quantize + factor kernels, templated on CodeT
// ============================================================================

template<typename CodeT>
static void launch_quantize_scalar(
    const float* d_residual, size_t N, uint32_t padded_dim, size_t ex_bits,
    float const_scaling_factor, bool use_fast, int delta_mode,
    CodeT* d_total_code, float* d_delta, float* d_vl)
{
    constexpr int block = 256;
    int iN = static_cast<int>(N);
    int iD = static_cast<int>(padded_dim);
    int iB = static_cast<int>(ex_bits);

    size_t q_smem = (block + padded_dim + 3 * block) * sizeof(float);
    sa_quantize_total_code_kernel<CodeT><<<iN, block, q_smem>>>(
        d_residual, d_total_code, iN, iD, iB, const_scaling_factor, use_fast);
    CUDA_CHECK(cudaGetLastError());

    size_t f_smem = 3 * block * sizeof(float);
    sa_compute_delta_vl_kernel<CodeT><<<iN, block, f_smem>>>(
        d_residual, d_total_code, d_delta, d_vl, iN, iD, iB, delta_mode);
    CUDA_CHECK(cudaGetLastError());
}

template<typename CodeT>
static void launch_quantize_full(
    const float* d_residual, const float* d_rotated_cent,
    size_t N, uint32_t padded_dim, size_t ex_bits,
    float const_scaling_factor, bool use_fast,
    CodeT* d_total_code, float* d_factors)
{
    constexpr int block = 256;
    int iN = static_cast<int>(N);
    int iD = static_cast<int>(padded_dim);
    int iB = static_cast<int>(ex_bits);

    size_t q_smem = (block + padded_dim + 3 * block) * sizeof(float);
    sa_quantize_total_code_kernel<CodeT><<<iN, block, q_smem>>>(
        d_residual, d_total_code, iN, iD, iB, const_scaling_factor, use_fast);
    CUDA_CHECK(cudaGetLastError());

    size_t f_smem = 4 * block * sizeof(float);
    sa_compute_full_factors_kernel<CodeT><<<iN, block, f_smem>>>(
        d_residual, d_rotated_cent, d_total_code, d_factors, iN, iD, iB);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// quantize_scalar
// ============================================================================

// --- uint16_t quantize_scalar ---

void StandaloneQuantizerGPU::quantize_scalar(
    const float* d_data, size_t N,
    uint16_t* d_total_code, float* d_delta, float* d_vl) const
{
    quantize_scalar(d_data, nullptr, N, d_total_code, d_delta, d_vl);
}

void StandaloneQuantizerGPU::quantize_scalar(
    const float* d_data, const float* d_centroid, size_t N,
    uint16_t* d_total_code, float* d_delta, float* d_vl) const
{
    float* d_residual = nullptr;
    CUDA_CHECK(cudaMalloc(&d_residual, N * padded_dim_ * sizeof(float)));
    compute_residuals(d_data, d_centroid, N, d_residual);

    launch_quantize_scalar(d_residual, N, padded_dim_, ex_bits_,
                            const_scaling_factor_, use_fast_quantize_, 0,
                            d_total_code, d_delta, d_vl);

    CUDA_CHECK(cudaFree(d_residual));
}

// --- uint8_t quantize_scalar (efficient for 1-8 bits) ---

void StandaloneQuantizerGPU::quantize_scalar(
    const float* d_data, size_t N,
    uint8_t* d_total_code, float* d_delta, float* d_vl) const
{
    quantize_scalar(d_data, nullptr, N, d_total_code, d_delta, d_vl);
}

void StandaloneQuantizerGPU::quantize_scalar(
    const float* d_data, const float* d_centroid, size_t N,
    uint8_t* d_total_code, float* d_delta, float* d_vl) const
{
    float* d_residual = nullptr;
    CUDA_CHECK(cudaMalloc(&d_residual, N * padded_dim_ * sizeof(float)));
    compute_residuals(d_data, d_centroid, N, d_residual);

    launch_quantize_scalar(d_residual, N, padded_dim_, ex_bits_,
                            const_scaling_factor_, use_fast_quantize_, 0,
                            d_total_code, d_delta, d_vl);

    CUDA_CHECK(cudaFree(d_residual));
}

// ============================================================================
// quantize_full
// ============================================================================

// --- uint16_t quantize_full ---

void StandaloneQuantizerGPU::quantize_full(
    const float* d_data, size_t N,
    uint16_t* d_total_code, float* d_factors) const
{
    quantize_full(d_data, nullptr, N, d_total_code, d_factors);
}

void StandaloneQuantizerGPU::quantize_full(
    const float* d_data, const float* d_centroid, size_t N,
    uint16_t* d_total_code, float* d_factors) const
{
    float* d_residual = nullptr;
    CUDA_CHECK(cudaMalloc(&d_residual, N * padded_dim_ * sizeof(float)));
    compute_residuals(d_data, d_centroid, N, d_residual);

    // Compute rotated centroid for factor computation
    float* d_rotated_cent = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rotated_cent, padded_dim_ * sizeof(float)));
    if (d_centroid) {
        float* d_cent_padded = nullptr;
        CUDA_CHECK(cudaMalloc(&d_cent_padded, padded_dim_ * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_cent_padded, 0, padded_dim_ * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_cent_padded, d_centroid,
                               dim_ * sizeof(float), cudaMemcpyDeviceToDevice));
        rotator_.rotate(d_cent_padded, d_rotated_cent, 1);
        CUDA_CHECK(cudaFree(d_cent_padded));
    } else {
        CUDA_CHECK(cudaMemset(d_rotated_cent, 0, padded_dim_ * sizeof(float)));
    }

    launch_quantize_full(d_residual, d_rotated_cent, N, padded_dim_, ex_bits_,
                          const_scaling_factor_, use_fast_quantize_,
                          d_total_code, d_factors);

    CUDA_CHECK(cudaFree(d_residual));
    CUDA_CHECK(cudaFree(d_rotated_cent));
}

// ============================================================================
// Free functions: quantize on pre-computed residuals (no rotation)
// ============================================================================

void standalone_quantize_full_on_residuals(
    const float* d_residuals, const float* d_centroid,
    size_t N, size_t padded_dim,
    size_t ex_bits, float const_scaling_factor, bool use_fast,
    uint16_t* d_total_code, float* d_factors)
{
    launch_quantize_full(d_residuals, d_centroid, N, static_cast<uint32_t>(padded_dim), ex_bits,
                          const_scaling_factor, use_fast,
                          d_total_code, d_factors);
}

// --- Fused warp-cooperative free functions ---

template<typename CodeT>
static void launch_quantize_fused(
    const float* d_residual, size_t N, uint32_t padded_dim, size_t ex_bits,
    float const_scaling_factor, bool use_fast, int delta_mode,
    CodeT* d_total_code, float* d_delta, float* d_vl)
{
    constexpr int block = 256;
    int iN = static_cast<int>(N);
    int iD = static_cast<int>(padded_dim);
    int iB = static_cast<int>(ex_bits);
    constexpr int nwarps = block / 32;

    // smem = kBlockSize (reduce) + padded_dim (abs_norm) + 2*kNWarps (warp results)
    size_t q_smem = (block + padded_dim + 2 * nwarps) * sizeof(float);
    sa_quantize_fused_kernel<CodeT, block><<<iN, block, q_smem>>>(
        d_residual, d_total_code, iN, iD, iB, const_scaling_factor, use_fast);
    CUDA_CHECK(cudaGetLastError());

    size_t f_smem = 3 * block * sizeof(float);
    sa_compute_delta_vl_kernel<CodeT><<<iN, block, f_smem>>>(
        d_residual, d_total_code, d_delta, d_vl, iN, iD, iB, delta_mode);
    CUDA_CHECK(cudaGetLastError());
}

void standalone_quantize_fused_on_residuals(
    const float* d_residuals, size_t N, size_t padded_dim,
    size_t ex_bits, float const_scaling_factor, bool use_fast,
    uint16_t* d_total_code, float* d_delta, float* d_vl, int delta_mode)
{
    launch_quantize_fused(d_residuals, N, static_cast<uint32_t>(padded_dim), ex_bits,
                           const_scaling_factor, use_fast, delta_mode,
                           d_total_code, d_delta, d_vl);
}

void standalone_quantize_fused_on_residuals(
    const float* d_residuals, size_t N, size_t padded_dim,
    size_t ex_bits, float const_scaling_factor, bool use_fast,
    uint8_t* d_total_code, float* d_delta, float* d_vl, int delta_mode)
{
    launch_quantize_fused(d_residuals, N, static_cast<uint32_t>(padded_dim), ex_bits,
                           const_scaling_factor, use_fast, delta_mode,
                           d_total_code, d_delta, d_vl);
}

// --- uint8_t quantize_full (efficient for 1-8 bits) ---

void StandaloneQuantizerGPU::quantize_full(
    const float* d_data, size_t N,
    uint8_t* d_total_code, float* d_factors) const
{
    quantize_full(d_data, nullptr, N, d_total_code, d_factors);
}

void StandaloneQuantizerGPU::quantize_full(
    const float* d_data, const float* d_centroid, size_t N,
    uint8_t* d_total_code, float* d_factors) const
{
    float* d_residual = nullptr;
    CUDA_CHECK(cudaMalloc(&d_residual, N * padded_dim_ * sizeof(float)));
    compute_residuals(d_data, d_centroid, N, d_residual);

    float* d_rotated_cent = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rotated_cent, padded_dim_ * sizeof(float)));
    if (d_centroid) {
        float* d_cent_padded = nullptr;
        CUDA_CHECK(cudaMalloc(&d_cent_padded, padded_dim_ * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_cent_padded, 0, padded_dim_ * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_cent_padded, d_centroid,
                               dim_ * sizeof(float), cudaMemcpyDeviceToDevice));
        rotator_.rotate(d_cent_padded, d_rotated_cent, 1);
        CUDA_CHECK(cudaFree(d_cent_padded));
    } else {
        CUDA_CHECK(cudaMemset(d_rotated_cent, 0, padded_dim_ * sizeof(float)));
    }

    launch_quantize_full(d_residual, d_rotated_cent, N, padded_dim_, ex_bits_,
                          const_scaling_factor_, use_fast_quantize_,
                          d_total_code, d_factors);

    CUDA_CHECK(cudaFree(d_residual));
    CUDA_CHECK(cudaFree(d_rotated_cent));
}
