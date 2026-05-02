/*
 * Python binding for RaBitQ GPU quantization.
 *
 * API:
 *   rabitq.quantize(residuals, total_bits, fast=True, seed=42)
 *     -> (codes, delta, vl)
 *
 *   Reconstruction: recon[i,j] = float(codes[i,j]) * delta[i] + vl[i]
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include "quantizer/quantizer_standalone.cuh"
#include "quantizer/rescale_search_gpu.cuh"

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
quantize(torch::Tensor residuals, int64_t total_bits, bool fast, uint64_t seed) {
    TORCH_CHECK(residuals.is_cuda(), "residuals must be a CUDA tensor");
    TORCH_CHECK(residuals.dtype() == torch::kFloat32, "residuals must be float32");
    TORCH_CHECK(residuals.dim() == 2, "residuals must be 2D (N, D)");
    TORCH_CHECK(total_bits >= 1 && total_bits <= 8, "total_bits must be 1-8");

    auto data = residuals.contiguous();
    at::cuda::CUDAGuard device_guard(data.device());

    int64_t N = data.size(0);
    int64_t D = data.size(1);
    uint32_t padded_D = ((static_cast<uint32_t>(D) + 63u) / 64u) * 64u;
    size_t ex_bits = static_cast<size_t>(total_bits - 1);

    // Pad if needed
    torch::Tensor padded;
    if (static_cast<int64_t>(padded_D) != D) {
        padded = torch::zeros({N, static_cast<int64_t>(padded_D)},
                              torch::dtype(torch::kFloat32).device(data.device()));
        padded.index({torch::indexing::Slice(), torch::indexing::Slice(0, D)}) = data;
    } else {
        padded = data;
    }

    // Compute const_scaling_factor if fast mode
    float const_sf = 0.0f;
    if (fast && ex_bits > 0) {
        const_sf = rabitq_get_const_scaling_factor_gpu(padded_D, ex_bits, seed);
    }

    // Allocate outputs
    auto codes = torch::empty({N, static_cast<int64_t>(padded_D)},
                              torch::dtype(torch::kUInt8).device(data.device()));
    auto delta = torch::empty({N}, torch::dtype(torch::kFloat32).device(data.device()));
    auto vl = torch::empty({N}, torch::dtype(torch::kFloat32).device(data.device()));

    // Quantize
    standalone_quantize_fused_on_residuals(
        padded.data_ptr<float>(),
        static_cast<size_t>(N),
        static_cast<size_t>(padded_D),
        ex_bits,
        const_sf,
        fast,
        codes.data_ptr<uint8_t>(),
        delta.data_ptr<float>(),
        vl.data_ptr<float>(),
        0);  // delta_mode=RECONSTRUCTION

    // Trim codes to original dimension
    if (static_cast<int64_t>(padded_D) != D) {
        codes = codes.index({torch::indexing::Slice(), torch::indexing::Slice(0, D)}).contiguous();
    }

    return std::make_tuple(codes, delta, vl);
}

PYBIND11_MODULE(rabitq, m) {
    m.doc() = "RaBitQ GPU scalar quantization";
    m.def("quantize", &quantize,
          py::arg("residuals"),
          py::arg("total_bits"),
          py::arg("fast") = true,
          py::arg("seed") = 42,
          "Quantize pre-rotated residuals (N, D) float32 CUDA.\n"
          "Returns (codes, delta, vl). Recon: codes * delta[:, None] + vl[:, None]");
}
