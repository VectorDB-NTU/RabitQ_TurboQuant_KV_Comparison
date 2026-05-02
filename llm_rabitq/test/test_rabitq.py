"""
Tests for rabitq CUDA extension.

Usage:
    cd llm_rabitq
    pip install -e .
    python -m pytest test/ -v
"""

import pytest
import torch
import time
from pathlib import Path
import sys

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


class TestImport:
    def test_import(self):
        _require_cuda()
        import rabitq
        assert hasattr(rabitq, 'quantize')


class TestQuantizeBasic:
    """Basic API contract tests."""

    def test_output_shapes(self):
        _require_cuda()
        import rabitq
        N, D = 64, 128
        x = torch.randn(N, D, device='cuda', dtype=torch.float32)
        codes, delta, vl = rabitq.quantize(x, total_bits=2)
        assert codes.shape == (N, D)
        assert delta.shape == (N,)
        assert vl.shape == (N,)

    def test_output_devices(self):
        _require_cuda()
        import rabitq
        x = torch.randn(32, 128, device='cuda', dtype=torch.float32)
        codes, delta, vl = rabitq.quantize(x, total_bits=2)
        assert codes.is_cuda
        assert delta.is_cuda
        assert vl.is_cuda

    def test_codes_dtype(self):
        _require_cuda()
        import rabitq
        x = torch.randn(32, 128, device='cuda', dtype=torch.float32)
        codes, delta, vl = rabitq.quantize(x, total_bits=2)
        assert codes.dtype == torch.uint8
        assert delta.dtype == torch.float32
        assert vl.dtype == torch.float32


class TestQuantizeBitwidths:
    """Test all supported bitwidths (1-8)."""

    @pytest.mark.parametrize("bits", [1, 2, 3, 4, 5, 6, 7, 8])
    def test_bitwidth(self, bits):
        _require_cuda()
        import rabitq
        N, D = 32, 128
        x = torch.randn(N, D, device='cuda', dtype=torch.float32)
        codes, delta, vl = rabitq.quantize(x, total_bits=bits)
        assert codes.shape == (N, D)
        max_val = (1 << bits) - 1
        assert codes.max().item() <= max_val


class TestReconstruction:
    """Test quantize -> reconstruct quality."""

    def _reconstruct(self, codes, delta, vl):
        return codes.float() * delta.unsqueeze(-1) + vl.unsqueeze(-1)

    def test_reconstruction_not_zero(self):
        _require_cuda()
        import rabitq
        x = torch.randn(64, 128, device='cuda', dtype=torch.float32)
        codes, delta, vl = rabitq.quantize(x, total_bits=4)
        recon = self._reconstruct(codes, delta, vl)
        assert recon.abs().sum() > 0

    def test_higher_bits_lower_error(self):
        """More bits should give lower reconstruction error."""
        _require_cuda()
        import rabitq
        torch.manual_seed(42)
        x = torch.randn(256, 128, device='cuda', dtype=torch.float32)

        errors = []
        for bits in [1, 2, 4]:
            codes, delta, vl = rabitq.quantize(x, total_bits=bits)
            recon = self._reconstruct(codes, delta, vl)
            err = (x - recon).norm() / x.norm()
            errors.append(err.item())

        assert errors[0] > errors[1] > errors[2], f"Errors should decrease: {errors}"

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_reconstruction_error_bounded(self, bits):
        """Relative error should be reasonable (< 1.0 for bits >= 2)."""
        _require_cuda()
        import rabitq
        x = torch.randn(128, 128, device='cuda', dtype=torch.float32)
        codes, delta, vl = rabitq.quantize(x, total_bits=bits)
        recon = self._reconstruct(codes, delta, vl)
        rel_err = (x - recon).norm() / x.norm()
        assert rel_err.item() < 1.0, f"Relative error too high: {rel_err.item()}"


class TestDimensions:
    """Test with KV cache relevant dimensions."""

    @pytest.mark.parametrize("dim", [32, 64, 96, 128])
    def test_kv_dimensions(self, dim):
        _require_cuda()
        import rabitq
        N = 64
        x = torch.randn(N, dim, device='cuda', dtype=torch.float32)
        codes, delta, vl = rabitq.quantize(x, total_bits=2)
        assert codes.shape == (N, dim)

    def test_non_aligned_dimension(self):
        """Dimensions not multiple of 64 should still work (internal padding)."""
        _require_cuda()
        import rabitq
        x = torch.randn(32, 100, device='cuda', dtype=torch.float32)
        codes, delta, vl = rabitq.quantize(x, total_bits=2)
        assert codes.shape == (32, 100)

    def test_single_vector(self):
        _require_cuda()
        import rabitq
        x = torch.randn(1, 128, device='cuda', dtype=torch.float32)
        codes, delta, vl = rabitq.quantize(x, total_bits=2)
        assert codes.shape == (1, 128)

    def test_large_batch(self):
        _require_cuda()
        import rabitq
        x = torch.randn(4096, 128, device='cuda', dtype=torch.float32)
        codes, delta, vl = rabitq.quantize(x, total_bits=2)
        assert codes.shape == (4096, 128)


class TestInputValidation:
    """Test error handling."""

    def test_cpu_tensor_rejected(self):
        _require_cuda()
        import rabitq
        x = torch.randn(32, 128, dtype=torch.float32)
        with pytest.raises(RuntimeError, match="CUDA"):
            rabitq.quantize(x, total_bits=2)

    def test_wrong_dtype_rejected(self):
        _require_cuda()
        import rabitq
        x = torch.randn(32, 128, device='cuda', dtype=torch.float16)
        with pytest.raises(RuntimeError, match="float32"):
            rabitq.quantize(x, total_bits=2)

    def test_1d_rejected(self):
        _require_cuda()
        import rabitq
        x = torch.randn(128, device='cuda', dtype=torch.float32)
        with pytest.raises(RuntimeError, match="2D"):
            rabitq.quantize(x, total_bits=2)

    def test_bits_out_of_range(self):
        _require_cuda()
        import rabitq
        x = torch.randn(32, 128, device='cuda', dtype=torch.float32)
        with pytest.raises(RuntimeError):
            rabitq.quantize(x, total_bits=0)
        with pytest.raises(RuntimeError):
            rabitq.quantize(x, total_bits=9)


class TestDeterminism:
    """Test reproducibility."""

    def test_same_input_same_output(self):
        _require_cuda()
        import rabitq
        torch.manual_seed(42)
        x = torch.randn(64, 128, device='cuda', dtype=torch.float32)
        c1, d1, v1 = rabitq.quantize(x, total_bits=2, seed=42)
        c2, d2, v2 = rabitq.quantize(x, total_bits=2, seed=42)
        assert torch.equal(c1, c2)
        assert torch.equal(d1, d2)
        assert torch.equal(v1, v2)

    def test_same_seed_reproducible_across_seconds(self):
        """Guard against time-based seeding inside the CUDA extension."""
        _require_cuda()
        import rabitq
        torch.manual_seed(42)
        x = torch.randn(64, 128, device='cuda', dtype=torch.float32)

        c1, d1, v1 = rabitq.quantize(x, total_bits=2, seed=42)
        time.sleep(1.1)
        c2, d2, v2 = rabitq.quantize(x, total_bits=2, seed=42)

        assert torch.equal(c1, c2)
        assert torch.equal(d1, d2)
        assert torch.equal(v1, v2)

    def test_different_seed_changes_fast_scaling_factor(self):
        _require_cuda()
        import rabitq
        torch.manual_seed(42)
        x = torch.randn(64, 128, device='cuda', dtype=torch.float32)

        _, d1, v1 = rabitq.quantize(x, total_bits=2, seed=42)
        _, d2, v2 = rabitq.quantize(x, total_bits=2, seed=43)

        assert not (torch.equal(d1, d2) and torch.equal(v1, v2))

    def test_sketch_rng_reproducible_across_seconds(self):
        """Match the LLM path: RaBitQSketch gets seeds from a fixed torch.Generator."""
        _require_cuda()
        from kvcache_quant.rabitq_sketch import RaBitQSketch

        torch.manual_seed(42)
        x = torch.randn(64, 128, device='cuda', dtype=torch.float32)

        gen1 = torch.Generator(device='cuda')
        gen1.manual_seed(42)
        sketch1 = RaBitQSketch(dimension=128, bit_width=2, rng=gen1)
        q1 = sketch1.quantize(x)

        time.sleep(1.1)

        gen2 = torch.Generator(device='cuda')
        gen2.manual_seed(42)
        sketch2 = RaBitQSketch(dimension=128, bit_width=2, rng=gen2)
        q2 = sketch2.quantize(x)

        assert sketch1.seed == sketch2.seed
        assert torch.equal(sketch1.rotation, sketch2.rotation)
        assert torch.equal(q1, q2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
