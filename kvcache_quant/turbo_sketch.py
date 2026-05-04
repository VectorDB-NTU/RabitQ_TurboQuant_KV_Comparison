import math

import torch

from .kv_quantizer import repeat_kv_quant


def round_to_nearest_centroid(data: torch.Tensor, bitwidth: int) -> torch.Tensor:
    if bitwidth not in [1, 2, 3, 4, 5]:
        raise ValueError("Bitwidth must be one of [1, 2, 3, 4, 5]")
    centroids = [
        torch.tensor([-0.797885, 0.797885]),
        torch.tensor([-1.510017, -0.4526475, 0.4526475, 1.510017]),
        torch.tensor([-2.1509, -1.34335, -0.75567, -0.244893, 0.244961, 0.75567, 1.34335, 2.1509]),
        torch.tensor([-2.7235756, -2.0604305, -1.6096783, -1.2484536, -0.9357067, -0.6516434, -0.3848085, -0.12730813, 0.12730813, 0.3848085, 0.6516434, 0.9357067, 1.2484536, 1.6096783, 2.0604305, 2.7235756]),
        torch.tensor([-3.0996535, -2.5120323, -2.1263952, -1.829787, -1.5848435, -1.374458, -1.1892526, -1.0232878, -0.872524, -0.7339456, -0.60524833, -0.48459405, -0.37025818, -0.26076207, -0.1548669, -0.05133244, 0.05133244, 0.1548669, 0.26076207, 0.37025818, 0.48459405, 0.60524833, 0.7339456, 0.872524, 1.0232878, 1.1892526, 1.374458, 1.5848435, 1.829787, 2.1263952, 2.5120323, 3.0996535]),
    ]
    selected_centroids = centroids[bitwidth - 1].to(device=data.device, dtype=data.dtype) / math.sqrt(128)
    distances = torch.abs(data.unsqueeze(-1) - selected_centroids)
    closest_indices = torch.argmin(distances, dim=-1)
    return selected_centroids[closest_indices]


class TurboSketch(torch.nn.Module):
    needs_norm = True  # TurboQuant requires norm-normalized input

    def __init__(self, dimension, bit_width, device=None, rng=None, dtype=torch.bfloat16):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bit_width = bit_width
        self.dimension = dimension
        self.random_gaussian = torch.randn((bit_width, dimension, dimension),
                                           device=self.device, generator=rng)
        self.proj_dir = self._init_rot_dir().to(dtype).contiguous()

    def _init_rot_dir(self):
        rot_dir = []
        for it in range(self.bit_width):
            q, _ = torch.linalg.qr(self.random_gaussian[it, :, :], mode='reduced')
            rot_dir.append(q)
        return torch.stack(rot_dir)

    def quantize(self, keys):
        # Simple test implementation: quantize then immediately dequantize to observe
        # the error introduced by quantization. This does not save memory or compute.
        # In practice, one can compute inner products directly in the rotated space
        # using the codebook, without rotating back to the original space.
        assert keys.shape[-1] == self.dimension
        rotated_vectors = keys @ self.proj_dir[0, :, :].T
        quantized_vectors = round_to_nearest_centroid(rotated_vectors, self.bit_width).to(self.proj_dir.dtype)
        key_quant = quantized_vectors @ self.proj_dir[0, :, :]
        return key_quant

    def calc_score(self, query, data_quant, norm_data):
        assert query.shape[-1] == self.dimension
        h_k = data_quant.shape[1]
        h = query.shape[1]
        keys_repeat_norm = repeat_kv_quant(norm_data.unsqueeze(-1), n_rep=h // h_k)
        keys_repeat = repeat_kv_quant(data_quant, n_rep=h // h_k) * keys_repeat_norm
        scores = torch.matmul(query, keys_repeat.transpose(-1, -2))
        return scores
