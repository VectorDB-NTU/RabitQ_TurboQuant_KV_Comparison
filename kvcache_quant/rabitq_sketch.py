import numpy as np
import torch

import rabitq

from .kv_quantizer import repeat_kv_quant


def _make_rotation(dim, seed, device):
    rng = np.random.default_rng(seed)
    H = rng.standard_normal((dim, dim)).astype(np.float64)
    Q, R = np.linalg.qr(H)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    return torch.from_numpy(Q.astype(np.float32)).to(device)


class RaBitQSketch(torch.nn.Module):
    needs_norm = False  # RaBitQ handles scale via delta/vl internally

    def __init__(self, dimension, bit_width, device=None, rng=None, dtype=torch.bfloat16, seed=None):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bit_width = bit_width
        self.dimension = dimension
        self.dtype = dtype

        seed = 42 + dimension * 100 + bit_width if seed is None else seed
        if rng is not None:
            seed = int(torch.empty(1, device=self.device, dtype=torch.int64).random_(0, 2**31, generator=rng).item())
        self.seed = int(seed)
        rotation = _make_rotation(dimension, self.seed, self.device)
        self.register_buffer('rotation', rotation)

    def quantize(self, keys):
        assert keys.shape[-1] == self.dimension
        shape = keys.shape
        flat = keys.reshape(-1, self.dimension).float()

        rotated = flat @ self.rotation.T

        codes, delta, vl = rabitq.quantize(rotated, self.bit_width, seed=self.seed)
        recon = codes.float() * delta.unsqueeze(-1) + vl.unsqueeze(-1)

        dequant = recon @ self.rotation

        return dequant.reshape(shape).to(keys.dtype)

    def calc_score(self, query, data_quant, norm_data):
        assert query.shape[-1] == self.dimension
        h_k = data_quant.shape[1]
        h = query.shape[1]
        keys_repeat = repeat_kv_quant(data_quant, n_rep=h // h_k)
        scores = torch.matmul(query, keys_repeat.transpose(-1, -2))
        return scores
