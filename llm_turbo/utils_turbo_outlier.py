import math

import torch

def repeat_kv_quant(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def repeat_indices(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :].expand(batch, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, head_dim)


def split_outliers_and_residual(key_states, top_indices):
    b, h, n, d = key_states.shape
    k = top_indices.size(-1)

    top_idx_expanded = top_indices.unsqueeze(2).expand(b, h, n, k)
    outliers = torch.gather(key_states, dim=-1, index=top_idx_expanded)
    mask = torch.zeros(b, h, d, dtype=torch.bool, device=key_states.device)
    mask.scatter_(-1, top_indices, True)

    mask_expanded = mask.unsqueeze(2).expand(b, h, n, d)
    residual = key_states[~mask_expanded].view(b, h, n, d - k)

    return outliers, residual

def round_to_nearest_centroid(data: torch.Tensor, bitwidth: int) -> torch.Tensor:
    if bitwidth not in [1, 2, 3, 4, 5]:
        raise ValueError("Bitwidth must be one of [1, 2, 3, 4, 5]")
    centroids = [
        torch.tensor([-0.797885, 0.797885]),
        torch.tensor([-1.510017, -0.4526475, 0.4526475, 1.510017]),
        torch.tensor([-2.1509, -1.34335, -0.75567, -0.244893, 0.244961, 0.75567, 1.34335, 2.1509]),
        torch.tensor([-2.7235756,
                      -2.0604305,
                      -1.6096783,
                      -1.2484536,
                      -0.9357067,
                      -0.6516434,
                      -0.3848085,
                      -0.12730813,
                      0.12730813,
                      0.3848085,
                      0.6516434,
                      0.9357067,
                      1.2484536,
                      1.6096783,
                      2.0604305,
                      2.7235756]),
        torch.tensor([-3.0996535,
                      -2.5120323,
                      -2.1263952,
                      -1.829787,
                      -1.5848435,
                      -1.374458,
                      -1.1892526,
                      -1.0232878,
                      -0.872524,
                      -0.7339456,
                      -0.60524833,
                      -0.48459405,
                      -0.37025818,
                      -0.26076207,
                      -0.1548669,
                      -0.05133244,
                      0.05133244,
                      0.1548669,
                      0.26076207,
                      0.37025818,
                      0.48459405,
                      0.60524833,
                      0.7339456,
                      0.872524,
                      1.0232878,
                      1.1892526,
                      1.374458,
                      1.5848435,
                      1.829787,
                      2.1263952,
                      2.5120323,
                      3.0996535])
    ]
    selected_centroids = centroids[bitwidth - 1].to(device=data.device, dtype=data.dtype) / math.sqrt(128)
    distances = torch.abs(data.unsqueeze(-1) - selected_centroids)
    closest_indices = torch.argmin(distances, dim=-1)

    return selected_centroids[closest_indices]


class TurboSketch(torch.nn.Module):
    def __init__(self, dimension, bit_width, device=None, rng=None, dtype=torch.bfloat16):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bit_width = bit_width
        self.dimension = dimension
        self.random_gaussian = torch.randn((bit_width, dimension, dimension),
                                           device=self.device, generator=rng)
        self.proj_dir = self.init_rot_dir().to(dtype).contiguous()

    def init_rot_dir(self):
        rot_dir = []
        for it in range(self.bit_width):
            q, _ = torch.linalg.qr(self.random_gaussian[it, :, :], mode='reduced')
            rot_dir.append(q)
        return torch.stack(rot_dir)

    def quantize(self, keys):
        assert keys.shape[-1] == self.dimension, 'embedding dimension should match projection dimension'
        rotated_vectors = keys @ self.proj_dir[0, :, :].T
        quantized_vectors = round_to_nearest_centroid(rotated_vectors, self.bit_width).to(self.proj_dir.dtype)
        key_quant = (quantized_vectors @ self.proj_dir[0, :, :])
        return key_quant

    def calc_score(self, query, data_quant, norm_data):
        assert query.shape[-1] == self.dimension, 'embedding dimension should match projection dimension'
        h_k = data_quant.shape[1]
        h = query.shape[1]

        keys_repeat_norm = repeat_kv_quant(norm_data.unsqueeze(-1), n_rep=h // h_k)
        keys_repeat = repeat_kv_quant(data_quant, n_rep=h // h_k) * keys_repeat_norm
        scores = torch.matmul(query, keys_repeat.transpose(-1, -2))
        return scores


class TurboKeyQuantizer:
    def __init__(self, qjl_residual: TurboSketch, qjl_outlier: TurboSketch, buffer_size: int, group_size: int,
                 bit_width: int, top_channels: int = 32) -> None:
        self.qjl_residual = qjl_residual
        self.qjl_outlier = qjl_outlier

        self.buffer_size = buffer_size
        self.group_size = group_size
        self.bit_width = bit_width
        self.seq_len = None

        self.residual_quant_binary = None
        self.outliers_quant_binary = None

        self.residual_norm = None
        self.outliers_norm = None

        self.key_buffered = None
        self.quantized_len = 0

        self.bit_pack_len = 8
        self.top_channels = top_channels
        self.outliers_indices = None

    def build_sketch(self, key_states: torch.Tensor) -> None:
        b, h, _, dim = key_states.shape
        self.seq_len = key_states.shape[-2]
        residual_size = self.seq_len % self.buffer_size

        if residual_size > 0:
            self.key_buffered = key_states[:, :, self.seq_len - residual_size:, :]
        if residual_size == self.seq_len:
            self.quantized_len = 0
            return None

        self.quantized_len = self.seq_len - residual_size

        norms_channel = torch.norm(key_states, dim=-2)
        self.outliers_indices = torch.topk(norms_channel, k=self.top_channels, dim=-1, largest=True).indices

        key_states = key_states[:, :, :self.seq_len - residual_size, :].reshape((b, h, -1, dim)).contiguous()

        outliers, residual = split_outliers_and_residual(key_states, self.outliers_indices)

        self.residual_norm = torch.norm(residual, dim=-1).clamp_min(1e-8)
        self.outliers_norm = torch.norm(outliers, dim=-1).clamp_min(1e-8)

        self.residual_quant_binary = self.qjl_residual.quantize(residual / self.residual_norm.unsqueeze(-1))
        self.outliers_quant_binary = self.qjl_outlier.quantize(outliers / self.outliers_norm.unsqueeze(-1))

    def _flush_buffer(self) -> None:
        b, h, n, dim = self.key_buffered.shape

        if self.outliers_indices is None:
            norms_channel = torch.norm(self.key_buffered, dim=-2)
            self.outliers_indices = torch.topk(norms_channel, k=self.top_channels, dim=-1, largest=True).indices

        outliers, residual = split_outliers_and_residual(self.key_buffered, self.outliers_indices)

        res_norm = torch.norm(residual, dim=-1).clamp_min(1e-8)
        out_norm = torch.norm(outliers, dim=-1).clamp_min(1e-8)
        res_quant = self.qjl_residual.quantize(residual / res_norm.unsqueeze(-1))
        out_quant = self.qjl_outlier.quantize(outliers / out_norm.unsqueeze(-1))

        if self.residual_quant_binary is not None:
            self.residual_quant_binary = torch.cat([self.residual_quant_binary, res_quant], dim=2)
            self.outliers_quant_binary = torch.cat([self.outliers_quant_binary, out_quant], dim=2)
            self.residual_norm = torch.cat([self.residual_norm, res_norm], dim=2)
            self.outliers_norm = torch.cat([self.outliers_norm, out_norm], dim=2)
        else:
            self.residual_quant_binary = res_quant
            self.outliers_quant_binary = out_quant
            self.residual_norm = res_norm
            self.outliers_norm = out_norm

        self.quantized_len += n
        self.key_buffered = None

    def update_sketch(self, key_states: torch.Tensor) -> None:
        assert key_states.shape[-2] == 1, 'appending more than one embedding in the stream!'
        self.seq_len += 1

        if self.key_buffered is not None:
            self.key_buffered = torch.cat([self.key_buffered, key_states], dim=-2)
        else:
            self.key_buffered = key_states

        if self.key_buffered.shape[-2] >= self.buffer_size:
            self._flush_buffer()

    def attention_score(self, query_states: torch.Tensor) -> torch.Tensor:
        b, h, _, dim = query_states.shape
        assert query_states.shape[-2] == 1, 'appending more than one embedding in the stream!'
        residual = None
        if self.key_buffered != None:
            h_k = self.key_buffered.shape[1]
            residual = repeat_kv_quant(self.key_buffered, n_rep=h // h_k)
            residual = torch.matmul(query_states, residual.transpose(-1, -2))

        if self.quantized_len == 0 or self.outliers_quant_binary is None:
            return residual

        h_k = self.outliers_indices.shape[1]
        query_outlier, query_residual = split_outliers_and_residual(query_states,
                                                                    repeat_indices(self.outliers_indices, h // h_k))
        scores = self.qjl_residual.calc_score(query_residual, self.residual_quant_binary, self.residual_norm)
        scores += self.qjl_outlier.calc_score(query_outlier, self.outliers_quant_binary, self.outliers_norm)

        if residual != None:
            return torch.cat([scores, residual], dim=-1)
        return scores


class TurboValueQuantizer:
    def __init__(self, quantizer_value: TurboSketch, buffer_size: int, group_size: int,
                 bit_width: int) -> None:
        self.quantizer_value = quantizer_value

        self.buffer_size = buffer_size
        self.group_size = group_size
        self.bit_width = bit_width
        self.seq_len = None

        self.quant_binary = None

        self.quant_norm = None

        self.value_buffered = None
        self.quantized_len = 0

    def build_sketch(self, value_states: torch.Tensor) -> None:
        b, h, _, dim = value_states.shape
        self.seq_len = value_states.shape[-2]
        residual_size = self.seq_len % self.buffer_size

        if residual_size > 0:
            self.value_buffered = value_states[:, :, self.seq_len - residual_size:, :]
        if residual_size == self.seq_len:
            self.quantized_len = 0
            return None

        self.quantized_len = self.seq_len - residual_size
        value_states = value_states[:, :, :self.quantized_len, :]
        self.value_states_norm = torch.norm(value_states, dim=-1).clamp_min(1e-8)

        self.value_states_quant = self.quantizer_value.quantize(value_states / self.value_states_norm.unsqueeze(-1))

    def _flush_buffer(self) -> None:
        b, h, n, dim = self.value_buffered.shape

        val_norm = torch.norm(self.value_buffered, dim=-1).clamp_min(1e-8)
        val_quant = self.quantizer_value.quantize(self.value_buffered / val_norm.unsqueeze(-1))

        if hasattr(self, 'value_states_quant') and self.value_states_quant is not None:
            self.value_states_quant = torch.cat([self.value_states_quant, val_quant], dim=2)
            self.value_states_norm = torch.cat([self.value_states_norm, val_norm], dim=2)
        else:
            self.value_states_quant = val_quant
            self.value_states_norm = val_norm

        self.quantized_len += n
        self.value_buffered = None

    def update_sketch(self, value_states: torch.Tensor) -> None:
        assert value_states.shape[-2] == 1, 'appending more than one embedding in the stream!'
        self.seq_len += 1

        if self.value_buffered is not None:
            self.value_buffered = torch.cat([self.value_buffered, value_states], dim=-2)
        else:
            self.value_buffered = value_states

        if self.value_buffered.shape[-2] >= self.buffer_size:
            self._flush_buffer()

    def attention_score(self, att: torch.Tensor) -> torch.Tensor:
        b, h, _, dim = att.shape
        residual = None
        res_len = self.quantized_len
        if self.value_buffered != None:
            h_k = self.value_buffered.shape[1]
            value_buffered_repeat = repeat_kv_quant(self.value_buffered, n_rep=h // h_k)
            residual = torch.matmul(att[:, :, :, res_len:], value_buffered_repeat)

        if res_len == 0:
            return residual

        h_k = self.value_states_quant.shape[1]
        values_repeat = repeat_kv_quant(self.value_states_quant, n_rep=h // h_k)
        norms_repeat = repeat_kv_quant(self.value_states_norm.unsqueeze(-1), n_rep=h // h_k)
        scores = torch.matmul(att[:, :, :, :res_len], values_repeat * norms_repeat)

        if residual != None:
            return scores + residual
        return scores
