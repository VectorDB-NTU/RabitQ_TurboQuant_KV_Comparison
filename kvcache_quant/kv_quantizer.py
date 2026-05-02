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


class KeyQuantizer:
    def __init__(self, sketch_residual, sketch_outlier, buffer_size, group_size, bit_width, top_channels=32):
        self.sketch_residual = sketch_residual
        self.sketch_outlier = sketch_outlier

        self.buffer_size = buffer_size
        self.group_size = group_size
        self.bit_width = bit_width
        self.seq_len = None

        self.residual_quant = None
        self.outliers_quant = None
        self.residual_norm = None
        self.outliers_norm = None

        self.key_buffered = None
        self.quantized_len = 0

        self.top_channels = top_channels
        self.outliers_indices = None

    def build_sketch(self, key_states):
        b, h, _, dim = key_states.shape
        self.seq_len = key_states.shape[-2]
        self._needs_norm = self.sketch_residual.needs_norm
        residual_size = self.seq_len % self.buffer_size

        if residual_size > 0:
            self.key_buffered = key_states[:, :, self.seq_len - residual_size:, :]
        if residual_size == self.seq_len:
            self.quantized_len = 0
            return None

        self.quantized_len = self.seq_len - residual_size

        norms_channel = torch.norm(key_states, dim=-2)
        self.outliers_indices = torch.topk(norms_channel, k=self.top_channels, dim=-1, largest=True).indices

        key_states = key_states[:, :, :self.quantized_len, :].reshape((b, h, -1, dim)).contiguous()

        outliers, residual = split_outliers_and_residual(key_states, self.outliers_indices)

        if self._needs_norm:
            self.residual_norm = torch.norm(residual, dim=-1).clamp_min(1e-8)
            self.outliers_norm = torch.norm(outliers, dim=-1).clamp_min(1e-8)
            self.residual_quant = self.sketch_residual.quantize(residual / self.residual_norm.unsqueeze(-1))
            self.outliers_quant = self.sketch_outlier.quantize(outliers / self.outliers_norm.unsqueeze(-1))
        else:
            self.residual_norm = None
            self.outliers_norm = None
            self.residual_quant = self.sketch_residual.quantize(residual)
            self.outliers_quant = self.sketch_outlier.quantize(outliers)

    def _flush_buffer(self):
        b, h, n, dim = self.key_buffered.shape

        if self.outliers_indices is None:
            norms_channel = torch.norm(self.key_buffered, dim=-2)
            self.outliers_indices = torch.topk(norms_channel, k=self.top_channels, dim=-1, largest=True).indices

        outliers, residual = split_outliers_and_residual(self.key_buffered, self.outliers_indices)

        if self._needs_norm:
            res_norm = torch.norm(residual, dim=-1).clamp_min(1e-8)
            out_norm = torch.norm(outliers, dim=-1).clamp_min(1e-8)
            res_quant = self.sketch_residual.quantize(residual / res_norm.unsqueeze(-1))
            out_quant = self.sketch_outlier.quantize(outliers / out_norm.unsqueeze(-1))
        else:
            res_norm = None
            out_norm = None
            res_quant = self.sketch_residual.quantize(residual)
            out_quant = self.sketch_outlier.quantize(outliers)

        if self.residual_quant is not None:
            self.residual_quant = torch.cat([self.residual_quant, res_quant], dim=2)
            self.outliers_quant = torch.cat([self.outliers_quant, out_quant], dim=2)
            if self._needs_norm:
                self.residual_norm = torch.cat([self.residual_norm, res_norm], dim=2)
                self.outliers_norm = torch.cat([self.outliers_norm, out_norm], dim=2)
        else:
            self.residual_quant = res_quant
            self.outliers_quant = out_quant
            if self._needs_norm:
                self.residual_norm = res_norm
                self.outliers_norm = out_norm

        self.quantized_len += n
        self.key_buffered = None

    def update_sketch(self, key_states):
        assert key_states.shape[-2] == 1
        self.seq_len += 1

        if self.key_buffered is not None:
            self.key_buffered = torch.cat([self.key_buffered, key_states], dim=-2)
        else:
            self.key_buffered = key_states

        if self.key_buffered.shape[-2] >= self.buffer_size:
            self._flush_buffer()

    def attention_score(self, query_states):
        b, h, _, dim = query_states.shape
        assert query_states.shape[-2] == 1
        residual = None
        if self.key_buffered is not None:
            h_k = self.key_buffered.shape[1]
            residual = repeat_kv_quant(self.key_buffered, n_rep=h // h_k)
            residual = torch.matmul(query_states, residual.transpose(-1, -2))

        if self.quantized_len == 0 or self.outliers_quant is None:
            return residual

        h_k = self.outliers_indices.shape[1]
        query_outlier, query_residual = split_outliers_and_residual(
            query_states, repeat_indices(self.outliers_indices, h // h_k))
        scores = self.sketch_residual.calc_score(query_residual, self.residual_quant, self.residual_norm)
        scores += self.sketch_outlier.calc_score(query_outlier, self.outliers_quant, self.outliers_norm)

        if residual is not None:
            return torch.cat([scores, residual], dim=-1)
        return scores


class ValueQuantizer:
    def __init__(self, sketch_value, buffer_size, group_size, bit_width):
        self.sketch_value = sketch_value

        self.buffer_size = buffer_size
        self.group_size = group_size
        self.bit_width = bit_width
        self.seq_len = None

        self.value_quant = None
        self.value_norm = None
        self.value_buffered = None
        self.quantized_len = 0

    def build_sketch(self, value_states):
        b, h, _, dim = value_states.shape
        self.seq_len = value_states.shape[-2]
        self._needs_norm = self.sketch_value.needs_norm
        residual_size = self.seq_len % self.buffer_size

        if residual_size > 0:
            self.value_buffered = value_states[:, :, self.seq_len - residual_size:, :]
        if residual_size == self.seq_len:
            self.quantized_len = 0
            return None

        self.quantized_len = self.seq_len - residual_size
        value_states = value_states[:, :, :self.quantized_len, :]

        if self._needs_norm:
            self.value_norm = torch.norm(value_states, dim=-1).clamp_min(1e-8)
            self.value_quant = self.sketch_value.quantize(value_states / self.value_norm.unsqueeze(-1))
        else:
            self.value_norm = None
            self.value_quant = self.sketch_value.quantize(value_states)

    def _flush_buffer(self):
        b, h, n, dim = self.value_buffered.shape

        if self._needs_norm:
            val_norm = torch.norm(self.value_buffered, dim=-1).clamp_min(1e-8)
            val_quant = self.sketch_value.quantize(self.value_buffered / val_norm.unsqueeze(-1))
        else:
            val_norm = None
            val_quant = self.sketch_value.quantize(self.value_buffered)

        if self.value_quant is not None:
            self.value_quant = torch.cat([self.value_quant, val_quant], dim=2)
            if self._needs_norm:
                self.value_norm = torch.cat([self.value_norm, val_norm], dim=2)
        else:
            self.value_quant = val_quant
            if self._needs_norm:
                self.value_norm = val_norm

        self.quantized_len += n
        self.value_buffered = None

    def update_sketch(self, value_states):
        assert value_states.shape[-2] == 1
        self.seq_len += 1

        if self.value_buffered is not None:
            self.value_buffered = torch.cat([self.value_buffered, value_states], dim=-2)
        else:
            self.value_buffered = value_states

        if self.value_buffered.shape[-2] >= self.buffer_size:
            self._flush_buffer()

    def attention_score(self, att):
        b, h, _, dim = att.shape
        residual = None
        res_len = self.quantized_len
        if self.value_buffered is not None:
            h_k = self.value_buffered.shape[1]
            value_buffered_repeat = repeat_kv_quant(self.value_buffered, n_rep=h // h_k)
            residual = torch.matmul(att[:, :, :, res_len:], value_buffered_repeat)

        if res_len == 0:
            return residual

        h_k = self.value_quant.shape[1]
        values_repeat = repeat_kv_quant(self.value_quant, n_rep=h // h_k)
        if self._needs_norm:
            norms_repeat = repeat_kv_quant(self.value_norm.unsqueeze(-1), n_rep=h // h_k)
            values_repeat = values_repeat * norms_repeat
        scores = torch.matmul(att[:, :, :, :res_len], values_repeat)

        if residual is not None:
            return scores + residual
        return scores
