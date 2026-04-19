# KV Cache Quantization

This repository explores KV-cache quantization methods for efficient long-context LLM inference. Multiple quantization approaches (TurboQuant, RaBitQ) are implemented and compared under a unified evaluation framework.

## TurboQuant

The TurboQuant implementation is based on the supplementary material of the paper submitted to OpenReview:  
**[Submission tO3ASKZlok](https://openreview.net/attachment?id=tO3ASKZlok&name=supplementary_material)**

The original code can be found in the [first commit](https://github.com/VectorDB-NTU/RabitQ_TurboQuant_KV_Comparison/commit/99dc551470923cfaaaba284cec432071cee93b19) of this repository.

TurboQuant uses the Quantized Johnson-Lindenstrauss (QJL) transform to compress KV caches. It replaces standard HuggingFace Transformers attention layers with quantization-aware variants that:

- **Rotate** key/value vectors into a random orthonormal basis (via QR decomposition of Gaussian matrices)
- **Snap** each coordinate to the nearest centroid at a configurable bitwidth (1–5 bits)
- **Separate outlier channels** — the top-32 highest-norm key dimensions are quantized with a dedicated sketch, preserving accuracy on the most informative channels

During prefill, full-precision Flash Attention is used. The KV cache is then quantized and dequantized, and subsequent attention computation is performed against the dequantized data.

### Problems in LLM of TurboQuant

The original TurboQuant LLM code has several issues that affect the core quantization logic:

1. **Missing value quantizer**: The value cache quantizer (`TurboSketch` for values) is never constructed. Only key-side sketches (`qjl_outlier`, `qjl_residual`) are created — the value side has no corresponding sketch instance, so value quantization cannot run at all during prefill.

2. **Decode-phase sketch never updates**: Both `TurboKeyQuantizer.update_sketch()` and `TurboValueQuantizer.update_sketch()` return unconditionally before reaching the quantization logic. During decode, new tokens are appended to the unquantized buffer but are never flushed into the quantized sketch. The buffer grows indefinitely, defeating the purpose of quantization.

3. **Value reconstruction uses wrong operation**: `TurboValueQuantizer.attention_score()` reuses `TurboSketch.calc_score()`, which computes `query @ quantized_keys^T` (an inner-product score). For value reconstruction the correct operation is `attention_weights @ quantized_values` — the two have different semantics and dimensions.

4. **Outlier separation not applied during decode updates**: The key update path during decode does not split new keys into outlier/residual channels before quantizing, unlike the prefill path which does. This means the outlier-aware quantization strategy is only applied once at prefill and lost for all subsequent tokens.

## Evaluation

### LongBench

Evaluates quantized models on 20+ long-context tasks (QA, summarization, classification, retrieval, code completion) from the [LongBench](https://github.com/THUDM/LongBench) benchmark.

### Needle-in-a-Haystack

Tests the model's ability to retrieve a specific piece of information ("needle") embedded within a long context ("haystack") under various quantization settings, measuring whether KV-cache compression degrades retrieval accuracy at different context lengths.

