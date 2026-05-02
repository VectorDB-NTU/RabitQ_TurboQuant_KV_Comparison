# KV Cache Quantization

This repository explores KV-cache quantization methods for efficient long-context LLM inference. Multiple quantization approaches (RaBitQ, TurboQuant) are implemented and compared under a unified evaluation framework.

## Problems in the Original TurboQuant Code

The TurboQuant LLM code from the [first commit](https://github.com/VectorDB-NTU/RabitQ_TurboQuant_KV_Comparison/commit/99dc551470923cfaaaba284cec432071cee93b19) (i.e., the OpenReview supplementary material) has several issues that affect the core quantization logic:

1. **Missing value quantizer**: The value cache quantizer (`TurboSketch` for values) is never constructed. Only key-side sketches (`qjl_outlier`, `qjl_residual`) are created — the value side has no corresponding sketch instance, so value quantization cannot run at all during prefill.

2. **Decode-phase sketch never updates**: Both `TurboKeyQuantizer.update_sketch()` and `TurboValueQuantizer.update_sketch()` return unconditionally before reaching the quantization logic. During decode, new tokens are appended to the unquantized buffer but are never flushed into the quantized sketch. The buffer grows indefinitely, defeating the purpose of quantization.

3. **Value reconstruction uses wrong operation**: `TurboValueQuantizer.attention_score()` reuses `TurboSketch.calc_score()`, which computes `query @ quantized_keys^T` (an inner-product score). For value reconstruction the correct operation is `attention_weights @ quantized_values` — the two have different semantics and dimensions.

4. **Outlier separation not applied during decode updates**: The key update path during decode does not split new keys into outlier/residual channels before quantizing, unlike the prefill path which does. This means the outlier-aware quantization strategy is only applied once at prefill and lost for all subsequent tokens.

## Project Structure

```
├── kvcache_quant/                        # Unified quantization framework
│   ├── kv_quantizer.py                   # Algorithm-agnostic orchestration (KeyQuantizer, ValueQuantizer)
│   ├── rabitq_sketch.py                  # RaBitQSketch (Haar rotation + RaBitQ quantization)
│   ├── turbo_sketch.py                   # TurboSketch (QR rotation + Lloyd-Max centroid quantization)
│   ├── modeling_llama_kv_quant.py        # Llama model with pluggable KV cache quantization
│   └── modeling_mistral_kv_quant.py      # Mistral model with pluggable KV cache quantization
│
├── eval/                                 # Evaluation and plotting scripts
│   ├── run_pred.py                       # LongBench model prediction
│   ├── run_longbench_eval.sh             # LongBench-E end-to-end evaluation
│   ├── run_needle.py                     # Needle-in-a-Haystack evaluation
│   ├── plot_results.py                   # LongBench result plotting
│   ├── plot_needle.py                    # Needle-in-a-Haystack result plotting
│   └── niah_haystack_order.patch         # Patch to fix haystack file order for reproducibility
│
├── llm_rabitq/                           # RaBitQ GPU quantization kernel
│   ├── binding.cpp                       # pybind11 binding, exposes rabitq.quantize()
│   ├── setup.py                          # pip install -e .
│   ├── include/                          # CUDA header files
│   ├── src/                              # CUDA source files
│   └── test/                             # Unit tests
│
├── llm_turbo/                            # Patched TurboQuant LLM implementation
│   ├── llama_turbo.py                    # Llama with QJL attention
│   ├── mistral_turbo.py                  # Mistral with QJL attention
│   ├── utils_turbo_outlier.py            # TurboSketch / quantizer utilities
│   └── script-llama3-turbo.py            # Original evaluation script
│
├── scripts/                              # Environment setup scripts
│   └── setup_ubuntu_cuda121.sh           # One-shot Ubuntu 22.04 + CUDA 12.1 setup
│
└── requirements.txt                      # Python dependencies
```

## Setup

### One-shot setup script

For a bare Ubuntu 22.04 machine, you can use:

```bash
bash scripts/setup_ubuntu_cuda121.sh
```

### A100 Environment Setup (Ubuntu 22.04, bare machine)

If starting from a bare Ubuntu 22.04 machine with only NVIDIA drivers installed (e.g. cloud GPU instances), follow these steps to set up the full build environment.

#### 1. System build tools

```bash
sudo apt update
sudo apt install -y build-essential g++ python3.11 python3.11-dev python3.11-venv g++-12
```

#### 2. CUDA 12.1 Toolkit

The CUDA toolkit provides `nvcc` and development headers needed to compile CUDA extensions. This does not affect the GPU driver.

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-1
```

**Note:** If the system already has a different CUDA toolkit version (e.g. 13.x), this installs 12.1 side-by-side under `/usr/local/cuda-12.1/`. The environment variables below ensure the correct version is used.

#### 3. Environment variables

```bash
# Add to ~/.bashrc for persistence
export PATH=/usr/local/cuda-12.1/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.1
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
```

Verify:
```bash
nvcc --version    # should show 12.1
nvidia-smi        # should show driver and GPU
```

#### 4. Python environment and dependencies

```bash
python3.11 -m venv ~/rabitq_env
source ~/rabitq_env/bin/activate
pip install --upgrade pip

# PyTorch (CUDA 12.1)
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# HuggingFace ecosystem
pip install transformers==4.48.2 accelerate==1.3.0 datasets==2.21.0 wheel

# Flash Attention
pip install flash-attn==2.6.3 --no-build-isolation

# Build tools, scientific computing, evaluation
pip install ninja numpy==1.26.4
pip install tqdm rouge fuzzywuzzy python-Levenshtein jieba pytest
pip install sentencepiece safetensors huggingface_hub tokenizers
```

#### 5. Build RaBitQ CUDA extension

```bash
export TORCH_CUDA_ARCH_LIST="8.0"   # A100; see table below for other GPUs
cd llm_rabitq
pip install . --no-build-isolation
python -m pytest test/ -v
cd ..
```

`TORCH_CUDA_ARCH_LIST` limits compilation to a specific GPU architecture (faster build). If unset, all architectures are compiled.

| GPU | Architecture |
|-----|-------------|
| V100 | `7.0` |
| A100 | `8.0` |
| A10/A30 | `8.6` |
| H100 | `9.0` |

#### 6. Verify

```bash
python -c "
import torch
print(f'torch:        {torch.__version__}')
print(f'CUDA:         {torch.version.cuda}')
print(f'GPU:          {torch.cuda.get_device_name(0)}')
import transformers
print(f'transformers: {transformers.__version__}')
import flash_attn
print(f'flash_attn:   {flash_attn.__version__}')
import rabitq
print('rabitq:       OK')
"
```

Expected output on A100:
```
torch:        2.4.1+cu121
CUDA:         12.1
GPU:          NVIDIA A100-SXM4-80GB
transformers: 4.48.2
flash_attn:   2.6.3
rabitq:       OK
```

### Quick Setup (if build tools and CUDA toolkit are already installed)

#### 1. Install Python dependencies

```bash
pip install -r requirements.txt
pip install flash-attn==2.6.3 --no-build-isolation
```

#### 2. Build RaBitQ CUDA extension (required for `--backend rabitq`)

```bash
cd llm_rabitq
pip install -e . --no-build-isolation
python -m pytest test/ -v
cd ..
```

#### 3. Initialize git submodules (required for LongBench and Needle-in-a-Haystack evaluation)

```bash
git submodule update --init
```

#### 4. HuggingFace login (for gated model access)

```bash
huggingface-cli login
```

## Evaluation

### LongBench

Evaluates quantized models on 13 long-context tasks (QA, summarization, classification, retrieval, code completion) from the [LongBench-E](https://github.com/THUDM/LongBench) benchmark.

`eval/run_longbench_eval.sh` runs prediction and scoring end-to-end. Intermediate predictions are saved under `pred_e/<model_name>/`, and final scores are written to `result.json` in the same directory. The script supports resume: existing dataset predictions are skipped.

```bash
# Initialize LongBench submodule (first time only)
git submodule update --init

# 
mkdir result

# FP16 baseline
bash eval/run_longbench_eval.sh --backend FP16 --gpu 0

# RaBitQ 2.5-bit
bash eval/run_longbench_eval.sh --backend rabitq --bits 2.5 --gpu 0

# RaBitQ 3.5-bit
bash eval/run_longbench_eval.sh --backend rabitq --bits 3.5 --gpu 0

# TurboQuant 2.5-bit
bash eval/run_longbench_eval.sh --backend turbo --bits 2.5 --gpu 0

# Use Ministral model
bash eval/run_longbench_eval.sh --backend rabitq --bits 2.5 --model mistralai/Ministral-8B-Instruct-2410 --gpu 1

# Use local data files instead of downloading from HuggingFace
bash eval/run_longbench_eval.sh --backend rabitq --bits 2.5 --data-dir /path/to/longbench_e_data --gpu 0

# Custom output directory
bash eval/run_longbench_eval.sh --backend rabitq --bits 2.5 --gpu 0 --output result/longbench
```

### Needle-in-a-Haystack

Tests retrieval accuracy across context lengths (4k–104k tokens) and needle positions.

The NIAH framework is included as a git submodule. A patch must be applied to fix the haystack file ordering for reproducibility (`glob.glob()` returns different orders on different filesystems):

```bash
# Initialize submodule (first time only)
git submodule update --init

# 
mkdir result

# Apply haystack ordering patch for reproducibility
cd LLMTest_NeedleInAHaystack
git apply ../eval/niah_haystack_order.patch
cd ..
```

Default evaluation settings: `--prompt_style tsa_text` (TSA-style book prompt), `--evaluator keyword` (local keyword-coverage scoring, no GPT needed), `--max_new_tokens 30`, context range 4k–104k with 15 intervals and 10 depth intervals.

```bash
# FP16 baseline
CUDA_VISIBLE_DEVICES=0 python eval/run_needle.py --backend FP16 \
  --prompt_style tsa_text --evaluator keyword --max_new_tokens 30 \
  --context_lengths_min 4000 --context_lengths_max 104000 \
  --context_lengths_num_intervals 15 --document_depth_percent_intervals 10 \
  --results_version 1 --output_dir result/needle_FP16_keyword

# RaBitQ 2.5-bit
CUDA_VISIBLE_DEVICES=0 python eval/run_needle.py --backend rabitq --bits 2.5 \
  --prompt_style tsa_text --evaluator keyword --max_new_tokens 30 \
  --context_lengths_min 4000 --context_lengths_max 104000 \
  --context_lengths_num_intervals 15 --document_depth_percent_intervals 10 \
  --results_version 1 --output_dir result/needle_rabitq_2_5_keyword

# TurboQuant 2.5-bit
CUDA_VISIBLE_DEVICES=0 python eval/run_needle.py --backend turbo --bits 2.5 \
  --prompt_style tsa_text --evaluator keyword --max_new_tokens 30 \
  --context_lengths_min 4000 --context_lengths_max 104000 \
  --context_lengths_num_intervals 15 --document_depth_percent_intervals 10 \
  --results_version 1 --output_dir result/needle_turbo_2_5_keyword

# RaBitQ 3.5-bit
CUDA_VISIBLE_DEVICES=0 python eval/run_needle.py --backend rabitq --bits 3.5 \
  --prompt_style tsa_text --evaluator keyword --max_new_tokens 30 \
  --context_lengths_min 4000 --context_lengths_max 104000 \
  --context_lengths_num_intervals 15 --document_depth_percent_intervals 10 \
  --results_version 1 --output_dir result/needle_rabitq_3_5_keyword

# TurboQuant 3.5-bit
CUDA_VISIBLE_DEVICES=0 python eval/run_needle.py --backend turbo --bits 3.5 \
  --prompt_style tsa_text --evaluator keyword --max_new_tokens 30 \
  --context_lengths_min 4000 --context_lengths_max 104000 \
  --context_lengths_num_intervals 15 --document_depth_percent_intervals 10 \
  --results_version 1 --output_dir result/needle_turbo_3_5_keyword
```
