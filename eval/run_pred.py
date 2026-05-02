"""
LongBench-E prediction for KV cache quantization (RaBitQ / TurboQuant).

This script monkey-patches the official LongBench pred.py, replacing only
load_model_and_tokenizer with our version that supports FP16/turbo/rabitq.
All prediction logic (get_pred, build_chat, post_process) is unchanged.

Changes from official pred.py:
  1. load_model_and_tokenizer: replaced to support FP16/turbo/rabitq backends
  2. parse_args / main: accepts --backend/--bits/--model, single-GPU,
     optional --data-dir for local data

Usage:
    # Predict
    CUDA_VISIBLE_DEVICES=0 python eval/run_pred.py --backend rabitq --bits 2.5 --e --data-dir /path/to/data

    # Evaluate (official LongBench eval, unchanged)
    cd LongBench/LongBench && python eval.py --model <model_name> --e
"""

import json
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
LONGBENCH_DIR = REPO_DIR / "LongBench" / "LongBench"
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(LONGBENCH_DIR))

# ---------------------------------------------------------------------------
# Monkey-patch load_model_and_tokenizer before importing pred
# ---------------------------------------------------------------------------

_loaded = {"model": None, "tokenizer": None}


def load_model_and_tokenizer(path, model_name, device):
    """Drop-in replacement: ignores path/model_name/device, returns pre-loaded model."""
    return _loaded["model"], _loaded["tokenizer"]


import torch.distributed as dist
dist.destroy_process_group = lambda: None  # no-op: we don't use distributed

import pred
pred.load_model_and_tokenizer = load_model_and_tokenizer

from pred import get_pred, seed_everything

# ---------------------------------------------------------------------------
# Our model loading (the only real custom code)
# ---------------------------------------------------------------------------

def load_model(model_name, backend, bits):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if backend == "FP16":
        from transformers import AutoModelForCausalLM
        print(f"Loading {model_name} with FP16 (no quantization)...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto")
        model.eval()
        print(f"Model loaded.", flush=True)
        return model, tokenizer

    from transformers import AutoConfig

    bits_config = {"2.5": (3, 2), "3.5": (4, 3)}
    outlier_bits, residual_bits = bits_config[bits]

    if backend == "turbo":
        from kvcache_quant.turbo_sketch import TurboSketch as SketchClass
    elif backend == "rabitq":
        from kvcache_quant.rabitq_sketch import RaBitQSketch as SketchClass
    else:
        raise ValueError(f"Unknown backend: {backend}")

    config = AutoConfig.from_pretrained(model_name)
    config.attention_dropout = 0.0
    config.use_cache = True

    device = torch.device('cuda')
    generator = torch.Generator(device=device)
    generator.manual_seed(42)

    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    outlier_dim = 32
    residual_dim = head_dim - outlier_dim

    config.sketch_outlier = SketchClass(dimension=outlier_dim, bit_width=outlier_bits, rng=generator)
    config.sketch_residual = SketchClass(dimension=residual_dim, bit_width=residual_bits, rng=generator)
    config.sketch_value = SketchClass(dimension=head_dim, bit_width=2, rng=generator)

    config.outlier_count_general = outlier_dim
    config.key_quantization_bits = outlier_bits * head_dim
    config.value_quantization_bits = 2
    config.group_size = 32
    config.buffer_size = 128

    is_mistral = config.model_type == "mistral"
    if is_mistral:
        from kvcache_quant.modeling_mistral_kv_quant import MistralForCausalLM_KVQuant as ModelClass
    else:
        from kvcache_quant.modeling_llama_kv_quant import LlamaForCausalLM_KVQuant as ModelClass

    print(f"Loading {model_name} ({config.model_type}) with {backend} {bits}-bit...", flush=True)
    model = ModelClass.from_pretrained(
        pretrained_model_name_or_path=model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded.", flush=True)

    return model, tokenizer


# ---------------------------------------------------------------------------
# Main (replaces official main: single-GPU, --backend/--bits, --data-dir)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', type=str, required=True, choices=["FP16", "turbo", "rabitq"])
    parser.add_argument('--bits', type=str, default="16", choices=["2.5", "3.5", "16"])
    parser.add_argument('--model', type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help="Specific datasets to run (default: all)")
    parser.add_argument('--data-dir', type=str, default=None)
    args = parser.parse_args()

    seed_everything(42)

    dataset2prompt = json.load(open(LONGBENCH_DIR / "config" / "dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open(LONGBENCH_DIR / "config" / "dataset2maxlen.json", "r"))

    model_name = f"{args.model.split('/')[-1]}_{args.backend}_{args.bits}bit"
    max_length = 100_000

    # Load model once, store in dict for the patched load_model_and_tokenizer
    _loaded["model"], _loaded["tokenizer"] = load_model(args.model, args.backend, args.bits)

    if args.datasets:
        datasets = args.datasets
    elif args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news",
                     "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique",
                     "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht",
                     "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

    out_dir = f"pred_e/{model_name}" if args.e else f"pred/{model_name}"
    os.makedirs(out_dir, exist_ok=True)

    for dataset in datasets:
        out_path = f"{out_dir}/{dataset}.jsonl"
        if os.path.exists(out_path):
            print(f"[SKIP] {dataset}: {out_path} already exists")
            continue

        print(f"\n{'='*60}\nDataset: {dataset}\n{'='*60}")

        if args.data_dir is not None:
            suffix = f"{dataset}_e" if args.e else dataset
            data_file = Path(args.data_dir).expanduser() / f"{suffix}.jsonl"
            if not data_file.exists():
                raise FileNotFoundError(f"Local data file not found: {data_file}")
            data = load_dataset('json', data_files=str(data_file), split='train')
        else:
            data_name = f"{dataset}_e" if args.e else dataset
            data = load_dataset('THUDM/LongBench', data_name, split='test', trust_remote_code=True)

        data_all = [sample for sample in data]
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]

        # Single-GPU: call get_pred directly with rank=0, world_size=1.
        # model2path needs a valid key for model_name to avoid KeyError inside
        # get_pred (model2path[model_name]), but the value is ignored because
        # load_model_and_tokenizer has been patched to return our pre-loaded model.
        get_pred(0, 1, data_all, max_length, max_gen, prompt_format,
                 dataset, torch.device('cuda'), model_name, {model_name: ""}, out_path)

    print(f"\nDone. Evaluate with:")
    print(f"  cd LongBench/LongBench && python eval.py --model {model_name} --e")
