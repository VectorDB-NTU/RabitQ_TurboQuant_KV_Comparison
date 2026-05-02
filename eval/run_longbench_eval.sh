#!/bin/bash
# End-to-end LongBench-E evaluation: predict + score.
#
# Usage:
#   bash eval/run_longbench_eval.sh --backend rabitq --bits 2.5
#   bash eval/run_longbench_eval.sh --backend rabitq --bits 2.5 --gpu 0
#   bash eval/run_longbench_eval.sh --backend FP16 --data-dir /path/to/data
#   bash eval/run_longbench_eval.sh --backend turbo --bits 2.5 --model mistralai/Ministral-8B-Instruct-2410
#
# Intermediate predictions are saved under <pred-dir>/<model_name>/.
# Final scores are written to <output>/result.json.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
LONGBENCH_DIR="$REPO_DIR/LongBench/LongBench"

# --- Defaults ---
BACKEND=""
BITS="16"
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
DATA_DIR=""
PRED_DIR=""
OUTPUT=""
GPU=""
DATASETS=""

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --backend)   BACKEND="$2"; shift 2 ;;
        --bits)      BITS="$2"; shift 2 ;;
        --model)     MODEL="$2"; shift 2 ;;
        --data-dir)  DATA_DIR="$2"; shift 2 ;;
        --pred-dir)  PRED_DIR="$2"; shift 2 ;;
        --output)    OUTPUT="$2"; shift 2 ;;
        --gpu)       GPU="$2"; shift 2 ;;
        --datasets)  shift; while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do DATASETS="$DATASETS $1"; shift; done ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$BACKEND" ]]; then
    echo "Error: --backend is required (FP16, turbo, rabitq)"
    exit 1
fi

# --- Derive model_name (must match run_pred.py logic) ---
MODEL_SHORT="${MODEL##*/}"
MODEL_NAME="${MODEL_SHORT}_${BACKEND}_${BITS}bit"

# --- Defaults for pred-dir and output ---
if [[ -z "$PRED_DIR" ]]; then
    PRED_DIR="$REPO_DIR/pred_e"
fi
if [[ -z "$OUTPUT" ]]; then
    OUTPUT="$PRED_DIR/$MODEL_NAME"
fi

# --- Set GPU ---
if [[ -n "$GPU" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

echo "========================================"
echo "GPU:         ${CUDA_VISIBLE_DEVICES:-0}"
echo "Backend:     $BACKEND"
echo "Bits:        $BITS"
echo "Model:       $MODEL"
echo "Model name:  $MODEL_NAME"
echo "Pred dir:    $PRED_DIR/$MODEL_NAME"
echo "Output:      $OUTPUT/result.json"
echo "========================================"

# --- Step 1: Generate predictions ---
echo ""
echo "[Step 1] Generating predictions..."

PRED_CMD=(python "$SCRIPT_DIR/run_pred.py" --backend "$BACKEND" --bits "$BITS" --model "$MODEL" --e)
if [[ -n "$DATA_DIR" ]]; then
    PRED_CMD+=(--data-dir "$DATA_DIR")
fi
if [[ -n "$DATASETS" ]]; then
    PRED_CMD+=(--datasets $DATASETS)
fi

# run_pred.py writes to pred_e/<model_name>/ relative to cwd,
# so we run from REPO_DIR
cd "$REPO_DIR"
"${PRED_CMD[@]}"

# --- Step 2: Evaluate ---
echo ""
echo "[Step 2] Scoring predictions..."

# eval.py reads from pred_e/<model_name>/ relative to its cwd
# We need pred_e/<model_name> to be accessible, so symlink if needed
EVAL_PRED_DIR="$LONGBENCH_DIR/pred_e/$MODEL_NAME"
if [[ "$PRED_DIR/$MODEL_NAME" != "$EVAL_PRED_DIR" ]]; then
    mkdir -p "$LONGBENCH_DIR/pred_e"
    ln -sfn "$PRED_DIR/$MODEL_NAME" "$EVAL_PRED_DIR"
fi

cd "$LONGBENCH_DIR"
python eval.py --model "$MODEL_NAME" --e

# --- Step 3: Compute per-dataset overall scores + category summary ---
#
# Scoring follows the TurboQuant convention (script-llama3-turbo.py):
#   1. Post-processing: all predictions go through strip() + split('\n')[0] +
#      split('  ')[0], matching TurboQuant's output cleanup before scoring.
#   2. Per-dataset score: full-sample mean (not split by length buckets),
#      equivalent to LongBench eval.py scorer() (without --e).
#   3. Category score: arithmetic mean of dataset scores within each group.
#   4. Average: arithmetic mean of the 6 category scores.
#
# Note on code_sim_score: fuzzywuzzy.fuzz.ratio gives different results
# depending on whether python-Levenshtein is installed (C implementation)
# or not (pure-python SequenceMatcher fallback). To reproduce published
# results, python-Levenshtein must be installed:
#   pip install python-Levenshtein==0.27.3
#
# Output: a .txt file compatible with plot_results.py in result/longbench/.
echo ""
echo "[Step 3] Computing category summary..."

python3 -c "
import json, os, sys, numpy as np

# Verify python-Levenshtein is installed for reproducible code_sim_score
try:
    import Levenshtein  # noqa: F401
except ImportError:
    print('ERROR: python-Levenshtein is not installed. Code scores will be wrong.')
    print('       pip install python-Levenshtein==0.27.3')
    sys.exit(1)

sys.path.insert(0, '$LONGBENCH_DIR')
from metrics import qa_f1_score, rouge_score, classification_score, retrieval_score, count_score, code_sim_score

dataset2metric = {
    'qasper': qa_f1_score, 'multifieldqa_en': qa_f1_score,
    'hotpotqa': qa_f1_score, '2wikimqa': qa_f1_score,
    'gov_report': rouge_score, 'multi_news': rouge_score,
    'trec': classification_score, 'triviaqa': qa_f1_score, 'samsum': rouge_score,
    'passage_count': count_score, 'passage_retrieval_en': retrieval_score,
    'lcc': code_sim_score, 'repobench-p': code_sim_score,
}
CATEGORIES = {
    'SingleQA': ['qasper', 'multifieldqa_en'],
    'MultiQA': ['hotpotqa', '2wikimqa'],
    'Summ': ['gov_report', 'multi_news'],
    'Few shot': ['trec', 'triviaqa', 'samsum'],
    'Synthetic': ['passage_count', 'passage_retrieval_en'],
    'Code': ['lcc', 'repobench-p'],
}

pred_dir = '$PRED_DIR/$MODEL_NAME'
dataset_scores = {}

for f in sorted(os.listdir(pred_dir)):
    if not f.endswith('.jsonl'):
        continue
    ds = f.replace('.jsonl', '')
    if ds not in dataset2metric:
        continue
    metric_fn = dataset2metric[ds]
    total, count = 0.0, 0
    with open(os.path.join(pred_dir, f)) as fp:
        for line in fp:
            d = json.loads(line)
            # TurboQuant-style post-processing (applied to all datasets)
            pred = d['pred'].strip()
            pred = pred.lstrip('\n').split('\n')[0]
            pred = pred.split('  ')[0]
            score = max(metric_fn(pred, gt, all_classes=d['all_classes']) for gt in d['answers'])
            total += score
            count += 1
    if count > 0:
        dataset_scores[ds] = total / count

# Write txt compatible with plot_results.py
out_dir = os.path.join('$REPO_DIR', 'result', 'longbench', '$MODEL_NAME')
os.makedirs(out_dir, exist_ok=True)
txt_path = os.path.join(out_dir, '$MODEL_NAME.txt')
with open(txt_path, 'w') as fp:
    for ds, sc in sorted(dataset_scores.items()):
        fp.write(f'Eval dataset {ds}, avg score: {sc:.4f}\n')

# Print per-dataset scores
print()
for ds, sc in sorted(dataset_scores.items()):
    print(f'  {ds:<25s} {sc*100:>6.2f}')

# Print category summary
print()
cat_scores = {}
for cat, datasets in CATEGORIES.items():
    scores = [dataset_scores[d] * 100 for d in datasets if d in dataset_scores]
    if scores:
        cat_scores[cat] = np.mean(scores)

all_vals = []
for cat in CATEGORIES:
    v = cat_scores.get(cat)
    if v is not None:
        print(f'  {cat:<12s} {v:>6.2f}')
        all_vals.append(v)
if all_vals:
    print(f'  {\"Average\":<12s} {np.mean(all_vals):>6.2f}')

print(f'\nSaved: {txt_path}')
"

# --- Step 4: Copy bucket result to output ---
RESULT_FILE="$EVAL_PRED_DIR/result.json"
if [[ "$OUTPUT" != "$PRED_DIR/$MODEL_NAME" ]]; then
    mkdir -p "$OUTPUT"
    cp "$RESULT_FILE" "$OUTPUT/result.json"
fi

echo ""
echo "Bucket results (LongBench-E):"
cat "$RESULT_FILE"
echo ""
echo "Saved to: $OUTPUT/result.json"
