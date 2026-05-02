"""
Generate LongBench table and Needle-in-a-Haystack heatmaps.

Usage:
    python eval/plot_results.py
"""

import json
import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 12

# ============================================================================
# LongBench Table
# ============================================================================

CATEGORIES = {
    "SingleQA": ["qasper", "multifieldqa_en"],
    "MultiQA": ["hotpotqa", "2wikimqa"],
    "Summ": ["gov_report", "multi_news"],
    "Few shot": ["trec", "triviaqa", "samsum"],
    "Synthetic": ["passage_count", "passage_retrieval_en"],
    "Code": ["lcc", "repobench-p"],
}

def parse_longbench_file(filepath):
    results = {}
    with open(filepath) as f:
        for line in f:
            m = re.match(r'.+dataset (\S+), avg score: ([\d.]+)', line)
            if m:
                results[m.group(1)] = float(m.group(2))
    return results

def compute_category_scores(results):
    cat_scores = {}
    for cat, datasets in CATEGORIES.items():
        scores = [results[d] * 100 for d in datasets if d in results]
        cat_scores[cat] = round(np.mean(scores), 2) if scores else None
    return cat_scores

def load_our_results(base_dir="result/longbench"):
    """Load our LongBench-E results from result/longbench/ subdirectories."""
    all_methods = {}
    if not os.path.isdir(base_dir):
        return all_methods
    for subdir in sorted(os.listdir(base_dir)):
        subdir_path = os.path.join(base_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for f in sorted(glob.glob(os.path.join(subdir_path, "*.txt"))):
            basename = os.path.basename(f)
            if "Llama" in basename:
                model = "Llama-3.1-8B"
            elif "Ministral" in basename:
                model = "Ministral-8B"
            else:
                continue

            if "FP16" in basename:
                method = "Full Cache"
                kv_size = 16
            elif "rabitq" in basename:
                bits = "2.5" if "2.5bit" in basename else "3.5"
                method = f"RaBitQ {bits}-bit"
                kv_size = float(bits)
            elif "turbo" in basename:
                bits = "2.5" if "2.5bit" in basename else "3.5"
                method = f"TurboQuant {bits}-bit"
                kv_size = float(bits)
            else:
                continue

            key = f"{model} / {method}"
            if key in all_methods:
                all_methods[key]["results"].update(parse_longbench_file(f))
            else:
                all_methods[key] = {
                    "model": model,
                    "method": method,
                    "kv_size": kv_size,
                    "results": parse_longbench_file(f),
                }
    return all_methods

def generate_longbench_table():
    all_methods = load_our_results()
    cat_names = list(CATEGORIES.keys())

    # ---- Text output ----
    header = f"{'Method':<25s} {'KV':>4s}"
    for cat in cat_names:
        header += f" {cat:>10s}"
    header += f" {'Avg':>7s}"

    model_order = ["Llama-3.1-8B", "Ministral-8B"]
    method_order = ["Full Cache", "RaBitQ 2.5-bit", "RaBitQ 3.5-bit",
                    "TurboQuant 2.5-bit", "TurboQuant 3.5-bit"]

    for model in model_order:
        print(f"\n  {model}")
        print(header)
        print("-" * len(header))
        for method in method_order:
            key = f"{model} / {method}"
            if key not in all_methods:
                continue
            info = all_methods[key]
            cat_scores = compute_category_scores(info["results"])
            line = f"  {info['method']:<23s} {info['kv_size']:>4g}"
            vals = []
            for cat in cat_names:
                v = cat_scores.get(cat)
                if v is not None:
                    line += f" {v:>10.2f}"
                    vals.append(v)
                else:
                    line += f" {'N/A':>10s}"
            # Dataset-level average (over all individual dataset scores)
            all_ds_scores = [v * 100 for v in info["results"].values()]
            avg = np.mean(all_ds_scores) if all_ds_scores else 0
            line += f" {avg:>7.2f}"
            print(line)

    # ---- Generate figure ----
    col_labels = ["Method", "KV"] + cat_names + ["Avg"]
    table_data = []
    row_colors = []

    # Collect per-model groups to find best non-baseline scores
    # bold_cells: set of (row_index_in_table_data, col_index) to bold
    bold_cells = set()

    for model in model_order:
        sep = [model] + [""] * (len(col_labels) - 1)
        table_data.append(sep)
        row_colors.append("separator")

        # Collect non-baseline rows for this model to find per-column best
        group_start = len(table_data)
        group_rows = []  # list of (row_idx_in_table_data, row_data)
        for method in method_order:
            key = f"{model} / {method}"
            if key not in all_methods:
                continue
            info = all_methods[key]
            cat_scores = compute_category_scores(info["results"])
            r = [info["method"], f"{info['kv_size']:g}"]
            vals = []
            for cat in cat_names:
                v = cat_scores.get(cat)
                if v is not None:
                    r.append(f"{v:.2f}")
                    vals.append(v)
                else:
                    r.append("N/A")
            # Dataset-level average
            all_ds_scores = [v * 100 for v in info["results"].values()]
            r.append(f"{np.mean(all_ds_scores):.2f}" if all_ds_scores else "N/A")
            row_idx = len(table_data)
            table_data.append(r)
            row_colors.append("data")
            if method != "Full Cache":
                group_rows.append((row_idx, r))

        # Find best value per column among non-baseline rows (cols 2..end)
        for col_idx in range(2, len(col_labels)):
            best_val = -1
            best_row = -1
            for row_idx, r in group_rows:
                try:
                    v = float(r[col_idx])
                    if v > best_val:
                        best_val = v
                        best_row = row_idx
                except (ValueError, IndexError):
                    pass
            if best_row >= 0:
                bold_cells.add((best_row, col_idx))

    fig, ax = plt.subplots(figsize=(18, 0.45 * len(table_data) + 1.5))
    ax.axis('off')

    table = ax.table(cellText=table_data, colLabels=col_labels, loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    # Widen Source and Method columns
    table.auto_set_column_width(list(range(len(col_labels))))

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Style rows
    for i, color_type in enumerate(row_colors):
        if color_type == "separator":
            for j in range(len(col_labels)):
                table[i + 1, j].set_facecolor('#D9E2F3')
                table[i + 1, j].set_text_props(fontweight='bold')
        elif color_type == "data":
            for j in range(len(col_labels)):
                table[i + 1, j].set_facecolor('#E2EFDA')

    # Bold best non-baseline scores in our results
    for (row_idx, col_idx) in bold_cells:
        table[row_idx + 1, col_idx].set_text_props(fontweight='bold')

    plt.savefig('result_longbench_table.pdf', bbox_inches='tight', dpi=150)
    plt.savefig('result_longbench_table.png', bbox_inches='tight', dpi=150)
    print("\nSaved: result_longbench_table.pdf / .png")

if __name__ == "__main__":
    print("=" * 80)
    print("LongBench Results")
    print("=" * 80)
    generate_longbench_table()
