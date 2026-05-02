"""
Generate Needle-in-a-Haystack heatmaps matching TurboQuant paper style.

Usage:
    python eval/plot_needle.py
"""

import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


EXPECTED_ANSWER = "eat a sandwich and sit in Dolores Park on a sunny day."


def keyword_coverage(response, expected=EXPECTED_ANSWER):
    """Token-visualize score: fraction of answer keywords found in response."""
    response_words = set(response.lower().split())
    answer_words = set(expected.lower().split())
    return len(response_words & answer_words) / len(answer_words)


def load_needle_results(result_dir):
    methods = {}
    for f in glob.glob(os.path.join(result_dir, "**/*.json"), recursive=True):
        with open(f) as fp:
            d = json.load(fp)

        response = d.get("model_response")
        if response is None:
            continue

        name = os.path.basename(f)
        if "_FP16_" in name:
            method = "Full-Precision"
        elif "_rabitq_" in name:
            bits = "2.5" if "2_5bit" in name else "3.5"
            method = f"RaBitQ {bits}-bit"
        elif "_turbo_" in name:
            bits = "2.5" if "2_5bit" in name else "3.5"
            method = f"TurboQuant {bits}-bit"
        else:
            continue

        if method not in methods:
            methods[method] = []
        methods[method].append({
            "context_length": d.get("context_length"),
            "depth_percent": d.get("depth_percent"),
            "score": keyword_coverage(response),
        })

    return methods


def plot_needle_heatmaps(result_dir="result_merged/needle"):
    methods = load_needle_results(result_dir)
    if not methods:
        print("No needle results found")
        return

    method_order = ["Full-Precision", "RaBitQ 2.5-bit", "RaBitQ 3.5-bit",
                    "TurboQuant 2.5-bit", "TurboQuant 3.5-bit"]
    available = [m for m in method_order if m in methods]
    n = len(available)
    if n == 0:
        print("No recognized methods found")
        return

    # Colormap: red(0) -> yellow(0.5) -> green(1.0), matching paper style
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "needle", [(0.8, 0.2, 0.2), (0.95, 0.85, 0.4), (0.4, 0.75, 0.45)], N=256)

    # Each heatmap is 10 rows × 15 cols; use aspect ratio to make cells square
    cell_w, cell_h = 0.28, 0.42  # per-cell size in inches (h > w to make grid square)
    n_contexts = 15  # expected columns
    n_depths = 10    # expected rows
    subplot_w = cell_w * n_contexts
    subplot_h = cell_h * n_depths
    fig_w = subplot_w * n + 1.8  # extra for colorbar
    fig_h = subplot_h + 1.5      # extra for title/labels
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h))
    if n == 1:
        axes = [axes]

    im = None
    for idx, method in enumerate(available):
        data = methods[method]
        contexts = sorted(set(d["context_length"] for d in data))
        depths = sorted(set(d["depth_percent"] for d in data))

        # Build score matrix (normalized to 0-1)
        score_matrix = np.full((len(depths), len(contexts)), np.nan)
        for d in data:
            ci = contexts.index(d["context_length"])
            di = depths.index(d["depth_percent"])
            score_matrix[di, ci] = d["score"]

        avg_score = np.nanmean(score_matrix)

        ax = axes[idx]
        im = ax.pcolormesh(
            np.arange(len(contexts) + 1), np.arange(len(depths) + 1),
            score_matrix, cmap=cmap, vmin=0, vmax=1,
            edgecolors='black', linewidth=0.5)
        ax.set_aspect('equal')

        ax.set_title(f"{method}\nScore: {avg_score:.3f}", fontsize=11, fontweight='bold')

        # X axis
        context_labels = [f"{c // 1000}k" for c in contexts]
        ax.set_xticks(np.arange(len(contexts)) + 0.5)
        ax.set_xticklabels(context_labels, rotation=45, ha='right', fontsize=7)
        ax.set_xlabel("Token Limit", fontsize=9)

        # Y axis
        depth_labels = [str(int(d)) for d in depths]
        ax.set_yticks(np.arange(len(depths)) + 0.5)
        ax.set_yticklabels(depth_labels, fontsize=7)
        if idx == 0:
            ax.set_ylabel("Depth Percent", fontsize=9)
        else:
            ax.set_yticklabels([])

        ax.invert_yaxis()

    # Colorbar on the right
    cbar = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02, aspect=15)
    cbar.set_label("Score", fontsize=10)
    cbar.set_ticks([0, 0.25, 0.50, 0.75, 1.00])

    plt.savefig('result_needle_heatmaps.pdf', bbox_inches='tight', dpi=150)
    plt.savefig('result_needle_heatmaps.png', bbox_inches='tight', dpi=150)
    print("Saved: result_needle_heatmaps.pdf / .png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default="result/needle")
    args = parser.parse_args()
    plot_needle_heatmaps(args.result_dir)
