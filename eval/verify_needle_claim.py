"""Verify the NIAH claims in doc.tex for turbo 2.5-bit results."""

import argparse
import json
import glob
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to results directory")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--length_boundary", type=int, default=32571)
    args = parser.parse_args()

    results_dir = os.path.join(args.input, "results")
    if not os.path.isdir(results_dir):
        results_dir = args.input

    entries = []
    for f in glob.glob(os.path.join(results_dir, "*.json")):
        with open(f) as fh:
            entries.append(json.load(fh))

    scores = [e["score"] for e in entries]
    print(f"Total: {len(entries)}, mean: {np.mean(scores):.4f}")
    print(f"Below {args.threshold}: {sum(1 for s in scores if s < args.threshold)}/{len(entries)}")
    print()

    # By depth
    by_depth = {}
    for e in entries:
        by_depth.setdefault(e["depth_percent"], []).append(e["score"])

    print("=== By depth ===")
    for dp in sorted(by_depth):
        s = by_depth[dp]
        fully_correct = all(sc >= args.threshold for sc in s)
        print(f"  {dp:>5.0f}%: mean={np.mean(s):.3f}  <{args.threshold}={sum(1 for sc in s if sc < args.threshold):>2d}  {'✓ all correct' if fully_correct else ''}")

    print()

    # By length boundary
    short = [e["score"] for e in entries if e["context_length"] <= args.length_boundary]
    long = [e["score"] for e in entries if e["context_length"] > args.length_boundary]
    print(f"=== By length (boundary={args.length_boundary}) ===")
    print(f"  <={args.length_boundary}: mean={np.mean(short):.4f}  n={len(short)}")
    print(f"  > {args.length_boundary}: mean={np.mean(long):.4f}  n={len(long)}")


if __name__ == "__main__":
    main()
