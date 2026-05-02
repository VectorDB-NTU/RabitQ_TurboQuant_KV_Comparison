"""Find all NIAH results with score below a threshold."""

import argparse
import json
import glob
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to results directory")
    parser.add_argument("--threshold", type=float, default=0.8, help="Score threshold (default: 0.8)")
    args = parser.parse_args()

    results_dir = os.path.join(args.input, "results")
    if not os.path.isdir(results_dir):
        results_dir = args.input

    entries = []
    for f in glob.glob(os.path.join(results_dir, "*.json")):
        with open(f) as fh:
            data = json.load(fh)
            entries.append(data)

    entries.sort(key=lambda x: (x["context_length"], x["depth_percent"]))

    low = [e for e in entries if e["score"] < args.threshold]

    print(f"Total: {len(entries)}, below {args.threshold}: {len(low)}")
    print(f"{'ctx_len':>8s}  {'depth':>6s}  {'score':>5s}  response")
    print("-" * 100)
    for e in low:
        resp = e.get("model_response", "").replace("\n", " ")[:80]
        print(f"{e['context_length']:>8d}  {e['depth_percent']:>5.0f}%  {e['score']:>5.1f}  {resp}")


if __name__ == "__main__":
    main()
