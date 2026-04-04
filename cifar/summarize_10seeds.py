"""Summarize results from run_10seeds.py.

Loads outputs/data_seed_{seed}/results.pickle for seeds 0..NUM_SEEDS-1
and prints per-seed / per-round accuracy statistics.
"""

import argparse
import json
import os
import pickle

import numpy as np

NUM_SEEDS = 10
OUTPUT_BASE = "outputs"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-seeds", type=int, default=NUM_SEEDS,
                        help=f"使用するシード数 (default: {NUM_SEEDS})")
    return parser.parse_args()


def load_results(seed):
    path = os.path.join(OUTPUT_BASE, f"data_seed_{seed}", "results.pickle")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    args = parse_args()
    num_seeds = args.num_seeds

    all_accs = {}  # seed -> list of per-round acc

    for seed in range(num_seeds):
        results = load_results(seed)
        if results is None:
            print(f"seed {seed:2d}: results.pickle not found, skipping")
            continue
        try:
            accs = [r["test"]["acc"] for r in results]
            all_accs[seed] = accs
        except (KeyError, TypeError) as e:
            print(f"seed {seed:2d}: failed to parse results ({e}), skipping")

    if not all_accs:
        print("No results found.")
        return

    # --- per-seed summary (last round) ---
    print("===== Per-seed final accuracy =====")
    last_accs = []
    for seed in sorted(all_accs):
        acc = all_accs[seed][-1]
        last_accs.append(acc)
        print(f"  seed {seed:2d}: {acc:.5f}  (rounds={len(all_accs[seed])})")

    print()
    print(f"  Mean : {np.mean(last_accs):.5f}")
    print(f"  Std  : {np.std(last_accs):.5f}")
    print(f"  Min  : {np.min(last_accs):.5f}  (seed {sorted(all_accs)[int(np.argmin(last_accs))]})")
    print(f"  Max  : {np.max(last_accs):.5f}  (seed {sorted(all_accs)[int(np.argmax(last_accs))]})")
    print(f"  n    : {len(last_accs)}")

    # --- per-round mean ± std across seeds ---
    num_rounds = min(len(v) for v in all_accs.values())
    if num_rounds > 1:
        print()
        print("===== Per-round accuracy (mean ± std across seeds) =====")
        print(f"  {'round':>6}  {'mean':>8}  {'std':>8}  {'min':>8}  {'max':>8}")
        for r in range(num_rounds):
            if r % 10 != 9:
                continue
            round_accs = [all_accs[s][r] for s in sorted(all_accs)]
            print(
                f"  {r+1:6d}  {np.mean(round_accs):8.5f}  {np.std(round_accs):8.5f}"
                f"  {np.min(round_accs):8.5f}  {np.max(round_accs):8.5f}"
            )

    # --- save summary JSON ---
    summary = {
        "seeds": {str(s): all_accs[s] for s in sorted(all_accs)},
        "final": {
            "accs": last_accs,
            "mean": float(np.mean(last_accs)),
            "std": float(np.std(last_accs)),
            "min": float(np.min(last_accs)),
            "max": float(np.max(last_accs)),
            "n": len(last_accs),
        },
    }
    out_path = os.path.join(OUTPUT_BASE, "summary_10seeds.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    main()
