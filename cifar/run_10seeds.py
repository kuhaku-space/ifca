import json
import os
import pickle
import subprocess
import sys
import time

import numpy as np

NUM_SEEDS = 10
OUTPUT_BASE = "outputs"


def run_seed(seed):
    project_dir = os.path.join(OUTPUT_BASE, f"data_seed_{seed}")
    result_path = os.path.join(project_dir, "results.pickle")

    if os.path.exists(result_path):
        print(f"[seed {seed}] Already exists, skipping.")
        return result_path

    cfg = {"data_seed": seed, "project_dir": project_dir}
    cmd = [
        "python", "-u", "train_cluster_cifar_tf.py",
        "--config-override", json.dumps(cfg),
    ]

    print(f"[seed {seed}] Starting: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, env={**os.environ, "OMP_NUM_THREADS": "4"})
    duration = time.time() - t0

    if result.returncode != 0:
        print(f"[seed {seed}] FAILED (returncode={result.returncode})")
        return None

    print(f"[seed {seed}] Done in {duration:.1f}s")
    return result_path


def load_last_acc(result_path):
    with open(result_path, "rb") as f:
        results = pickle.load(f)
    return results[-1]["test"]["acc"]


def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    result_paths = []
    for seed in range(NUM_SEEDS):
        path = run_seed(seed)
        result_paths.append((seed, path))

    print("\n===== Results =====")
    accs = []
    for seed, path in result_paths:
        if path is None or not os.path.exists(path):
            print(f"seed {seed:2d}: FAILED")
            continue
        acc = load_last_acc(path)
        accs.append(acc)
        print(f"seed {seed:2d}: test_acc = {acc:.5f}")

    if accs:
        print(f"\nMean: {np.mean(accs):.5f}  Std: {np.std(accs):.5f}  (n={len(accs)})")

    summary = {"accs": accs, "mean": float(np.mean(accs)), "std": float(np.std(accs))}
    summary_path = os.path.join(OUTPUT_BASE, "summary_10seeds.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    t_start = time.time()
    main()
    duration = time.time() - t_start
    print(f"\n--- run_10seeds.py Ended in {duration/3600:.2f} hour ({duration:.1f}s) ---")
