#!/usr/bin/env python3
"""Phase 3 — Naive attacks on temperature-scaled outputs.

The attacker does not know calibration was applied and runs the
same loss/confidence/entropy attacks as Project 1.

Usage:
    python scripts/run_naive_attacks.py --config configs/baseline.yaml
"""

import argparse
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.attacks.base_attacks import run_all_attacks


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


def mean_std(vals):
    return f"{np.mean(vals):.3f}+/-{np.std(vals, ddof=1):.3f}"


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    all_rows = []

    for regime in config["regimes"]:
        for seed in config["seeds"]:
            run_id  = f"{config['dataset']}_{config['architecture']}" \
                      f"_{regime}_seed{seed}"
            out_dir = f"experiments/{run_id}"

            path = f"{out_dir}/sample_outputs_temp_scaled.json"
            if not os.path.exists(path):
                print(f"  SKIP {run_id} — run Phase 2 first")
                continue

            with open(path) as f:
                records = json.load(f)

            results = run_all_attacks(records, suffix="_naive")

            with open(f"{out_dir}/attack_results_naive_scaled.json", "w") as f:
                json.dump(results, f, indent=2)

            for r in results:
                all_rows.append({"regime": regime, "seed": seed, **r})

            print(f"  {run_id}: loss_auc={results[0]['auc']:.4f}  "
                  f"entr_tpr1%={results[2]['tpr_at_001_fpr']:.4f}")

    print("\n" + "="*68)
    print(f"{'Regime':<10} {'Attack':<22} {'AUC':>10} {'TPR@1%':>10}")
    print("="*68)

    for regime in config["regimes"]:
        for attack in ["loss_naive", "confidence_naive", "entropy_naive"]:
            rows  = [r for r in all_rows
                     if r["regime"] == regime and r["attack"] == attack]
            aucs  = [r["auc"]            for r in rows]
            tpr2s = [r["tpr_at_001_fpr"] for r in rows]
            print(f"{regime:<10} {attack:<22} "
                  f"{mean_std(aucs):>10}  "
                  f"{mean_std(tpr2s):>10}")

    import csv
    os.makedirs("results", exist_ok=True)
    with open("results/naive_scaled_attacks.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    print("\nSaved -> results/naive_scaled_attacks.csv")


if __name__ == "__main__":
    main()
