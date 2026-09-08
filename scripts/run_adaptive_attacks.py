#!/usr/bin/env python3
"""Phases 4+5 — Known-T and semi-adaptive attacks.

Phase 4 (known-T): attacker has exact T, inverts scaling exactly.
  Expected: AUC recovers to approximately the pre-scaling baseline.
  If it does not, suspect the implementation.

Phase 5 (semi-adaptive): attacker knows calibration was applied,
  estimates T from 200 shadow queries (class-labeled test samples),
  then attacks the full pool.

Usage:
    python scripts/run_adaptive_attacks.py --config configs/baseline.yaml
"""

import argparse
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.attacks.adaptive_attack import run_adaptive_attacks



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

            scaled_path = f"{out_dir}/sample_outputs_temp_scaled.json"
            cal_path    = f"{out_dir}/calibration_temp_scaled.json"

            if not os.path.exists(scaled_path) or not os.path.exists(cal_path):
                print(f"  SKIP {run_id} — run Phases 2+3 first")
                continue

            with open(scaled_path) as f:
                records = json.load(f)
            with open(cal_path) as f:
                cal = json.load(f)

            T = cal["temperature"]

            # Known-T adaptive attack
            adaptive_results = run_adaptive_attacks(records, T)

            with open(f"{out_dir}/attack_results_adaptive.json", "w") as f:
                json.dump(adaptive_results, f, indent=2)

            for r in adaptive_results:
                all_rows.append({"regime": regime, "seed": seed, **r})

            print(f"  {run_id} | T={T:.3f} | "
                  f"adaptive_loss_auc={adaptive_results[0]['auc']:.4f}")

    print("\n" + "="*72)
    print(f"{'Regime':<10} {'Attack':<28} {'AUC':>10} {'TPR@1%':>10}")
    print("="*72)

    for regime in config["regimes"]:
        for attack in ["loss_adaptive", "entropy_adaptive"]:
            rows  = [r for r in all_rows
                     if r["regime"] == regime and r["attack"] == attack]
            if not rows:
                continue
            aucs  = [r["auc"]            for r in rows]
            tpr2s = [r["tpr_at_001_fpr"] for r in rows]
            print(f"{regime:<10} {attack:<28} "
                  f"{mean_std(aucs):>10}  "
                  f"{mean_std(tpr2s):>10}")

    import csv
    os.makedirs("results", exist_ok=True)
    with open("results/adaptive_attacks.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    print("\nSaved -> results/adaptive_attacks.csv")


if __name__ == "__main__":
    main()
