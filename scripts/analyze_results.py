#!/usr/bin/env python3
"""Phase 7 — Paired delta analysis across all three threat models.

Builds the final comparison table and generates all plots.

Usage:
    python scripts/analyze_results.py --config configs/baseline.yaml
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REGIMES  = ["scratch", "frozen", "partial", "full"]
LABELS   = ["Scratch", "Frozen", "Partial FT", "Full FT"]
COLORS   = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
MARKERS  = ["o", "s", "^", "D"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",  required=True)
    p.add_argument("--ls_smoothing", type=float, default=0.1)
    return p.parse_args()


def load_run_data(run_id, out_dir):
    data = {"run_id": run_id}

    # Baseline attacks (from Project 1)
    p = f"{out_dir}/attack_results_baseline.json"
    if os.path.exists(p):
        with open(p) as f:
            atk = {a["attack"]: a for a in json.load(f)}
        data["baseline_loss_auc"]  = atk.get("loss",     {}).get("auc", None)
        data["baseline_entr_tpr1"] = atk.get("entropy",  {}).get("tpr_at_001_fpr", None)
        data["baseline_ece"]       = None

    # Baseline calibration
    p = f"{out_dir}/calibration_baseline.json"
    if os.path.exists(p):
        with open(p) as f:
            cal = json.load(f)
        data["baseline_ece"]         = cal["calibration"]["ece"]
        data["baseline_entr_gap"]    = cal["entropy_analysis"].get("entropy_gap")

    # Temperature scaling calibration
    p = f"{out_dir}/calibration_temp_scaled.json"
    if os.path.exists(p):
        with open(p) as f:
            cal = json.load(f)
        data["T"]               = cal["temperature"]
        data["scaled_ece"]      = cal["calibration"]["ece"]
        data["scaled_entr_gap"] = cal["entropy_analysis"].get("entropy_gap")

    # Naive attacks on scaled outputs
    p = f"{out_dir}/attack_results_naive_scaled.json"
    if os.path.exists(p):
        with open(p) as f:
            atk = {a["attack"]: a for a in json.load(f)}
        data["naive_loss_auc"]  = atk.get("loss_naive",    {}).get("auc", None)
        data["naive_entr_tpr1"] = atk.get("entropy_naive", {}).get("tpr_at_001_fpr", None)

    # Adaptive attacks
    p = f"{out_dir}/attack_results_adaptive.json"
    if os.path.exists(p):
        with open(p) as f:
            atk = {a["attack"]: a for a in json.load(f)}
        data["adaptive_loss_auc"]  = atk.get("loss_adaptive",     {}).get("auc", None)
        data["semi_loss_auc"]      = atk.get("loss_semiadaptive",  {}).get("auc", None)

    return data


def ms(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return "N/A"
    return f"{np.mean(v):.3f}+/-{np.std(v, ddof=1):.3f}"


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs("results", exist_ok=True)
    os.makedirs("plots",   exist_ok=True)

    # ── Load all runs ──────────────────────────────────────────────────────
    rows = []
    for regime in REGIMES:
        for seed in config["seeds"]:
            run_id  = (f"{config['dataset']}_{config['architecture']}"
                       f"_{regime}_seed{seed}")
            out_dir = f"experiments/{run_id}"
            if not os.path.isdir(out_dir):
                continue
            data = load_run_data(run_id, out_dir)
            data.update({"regime": regime, "seed": seed})
            rows.append(data)

    # ── Paired delta table ─────────────────────────────────────────────────
    print("\n" + "="*80)
    print("Paired delta: temperature scaling vs baseline (mean +/- sample SD, n=3)")
    print(f"{'Regime':<12} {'T':>6} {'ΔECE':>10} {'ΔNaiveMIA':>12} "
          f"{'AdaptMIA':>10} {'SemiMIA':>10}")
    print("="*80)

    summary_rows = []
    for regime in REGIMES:
        r = [x for x in rows if x["regime"] == regime]
        Ts         = [x.get("T")                for x in r]
        delta_ece  = [x.get("scaled_ece", 0)  - x.get("baseline_ece", 0)  for x in r]
        delta_mia  = [x.get("naive_loss_auc", 0) - x.get("baseline_loss_auc", 0) for x in r]
        adapt_auc  = [x.get("adaptive_loss_auc") for x in r]
        semi_auc   = [x.get("semi_loss_auc")     for x in r]

        print(f"{regime:<12} {ms(Ts):>6}  {ms(delta_ece):>10}  "
              f"{ms(delta_mia):>12}  {ms(adapt_auc):>10}  {ms(semi_auc):>10}")

        for i, x in enumerate(r):
            summary_rows.append({
                "regime":              regime,
                "seed":                x["seed"],
                "T":                   x.get("T"),
                "baseline_ece":        x.get("baseline_ece"),
                "scaled_ece":          x.get("scaled_ece"),
                "baseline_loss_auc":   x.get("baseline_loss_auc"),
                "naive_loss_auc":      x.get("naive_loss_auc"),
                "adaptive_loss_auc":   x.get("adaptive_loss_auc"),
                "semi_loss_auc":       x.get("semi_loss_auc"),
                "baseline_entr_tpr1":  x.get("baseline_entr_tpr1"),
                "naive_entr_tpr1":     x.get("naive_entr_tpr1"),
                "baseline_entr_gap":   x.get("baseline_entr_gap"),
                "scaled_entr_gap":     x.get("scaled_entr_gap"),
            })

    with open("results/defense_comparison.csv", "w", newline="") as f:
        if summary_rows:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
    print("\nSaved -> results/defense_comparison.csv")

    # ── Plot 1: Baseline vs naive vs adaptive MIA AUC ─────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(REGIMES))
    width = 0.25

    for i, (key, label) in enumerate([
        ("baseline_loss_auc", "Baseline (no defense)"),
        ("naive_loss_auc",    "After temp scaling (naive attacker)"),
        ("adaptive_loss_auc", "After temp scaling (adaptive attacker)"),
    ]):
        means = [np.mean([r.get(key) for r in rows if r["regime"] == reg
                          and r.get(key) is not None])
                 for reg in REGIMES]
        stds  = [np.std([r.get(key) for r in rows if r["regime"] == reg
                         and r.get(key) is not None], ddof=1)
                 for reg in REGIMES]
        ax.bar(x + (i - 1) * width, means, width, yerr=stds,
               label=label, alpha=0.85, capsize=4)

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, label="Random (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("Loss MIA AUROC")
    ax.set_title("Temperature Scaling: Naive vs Adaptive Attacker\n"
                 "DermaMNIST x ResNet-18, mean +/- SD, 3 seeds")
    ax.legend(fontsize=8)
    ax.set_ylim(0.45, 0.78)
    plt.tight_layout()
    plt.savefig("plots/plot1_defense_comparison.png", dpi=150)
    plt.close()
    print("Saved plots/plot1_defense_comparison.png")

    # ── Plot 2: Temperature T by regime ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, regime in enumerate(REGIMES):
        Ts = [r.get("T") for r in rows if r["regime"] == regime
              and r.get("T") is not None]
        if Ts:
            ax.scatter([i] * len(Ts), Ts, color=COLORS[i],
                       marker=MARKERS[i], s=80, zorder=3)
            ax.plot([i - 0.15, i + 0.15], [np.mean(Ts)] * 2,
                    color=COLORS[i], linewidth=2)

    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1, label="T=1 (no scaling)")
    ax.set_xticks(range(len(REGIMES)))
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("Optimal temperature T")
    ax.set_title("Calibration Temperature by Training Regime\n"
                 "T fitted on validation set; T>1 indicates overconfidence")
    ax.legend()
    plt.tight_layout()
    plt.savefig("plots/plot2_temperature_by_regime.png", dpi=150)
    plt.close()
    print("Saved plots/plot2_temperature_by_regime.png")

    # ── Plot 3: Entropy gap before and after scaling ───────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(REGIMES))
    width = 0.35

    for i, (key, label) in enumerate([
        ("baseline_entr_gap", "Before temp scaling"),
        ("scaled_entr_gap",   "After temp scaling"),
    ]):
        means = [np.mean([r.get(key) for r in rows if r["regime"] == reg
                          and r.get(key) is not None])
                 for reg in REGIMES]
        stds  = [np.std([r.get(key) for r in rows if r["regime"] == reg
                         and r.get(key) is not None], ddof=1)
                 for reg in REGIMES]
        ax.bar(x + (i - 0.5) * width, means, width, yerr=stds,
               label=label, alpha=0.85, capsize=4)

    ax.axhline(0.0, color="grey", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("Entropy gap (non-member - member)")
    ax.set_title("Member vs Non-member Entropy Gap\nBefore and After Temperature Scaling")
    ax.legend()
    plt.tight_layout()
    plt.savefig("plots/plot3_entropy_gap.png", dpi=150)
    plt.close()
    print("Saved plots/plot3_entropy_gap.png")

    print("\nPhase 7 complete.")


if __name__ == "__main__":
    main()
