#!/usr/bin/env python3
"""Phase 1 — Collect calibration metrics from Project 1 checkpoints.

Loads each trained model, evaluates on the test split, and computes
ECE, NLL, Brier score, and average entropy. Also reports member vs
non-member entropy from the existing sample_outputs.json files.

Data separation enforced:
    val set       -> temperature calibration (Phase 2 only)
    test set      -> calibration quality metrics (this script)
    attack pool   -> MIA evaluation only

Usage:
    python scripts/collect_baseline.py \
        --config configs/baseline.yaml \
        --p1_dir /path/to/medical-membership-privacy
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.seed import set_seed
from src.data.transforms import get_transforms
from src.data.medmnist_loader import get_dataloaders
from src.models.model_factory import build_model
from src.training.checkpoint import load_checkpoint
from src.calibration.metrics import (
    compute_all_metrics,
    member_nonmember_entropy,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",  required=True)
    p.add_argument("--p1_dir",  default=None,
                   help="Override p1_dir from config")
    return p.parse_args()


def main():
    args   = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    p1_dir = args.p1_dir or config["p1_dir"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Project 1 dir: {p1_dir}")

    transforms = get_transforms(config["image_size"])

    for regime in config["regimes"]:
        for seed in config["seeds"]:
            run_id  = f"{config['dataset']}_{config['architecture']}" \
                      f"_{regime}_seed{seed}"
            out_dir = f"experiments/{run_id}"
            os.makedirs(out_dir, exist_ok=True)

            # ── Load checkpoint from Project 1 ────────────────────────────
            ckpt_path = os.path.join(p1_dir, "checkpoints", run_id, "best.pt")
            if not os.path.exists(ckpt_path):
                print(f"  SKIP {run_id} — checkpoint not found")
                continue

            set_seed(seed)
            loaders, n_classes = get_dataloaders(
                config["dataset"], transforms,
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
                image_size=config["image_size"],
            )

            model = build_model(
                config["architecture"], regime, n_classes
            ).to(device)
            model, _ = load_checkpoint(model, run_id, device,
                                       ckpt_dir=os.path.join(p1_dir, "checkpoints"))

            # ── Calibration metrics on test set ───────────────────────────
            model.eval()
            all_probs, all_labels = [], []
            with torch.no_grad():
                for inputs, targets in loaders["test"]:
                    inputs  = inputs.to(device)
                    targets = targets.view(-1).long()
                    logits  = model(inputs)
                    probs   = torch.softmax(logits, dim=1).cpu().numpy()
                    all_probs.append(probs)
                    all_labels.append(targets.numpy())

            probs_np  = np.concatenate(all_probs)
            labels_np = np.concatenate(all_labels)
            cal_metrics = compute_all_metrics(probs_np, labels_np)

            # ── Member vs non-member entropy from Project 1 outputs ───────
            p1_outputs = os.path.join(
                p1_dir, "experiments", run_id, "sample_outputs.json"
            )
            if os.path.exists(p1_outputs):
                with open(p1_outputs) as f:
                    records = json.load(f)
                entr_stats = member_nonmember_entropy(records)
                # Copy p1 outputs to local experiments dir
                with open(f"{out_dir}/sample_outputs_baseline.json", "w") as f:
                    json.dump(records, f)
            else:
                entr_stats = {}
                print(f"  WARNING: {p1_outputs} not found")

            # ── Copy p1 attack results as baseline ────────────────────────
            p1_attacks = os.path.join(
                p1_dir, "experiments", run_id, "attack_results.json"
            )
            if os.path.exists(p1_attacks):
                with open(p1_attacks) as f:
                    baseline_attacks = json.load(f)
                with open(f"{out_dir}/attack_results_baseline.json", "w") as f:
                    json.dump(baseline_attacks, f, indent=2)

            result = {
                "run_id":           run_id,
                "regime":           regime,
                "seed":             seed,
                "calibration":      cal_metrics,
                "entropy_analysis": entr_stats,
            }
            with open(f"{out_dir}/calibration_baseline.json", "w") as f:
                json.dump(result, f, indent=2)

            print(f"  {run_id} | ECE={cal_metrics['ece']:.4f} "
                  f"NLL={cal_metrics['nll']:.4f} "
                  f"entr_gap={entr_stats.get('entropy_gap', 'N/A')}")

    print("\nPhase 1 complete.")


if __name__ == "__main__":
    main()
