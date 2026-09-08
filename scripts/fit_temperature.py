#!/usr/bin/env python3
"""Phase 2 — Fit temperature scaling on each checkpoint.

Calibrates T on val set, re-extracts per-sample outputs using scaled
probabilities, and saves calibration metrics.

Usage:
    python scripts/fit_temperature.py \
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
from src.calibration.temperature_scaling import TemperatureScaler, CalibratedModel
from src.calibration.metrics import (
    compute_all_metrics,
    member_nonmember_entropy,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",  required=True)
    p.add_argument("--p1_dir",  default=None)
    return p.parse_args()


def extract_outputs(model, loader, device, membership_label):
    model.eval()
    records = []
    idx = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs  = inputs.to(device)
            targets = targets.view(-1).long()
            logits  = model(inputs)
            probs   = torch.softmax(logits, dim=1).cpu()
            preds   = probs.argmax(dim=1)

            for i in range(len(targets)):
                t      = targets[i].item()
                p_vec  = probs[i].numpy()
                p_true = float(p_vec[t])
                p_vec_f64 = p_vec.astype(np.float64)
                p_vec_f64 /= p_vec_f64.sum()
                tiny = np.finfo(np.float64).tiny
                log_p = np.log(np.clip(p_vec_f64, tiny, 1.0))
                loss   = float(-log_p[t])
                entr   = float(-(p_vec_f64 * log_p).sum())
                records.append({
                    "sample_idx": idx,
                    "true_class": t,
                    "pred_class": int(preds[i].item()),
                    "prob_vector": p_vec.tolist(),
                    "p_true":     p_true,
                    "max_conf":   float(p_vec.max()),
                    "loss":       loss,
                    "entropy":    entr,
                    "correct":    int(preds[i].item() == t),
                    "membership": membership_label,
                })
                idx += 1
    return records


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    p1_dir = args.p1_dir or config["p1_dir"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    transforms = get_transforms(config["image_size"])

    for regime in config["regimes"]:
        for seed in config["seeds"]:
            run_id  = f"{config['dataset']}_{config['architecture']}" \
                      f"_{regime}_seed{seed}"
            out_dir = f"experiments/{run_id}"
            os.makedirs(out_dir, exist_ok=True)

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

            # Load base model
            model = build_model(
                config["architecture"], regime, n_classes
            ).to(device)
            model, _ = load_checkpoint(
                model, run_id, device,
                ckpt_dir=os.path.join(p1_dir, "checkpoints")
            )

            # Fit temperature on val set only
            scaler = TemperatureScaler().to(device)
            T = scaler.calibrate(model, loaders["val"], device)
            print(f"  {run_id} | T={T:.4f}")

            # Calibrated model for inference
            cal_model = CalibratedModel(model, scaler)

            # Calibration quality on test set (never on attack pool)
            cal_model.eval()
            all_probs, all_labels = [], []
            with torch.no_grad():
                for inputs, targets in loaders["test"]:
                    inputs  = inputs.to(device)
                    targets = targets.view(-1).long()
                    logits  = cal_model(inputs)
                    probs   = torch.softmax(logits, dim=1).cpu().numpy()
                    all_probs.append(probs)
                    all_labels.append(targets.numpy())

            probs_np  = np.concatenate(all_probs)
            labels_np = np.concatenate(all_labels)
            cal_metrics = compute_all_metrics(probs_np, labels_np)

            # Re-extract attack pool outputs with scaled probabilities
            print(f"    Extracting members...")
            members    = extract_outputs(
                cal_model, loaders["train_eval"], device, membership_label=1
            )
            print(f"    Extracting non-members...")
            nonmembers = extract_outputs(
                cal_model, loaders["test"], device, membership_label=0
            )

            records = members + nonmembers
            with open(f"{out_dir}/sample_outputs_temp_scaled.json", "w") as f:
                json.dump(records, f)

            entr_stats = member_nonmember_entropy(records)

            result = {
                "run_id":           run_id,
                "regime":           regime,
                "seed":             seed,
                "temperature":      round(T, 6),
                "calibration":      cal_metrics,
                "entropy_analysis": entr_stats,
            }
            with open(f"{out_dir}/calibration_temp_scaled.json", "w") as f:
                json.dump(result, f, indent=2)

            print(f"    ECE={cal_metrics['ece']:.4f} "
                  f"entr_gap={entr_stats.get('entropy_gap', 'N/A')}")

    print("\nPhase 2 complete.")


if __name__ == "__main__":
    main()
