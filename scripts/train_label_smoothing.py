#!/usr/bin/env python3
"""Phase 6 — Retrain with label smoothing (training-time defense).

Trains the full FT regime (highest leakage in Project 1) with
label smoothing alpha=0.1 as a comparison against post-hoc
temperature scaling.

Usage:
    python scripts/train_label_smoothing.py \
        --config configs/label_smoothing.yaml \
        --regime full \
        --seed 42
"""

import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.seed import set_seed
from src.utils.logging import setup_logger
from src.data.transforms import get_transforms
from src.data.medmnist_loader import get_dataloaders
from src.models.model_factory import build_model, get_optimizer_groups
from src.training.trainer import train
from src.training.evaluate import evaluate
from src.training.checkpoint import load_checkpoint
from src.defenses.label_smoothing import LabelSmoothingLoss
from src.attacks.base_attacks import run_all_attacks

SEEDS = [42, 123, 2026]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--regime", required=True,
                   choices=["scratch", "frozen", "partial", "full"])
    p.add_argument("--seed",   type=int, required=True, choices=SEEDS)
    return p.parse_args()


def scalar_metrics(m):
    return {k: v for k, v in m.items()
            if k not in ("targets", "probs", "preds")}


def extract_outputs(model, loader, device, membership_label):
    model.eval()
    records, idx = [], 0
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
                records.append({
                    "sample_idx": idx,
                    "true_class": t,
                    "pred_class": int(preds[i].item()),
                    "prob_vector": p_vec.tolist(),
                    "p_true":     p_true,
                    "max_conf":   float(p_vec.max()),
                    "loss":       float(-np.log(np.clip(p_true, 1e-300, None))),
                    "entropy":    float(-(p_vec * np.log(
                                    np.clip(p_vec, 1e-300, None))).sum()),
                    "correct":    int(preds[i].item() == t),
                    "membership": membership_label,
                })
                idx += 1
    return records


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_id = (f"{config['dataset']}_{config['architecture']}"
              f"_{args.regime}_ls{config['smoothing']}_seed{args.seed}")
    logger = setup_logger(run_id)
    logger.info(f"Run: {run_id}")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    transforms = get_transforms(config["image_size"])
    loaders, n_classes = get_dataloaders(
        config["dataset"], transforms,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        image_size=config["image_size"],
    )
    config["num_classes"] = n_classes

    model = build_model(config["architecture"], args.regime, n_classes).to(device)
    criterion = LabelSmoothingLoss(n_classes, smoothing=config["smoothing"])

    out_dir = f"experiments/{run_id}"
    os.makedirs(out_dir, exist_ok=True)

    with open(f"{out_dir}/config.json", "w") as f:
        json.dump({
            **config, "training_regime": args.regime,
            "seed": args.seed, "run_id": run_id,
            "defense": "label_smoothing",
            "torch_version": torch.__version__,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, f, indent=2)

    param_groups = get_optimizer_groups(
        config["architecture"], model, args.regime, config
    )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["max_epochs"]
    )

    logger.info("Training with label smoothing...")
    history, best_val_auc = train(
        model, loaders, optimizer, scheduler, config,
        run_id, device, logger, criterion=criterion
    )

    with open(f"{out_dir}/history.json", "w") as f:
        json.dump(history, f, indent=2)

    model, _ = load_checkpoint(model, run_id, device)

    results = {}
    for split, key in [("train", "train_eval"), ("val", "val"), ("test", "test")]:
        results[split] = scalar_metrics(
            evaluate(model, loaders[key], device, n_classes)
        )
    results["generalization"] = {
        "accuracy_gap": round(results["train"]["accuracy"] -
                               results["test"]["accuracy"], 6),
        "loss_gap":     round(results["test"]["loss"] -
                               results["train"]["loss"], 6),
    }

    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Extract attack pool outputs and run attacks
    members    = extract_outputs(model, loaders["train_eval"], device, 1)
    nonmembers = extract_outputs(model, loaders["test"],       device, 0)
    records    = members + nonmembers

    with open(f"{out_dir}/sample_outputs.json", "w") as f:
        json.dump(records, f)

    attack_results = run_all_attacks(records)
    with open(f"{out_dir}/attack_results.json", "w") as f:
        json.dump(attack_results, f, indent=2)

    logger.info(f"Test  | acc {results['test']['accuracy']:.4f} "
                f"auc {results['test']['macro_auc']:.4f}")
    logger.info(f"Loss MIA AUC: {attack_results[0]['auc']:.4f}")
    logger.info(f"Gen gap: {results['generalization']['accuracy_gap']:+.4f}")


if __name__ == "__main__":
    main()
