import logging
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from src.training.evaluate import evaluate
from src.training.checkpoint import save_checkpoint


def train(
    model: nn.Module,
    loaders: dict,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: dict,
    run_id: str,
    device: torch.device,
    logger: Optional[logging.Logger] = None,
    criterion: Optional[nn.Module] = None,
) -> tuple[list, float]:
    """Train and checkpoint on best validation AUROC.

    criterion defaults to CrossEntropyLoss. Pass LabelSmoothingLoss
    for the Phase 6 label smoothing experiments.
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    _log = logger.info if logger else print
    best_val_auc = -float("inf")
    history: list[dict] = []

    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0

        pbar = tqdm(loaders["train"], desc=f"E{epoch:03d}", leave=False,
                    dynamic_ncols=True)
        for inputs, targets in pbar:
            inputs  = inputs.to(device, non_blocking=True)
            targets = targets.view(-1).long().to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(inputs)
            loss   = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            bs = inputs.size(0)
            epoch_loss    += loss.item() * bs
            epoch_correct += (logits.argmax(1) == targets).sum().item()
            epoch_total   += bs
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})

        scheduler.step()

        train_loss = epoch_loss / epoch_total
        train_acc  = epoch_correct / epoch_total
        val = evaluate(model, loaders["val"], device, config["num_classes"])

        record = {
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc, 6),
            "val_loss":   round(val["loss"], 6),
            "val_acc":    round(val["accuracy"], 6),
            "val_f1":     round(val["macro_f1"], 6),
            "val_auc":    round(val["macro_auc"], 6),
        }
        history.append(record)

        marker = ""
        if val["macro_auc"] > best_val_auc:
            best_val_auc = val["macro_auc"]
            save_checkpoint(model, optimizer, epoch, record, run_id)
            marker = "  <- best"

        _log(f"E{epoch:3d} | train loss {train_loss:.4f} acc {train_acc:.4f} | "
             f"val loss {val['loss']:.4f} acc {val['accuracy']:.4f} "
             f"auc {val['macro_auc']:.4f}{marker}")

    return history, best_val_auc
