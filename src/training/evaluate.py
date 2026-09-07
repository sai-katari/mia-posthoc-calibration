import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score


def evaluate(model: nn.Module, loader, device, num_classes: int) -> dict:
    """Run inference on a DataLoader and return aggregate + raw outputs.

    Returns
    -------
    dict with keys:
        loss, accuracy, macro_f1, macro_auc   — scalars
        targets, probs, preds                  — numpy arrays (for MIA use in Phase 3+)
    """
    criterion = nn.CrossEntropyLoss(reduction="sum")
    model.eval()

    all_targets: list[np.ndarray] = []
    all_probs:   list[np.ndarray] = []
    all_preds:   list[np.ndarray] = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs  = inputs.to(device, non_blocking=True)
            targets = targets.squeeze(1).long().to(device, non_blocking=True)

            logits = model(inputs)
            loss   = criterion(logits, targets)
            probs  = torch.softmax(logits, dim=1)
            preds  = probs.argmax(dim=1)

            total_loss    += loss.item()
            total_samples += inputs.size(0)

            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    targets_np = np.concatenate(all_targets)
    probs_np   = np.concatenate(all_probs)
    preds_np   = np.concatenate(all_preds)

    avg_loss = total_loss / total_samples
    accuracy = float((preds_np == targets_np).mean())
    macro_f1 = float(f1_score(targets_np, preds_np, average="macro", zero_division=0))

    try:
        macro_auc = float(
            roc_auc_score(targets_np, probs_np, multi_class="ovr", average="macro")
        )
    except ValueError:
        # Raised when only one class is present in a tiny split
        macro_auc = float("nan")

    return {
        "loss":      avg_loss,
        "accuracy":  accuracy,
        "macro_f1":  macro_f1,
        "macro_auc": macro_auc,
        # Raw arrays preserved for MIA attack computation (Phase 3+)
        "targets": targets_np,
        "probs":   probs_np,
        "preds":   preds_np,
    }
