import os
import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    run_id: str,
    ckpt_dir: str = "checkpoints",
) -> None:
    save_dir = os.path.join(ckpt_dir, run_id)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(
        {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics":              metrics,
        },
        os.path.join(save_dir, "best.pt"),
    )


def load_checkpoint(
    model: nn.Module,
    run_id: str,
    device: torch.device,
    ckpt_dir: str = "checkpoints",
) -> tuple[nn.Module, dict]:
    path = os.path.join(ckpt_dir, run_id, "best.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint found at {path}")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt["metrics"]
