import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

REGIMES = ("scratch", "frozen", "partial", "full")

# Partial fine-tuning: unfreeze these module name prefixes
_PARTIAL_UNFREEZE = ("layer4.", "fc.")


def build_resnet18(regime: str, num_classes: int) -> nn.Module:
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {REGIMES}, got '{regime}'")

    weights = None if regime == "scratch" else ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    # Replace classification head for target task
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if regime == "frozen":
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False

    elif regime == "partial":
        for name, param in model.named_parameters():
            if not any(name.startswith(pfx) for pfx in _PARTIAL_UNFREEZE):
                param.requires_grad = False

    # scratch and full: all params remain requires_grad=True by default

    return model


def get_parameter_groups(model: nn.Module, regime: str, config: dict) -> list[dict]:
    """Return AdamW-compatible parameter groups with per-component learning rates."""
    if regime == "scratch":
        return [{"params": list(model.parameters()), "lr": config["lr_scratch"]}]

    if regime == "frozen":
        return [{"params": list(model.fc.parameters()), "lr": config["lr_frozen_head"]}]

    if regime == "partial":
        backbone_params = [
            p for n, p in model.named_parameters()
            if n.startswith("layer4.") and p.requires_grad
        ]
        return [
            {"params": backbone_params,             "lr": config["lr_partial_backbone"]},
            {"params": list(model.fc.parameters()), "lr": config["lr_partial_head"]},
        ]

    if regime == "full":
        backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
        return [
            {"params": backbone_params,             "lr": config["lr_full_backbone"]},
            {"params": list(model.fc.parameters()), "lr": config["lr_full_head"]},
        ]

    raise ValueError(f"Unknown regime: {regime}")


def print_trainable_summary(model: nn.Module, regime: str) -> None:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / total
    print(f"  [{regime:8s}] {total:,} total params | {trainable:,} trainable ({pct:.1f}%)")


def verify_frozen_layers(model: nn.Module) -> None:
    """Print requires_grad status for every parameter tensor."""
    for name, param in model.named_parameters():
        status = "train" if param.requires_grad else "FROZEN"
        print(f"    {status:6s}  {name:50s}  {tuple(param.shape)}")
