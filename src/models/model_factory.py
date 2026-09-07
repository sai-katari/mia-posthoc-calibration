import torch.nn as nn
from src.models.resnet import (
    build_resnet18,
    get_parameter_groups as resnet_param_groups,
    print_trainable_summary,
)

SUPPORTED_ARCHITECTURES = {"resnet18"}


def build_model(architecture: str, regime: str, num_classes: int) -> nn.Module:
    arch = architecture.lower()
    if arch == "resnet18":
        model = build_resnet18(regime, num_classes)
        print_trainable_summary(model, regime)
        return model
    raise ValueError(f"Architecture '{arch}' not supported. Choose from {SUPPORTED_ARCHITECTURES}")


def get_optimizer_groups(architecture: str, model: nn.Module, regime: str, config: dict) -> list[dict]:
    arch = architecture.lower()
    if arch == "resnet18":
        return resnet_param_groups(model, regime, config)
    raise ValueError(f"Architecture '{arch}' not supported.")
