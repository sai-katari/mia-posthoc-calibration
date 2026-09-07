import numpy as np
import torch
from torch.utils.data import DataLoader
import medmnist
from medmnist import INFO

# Only datasets used in this study; extend here when adding BloodMNIST (Phase 7)
DATASET_REGISTRY = {
    "dermamnist": medmnist.DermaMNIST,
    "bloodmnist": medmnist.BloodMNIST,
}


def get_dataset_info(dataset_name: str) -> dict:
    info = INFO[dataset_name.lower()]
    return {
        "n_classes":   len(info["label"]),
        "n_channels":  info["n_channels"],
        "task":        info["task"],
        "label_names": info["label"],
    }


def get_dataloaders(
    dataset_name: str,
    transforms: dict,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    image_size: int = 224,
) -> tuple[dict, int]:
    """Return train / train_eval / val / test loaders and number of classes.

    train_eval uses the same data as train but with shuffle=False and
    test-split transforms, so it can be used for unbiased generalization
    gap measurement after checkpoint selection.
    """
    name = dataset_name.lower()
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{name}' not in registry. Available: {list(DATASET_REGISTRY)}")

    cls = DATASET_REGISTRY[name]
    n_classes = get_dataset_info(name)["n_classes"]

    loaders: dict[str, DataLoader] = {}

    for split in ("train", "val", "test"):
        transform = transforms.get(split, transforms["test"])
        ds = cls(split=split, transform=transform, download=True, size=image_size)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    # Non-shuffled train loader for post-training evaluation
    train_eval_ds = cls(
        split="train",
        transform=transforms.get("test", transforms["test"]),
        download=True,
        size=image_size,
    )
    loaders["train_eval"] = DataLoader(
        train_eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return loaders, n_classes


def verify_dataset(dataset_name: str, image_size: int = 224) -> None:
    """Print split sizes and image shape. Call before training to catch config errors."""
    from torchvision import transforms as T

    cls = DATASET_REGISTRY[dataset_name.lower()]
    probe_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])

    print(f"\nDataset: {dataset_name}")
    for split in ("train", "val", "test"):
        ds = cls(split=split, transform=probe_transform, download=True, size=image_size)
        img, lbl = ds[0]
        print(f"  {split:5s}: {len(ds):6,} samples | img {tuple(img.shape)} | lbl {lbl.shape}")

    info = get_dataset_info(dataset_name)
    print(f"  n_classes={info['n_classes']} | n_channels={info['n_channels']} | task={info['task']}")
    print()
