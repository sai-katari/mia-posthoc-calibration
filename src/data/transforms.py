from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(image_size: int = 224) -> dict:
    """Minimal baseline transforms shared across all splits.

    Augmentation is intentionally absent: augmentation affects prediction
    confidence and therefore membership leakage, so it is only introduced
    as an explicit privacy regularisation experiment (Phase 11).
    """
    t = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return {"train": t, "val": t, "test": t}
