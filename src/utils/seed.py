import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set all RNG seeds for reproducibility.

    Note: some CUDA ops (e.g. atomics in scatter) remain non-deterministic
    even with deterministic=True. Record this in experiment configs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
