import torch
import torch.nn as nn


class LabelSmoothingLoss(nn.Module):
    """Cross-entropy with label smoothing.

    Replaces hard one-hot targets with:
        y_smooth_i = (1 - alpha)  if i == true class
        y_smooth_i = alpha / (C - 1)  otherwise

    This is a training-time defense. It reduces overconfidence and
    the generalization gap, serving as a comparison baseline against
    the post-hoc temperature scaling defense studied in this project.
    """

    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError(f"smoothing must be in [0, 1), got {smoothing}")
        self.smoothing   = smoothing
        self.num_classes = num_classes

    def forward(self, logits, targets):
        confidence = 1.0 - self.smoothing
        smooth_val = self.smoothing / (self.num_classes - 1)
        one_hot    = torch.zeros_like(logits).scatter_(
            1, targets.unsqueeze(1), 1
        )
        smooth_targets = one_hot * confidence + (1 - one_hot) * smooth_val
        log_probs      = nn.functional.log_softmax(logits, dim=1)
        return -(smooth_targets * log_probs).sum(dim=1).mean()
