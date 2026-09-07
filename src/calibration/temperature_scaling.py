import torch
import torch.nn as nn
from torch.optim import LBFGS


class TemperatureScaler(nn.Module):
    """Post-hoc calibration via temperature scaling.

    Log-temperature parameterization guarantees T > 0 without clamping.
    T is fitted by minimizing NLL on the validation set only — not on
    the test set or on the attack pool.
    """

    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self):
        return self.log_temperature.exp()

    def forward(self, logits):
        return logits / self.temperature

    def calibrate(self, model, val_loader, device):
        """Fit T on validation set. Returns the learned T value."""
        model.eval()
        logits_list, labels_list = [], []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs  = inputs.to(device)
                targets = targets.view(-1).long().to(device)  # handles [B] and [B,1]
                logits  = model(inputs)
                logits_list.append(logits)
                labels_list.append(targets)

        logits_all = torch.cat(logits_list)
        labels_all = torch.cat(labels_list)

        criterion  = nn.CrossEntropyLoss()
        optimizer  = LBFGS([self.log_temperature], lr=0.01, max_iter=100)

        def eval_step():
            optimizer.zero_grad()
            loss = criterion(logits_all / self.temperature, labels_all)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        return self.temperature.item()


class CalibratedModel(nn.Module):
    """Wraps a model with a fitted TemperatureScaler for inference."""

    def __init__(self, model, scaler):
        super().__init__()
        self.model  = model
        self.scaler = scaler

    def forward(self, x):
        return self.scaler(self.model(x))
