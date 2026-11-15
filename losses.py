"""Auxiliary losses for modality routing."""
import torch


def eva_imp(y_true, y_pred):
    return (y_pred - y_true) ** 2


def entropy_balance(probs):
    probs = torch.clamp(probs, min=1e-9)
    return torch.mean(probs.size(1) * torch.sum(probs * torch.log(probs), dim=1))
