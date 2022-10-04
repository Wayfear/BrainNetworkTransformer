import torch.nn as nn


def count_params(model: nn.Module, only_requires_grad: bool = False):
    "count number trainable parameters in a pytorch model"
    if only_requires_grad:
        total_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    else:
        total_params = sum(p.numel() for p in model.parameters())
    return total_params
