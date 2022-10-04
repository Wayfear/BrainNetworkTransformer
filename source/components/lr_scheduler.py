import bisect
import math
from typing import List
from omegaconf import DictConfig
import torch


class LRScheduler:
    def __init__(self, cfg: DictConfig,  optimizer_cfg: DictConfig):
        self.lr_config = optimizer_cfg.lr_scheduler
        self.training_config = cfg
        self.lr = optimizer_cfg.lr

        assert self.lr_config.mode in [
            'step', 'poly', 'cos', 'linear', 'decay']

    def update(self, optimizer: torch.optim.Optimizer, step: int):
        lr_config = self.lr_config
        lr_mode = lr_config.mode
        base_lr = lr_config.base_lr
        target_lr = lr_config.target_lr

        warm_up_from = lr_config.warm_up_from
        warm_up_steps = lr_config.warm_up_steps
        total_steps = self.training_config.total_steps

        assert 0 <= step <= total_steps
        if step < warm_up_steps:
            current_ratio = step / warm_up_steps
            self.lr = warm_up_from + (base_lr - warm_up_from) * current_ratio
        else:
            current_ratio = (step - warm_up_steps) / \
                (total_steps - warm_up_steps)
            if lr_mode == 'step':
                count = bisect.bisect_left(lr_config.milestones, current_ratio)
                self.lr = base_lr * pow(lr_config.decay_factor, count)
            elif lr_mode == 'poly':
                poly = pow(1 - current_ratio, lr_config.poly_power)
                self.lr = target_lr + (base_lr - target_lr) * poly
            elif lr_mode == 'cos':
                cosine = math.cos(math.pi * current_ratio)
                self.lr = target_lr + (base_lr - target_lr) * (1 + cosine) / 2
            elif lr_mode == 'linear':
                self.lr = target_lr + \
                    (base_lr - target_lr) * (1 - current_ratio)
            elif lr_mode == 'decay':
                epoch = step // self.training_config.steps_per_epoch
                self.lr = base_lr * lr_config.lr_decay ** epoch

        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr


def lr_scheduler_factory(lr_configs: List[DictConfig], cfg: DictConfig) -> List[LRScheduler]:
    return [LRScheduler(cfg=cfg, optimizer_cfg=lr_config) for lr_config in lr_configs]
