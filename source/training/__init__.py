from operator import mod
from .training import Train
from .FBNettraining import FBNetTrain
from omegaconf import DictConfig
from typing import List
import torch
from source.components import LRScheduler
import logging
import torch.utils.data as utils


def training_factory(config: DictConfig,
                     model: torch.nn.Module,
                     optimizers: List[torch.optim.Optimizer],
                     lr_schedulers: List[LRScheduler],
                     dataloaders: List[utils.DataLoader],
                     logger: logging.Logger) -> Train:

    train = config.model.get("train", None)
    if not train:
        train = config.training.name
    return eval(train)(cfg=config,
                       model=model,
                       optimizers=optimizers,
                       lr_schedulers=lr_schedulers,
                       dataloaders=dataloaders,
                       logger=logger)
