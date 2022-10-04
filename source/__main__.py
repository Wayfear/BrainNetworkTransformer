from datetime import datetime
import wandb
import hydra
from omegaconf import DictConfig, open_dict
from .dataset import dataset_factory
from .models import model_factory
from .components import lr_scheduler_factory, optimizers_factory, logger_factory
from .training import training_factory
from datetime import datetime


def model_training(cfg: DictConfig):

    with open_dict(cfg):
        cfg.unique_id = datetime.now().strftime("%m-%d-%H-%M-%S")

    dataloaders = dataset_factory(cfg)
    logger = logger_factory(cfg)
    model = model_factory(cfg)
    optimizers = optimizers_factory(
        model=model, optimizer_configs=cfg.optimizer)
    lr_schedulers = lr_scheduler_factory(lr_configs=cfg.optimizer,
                                         cfg=cfg)
    training = training_factory(cfg, model, optimizers,
                                lr_schedulers, dataloaders, logger)

    training.train()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    group_name = f"{cfg.dataset.name}_{cfg.model.name}_{cfg.datasz.percentage}_{cfg.preprocess.name}"
    # _{cfg.training.name}\
    # _{cfg.optimizer[0].lr_scheduler.mode}"

    for _ in range(cfg.repeat_time):
        run = wandb.init(project=cfg.project, entity=cfg.wandb_entity, reinit=True,
                         group=f"{group_name}", tags=[f"{cfg.dataset.name}"])
        model_training(cfg)

        run.finish()


if __name__ == '__main__':
    main()
