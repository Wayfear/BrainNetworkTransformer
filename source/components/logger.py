import logging
from pathlib import Path
from typing import Tuple
from omegaconf import DictConfig


def get_formatter() -> logging.Formatter:
    return logging.Formatter('[%(asctime)s][%(filename)s][L%(lineno)d][%(levelname)s] %(message)s')


def initialize_logger() -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()

    formatter = get_formatter()
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def set_file_handler(log_file_path: Path) -> logging.Logger:
    logger = initialize_logger()
    formatter = get_formatter()
    file_handler = logging.FileHandler(str(log_file_path))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def logger_factory(config: DictConfig) -> Tuple[logging.Logger]:
    log_path = Path(config.log_path) / config.unique_id
    log_path.mkdir(exist_ok=True, parents=True)
    logger = set_file_handler(log_file_path=log_path
                              / config.unique_id)
    return logger
