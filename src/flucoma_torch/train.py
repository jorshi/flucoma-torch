"""
CLI entry point for training the model.
"""

import hydra
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf

from flucoma_torch.config import Config


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    logger.info("Starting training with config:")
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    cfg.mlp["input_size"] = 1
    cfg.mlp["output_size"] = 1
    mlp = instantiate(cfg.mlp)

    print("Model instantiated:", mlp)

    # task = instantiate(cfg.module)

    # data = instantiate(cfg.datamodule)
    # data.setup("fit")
    # train_loader = data.train_dataloader()

    # trainer = instantiate(cfg.trainer)
    # trainer.fit(model=task, train_dataloaders=train_loader)


if __name__ == "__main__":
    main()
