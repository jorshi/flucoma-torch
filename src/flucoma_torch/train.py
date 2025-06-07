"""
CLI entry point for training the model.
"""

import hydra
from hydra.utils import instantiate
import lightning as L
from loguru import logger
from omegaconf import OmegaConf
import torch

from flucoma_torch.config import Config
from flucoma_torch.data import load_regression_dataset


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    logger.info("Starting training with config:")
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    # Load the dataset
    dataset, source_scaler, target_scaler = load_regression_dataset(
        source_filename=hydra.utils.to_absolute_path(cfg.source),
        target_filename=hydra.utils.to_absolute_path(cfg.target),
        scaler=instantiate(cfg.scaler) if cfg.scaler else None,
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.mlp.batch_size, shuffle=True
    )

    cfg.mlp["input_size"] = dataset[0][0].shape[0]
    cfg.mlp["output_size"] = dataset[0][1].shape[0]
    mlp = instantiate(cfg.mlp)

    trainer = L.Trainer(max_epochs=cfg.mlp.max_iter)
    logger.info("Starting training...")
    trainer.fit(mlp, train_dataloader)

    # Save the model
    logger.info("Training complete. Saving model...")
    model_path = "model.json"
    mlp.model.save(model_path)


if __name__ == "__main__":
    main()
