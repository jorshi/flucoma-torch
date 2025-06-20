"""
CLI entry point for training the model.
"""

import json
from typing import Dict, List, Optional

import hydra
from hydra.utils import instantiate
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from loguru import logger
from omegaconf import DictConfig, OmegaConf
import torch

from flucoma_torch.config import RegressorConfig
from flucoma_torch.data import load_regression_dataset, split_dataset_for_validation


def setup_data(cfg: DictConfig):
    # Load the dataset
    scaler = instantiate(cfg.scaler) if cfg.scaler else None
    train_dataset, source_scaler, target_scaler = load_regression_dataset(
        source_filename=hydra.utils.to_absolute_path(cfg.source),
        target_filename=hydra.utils.to_absolute_path(cfg.target),
        scaler=scaler,
    )
    logger.info(f"Loaded dataset with {len(train_dataset)} samples.")

    # Split dataset if using validation
    val_ratio = cfg.mlp.validation
    val_dataloader = None
    callbacks = []
    if val_ratio > 0.0:
        logger.info(f"Using a validation split ratio of {val_ratio}")
        train_dataset, val_dataset = split_dataset_for_validation(
            train_dataset, val_ratio
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=cfg.mlp.batch_size, shuffle=False
        )
        early_stopping = EarlyStopping("val_loss", patience=20)
        callbacks.append(early_stopping)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.mlp.batch_size, shuffle=True
    )

    data = {
        "train_dataloader": train_dataloader,
        "train_dataset": train_dataset,
        "val_dataloader": val_dataloader,
        "callbacks": callbacks,
        "source_scaler": source_scaler,
        "target_scaler": target_scaler,
        "scaler_name": scaler.name if scaler is not None else "none",
    }
    return data


def fit_model(
    cfg: DictConfig, data: Dict, extra_callbacks: Optional[List[L.Callback]] = None
):
    # Initialize the model
    cfg.mlp["input_size"] = data["train_dataset"][0][0].shape[0]
    cfg.mlp["output_size"] = data["train_dataset"][0][1].shape[0]
    mlp = instantiate(cfg.mlp)

    # Setup callbacks
    callbacks = []
    if data["callbacks"] is not None:
        callbacks.extend(data["callbacks"])
    if extra_callbacks is not None:
        callbacks.extend(extra_callbacks)

    # Train the model
    trainer = L.Trainer(max_epochs=cfg.mlp.max_iter, callbacks=callbacks)
    logger.info("Starting training...")
    trainer.fit(mlp, data["train_dataloader"], val_dataloaders=data["val_dataloader"])

    return {"mlp": mlp, "trainer": trainer}


@hydra.main(version_base=None, config_name="regressor_config")
def main(cfg: RegressorConfig) -> None:
    logger.info("Starting training with config:")
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    # Load datasets and scalers
    data = setup_data(cfg)

    # Initialize and train model
    fit = fit_model(cfg, data)

    # Save the model
    logger.info("Training complete. Saving model...")
    model_path = "model.json"
    fit["mlp"].model.save(model_path)
    logger.info(f"Model saved to {model_path}")

    # Save the scalers if they exist
    if data["source_scaler"]:
        source_scaler_path = f"source_{data['scaler_name']}.json"
        with open(source_scaler_path, "w") as f:
            json.dump(data["source_scaler"], f, indent=4)
        logger.info(f"Source scaler saved to {source_scaler_path}")

    if data["target_scaler"]:
        target_scaler_path = f"target_{data['scaler_name']}.json"
        with open(target_scaler_path, "w") as f:
            json.dump(data["target_scaler"], f, indent=4)
        logger.info(f"Target scaler saved to {target_scaler_path}")


if __name__ == "__main__":
    main()
