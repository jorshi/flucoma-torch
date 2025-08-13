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

from flucoma_torch.config import ClassifierConfig
from flucoma_torch.data import load_classifier_dateset, split_dataset_for_validation


def setup_data(cfg: DictConfig):
    # Load the dataset
    # TODO: split dataset into validation as well.
    scaler = instantiate(cfg.scaler) if cfg.scaler else None
    train_dataset, source_scaler, labels = load_classifier_dateset(
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
        "scaler": source_scaler,
        "scaler_name": scaler.name if scaler is not None else "none",
        "labels": labels,
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


@hydra.main(version_base=None, config_name="classifier_config")
def main(cfg: ClassifierConfig) -> None:
    logger.info("Starting training with config:")
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    # Load the data
    data = setup_data(cfg)

    # Create and fit the model
    fit = fit_model(cfg, data)

    # Save the model
    logger.info("Training complete. Saving model...")

    # MLPClassifier needs labels corresponding to the onehot
    # prediction along with the model weights.
    model_dict = fit["mlp"].model.get_as_dict()
    labels_dict = {"labels": data["labels"], "rows": len(data["labels"])}
    classifier_dict = {
        "labels": labels_dict,
        "mlp": model_dict,
    }

    model_path = "model.json"
    with open(model_path, "w") as f:
        json.dump(classifier_dict, f, indent=4)

    logger.info(f"Model saved to {model_path}")

    # Save the input scaler if it exists
    if data["scaler"]:
        source_scaler_path = f"source_{data["scaler_name"]}.json"
        with open(source_scaler_path, "w") as f:
            json.dump(data["scaler"], f, indent=4)
        logger.info(f"Source scaler saved to {source_scaler_path}")


if __name__ == "__main__":
    main()
