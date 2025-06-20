"""
Hyperparameter Optimization with Optuna.
"""

from functools import partial
import os
from pathlib import Path
from typing import Optional

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf
import optuna
from optuna.artifacts import FileSystemArtifactStore

from flucoma_torch.config import OptimizeClassifierConfig
from flucoma_torch.train_classifier import setup_data, fit_model


def objective(
    trial, cfg: DictConfig, artifact_store: Optional[FileSystemArtifactStore] = None
):
    data = setup_data(cfg)

    # Model hyperparameters -- override the cfg
    n_layers = trial.suggest_int("n_layers", 1, 8)
    layers = []

    layer_sizes = [2**i for i in range(0, 9)]
    for i in range(n_layers):
        layers.append(trial.suggest_categorical(f"n_units_l{i}", layer_sizes))

    cfg.mlp.hidden_layers = layers
    cfg.mlp.activation = trial.suggest_int("activation", 0, 3)
    cfg.mlp.batch_size = trial.suggest_categorical("batch_size", [2, 4, 8, 16, 32, 64])
    cfg.mlp.learn_rate = trial.suggest_float("lr", 1e-6, 1.0, log=True)
    cfg.mlp.momentum = trial.suggest_float("momentum", 0.0, 1.0)

    fit = fit_model(cfg, data)
    return fit["trainer"].callback_metrics["val_loss"]


@hydra.main(version_base=None, config_name="optimize_classifier_config")
def main(cfg: OptimizeClassifierConfig) -> None:
    logger.info("Starting hyperparameter optimization with config:")
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=cfg.n_startup_trials, n_warmup_steps=cfg.n_warmup_steps
    )

    # Create the artifact store
    base_path = "./artifacts"
    os.makedirs(base_path, exist_ok=True)
    artifact_store = optuna.artifacts.FileSystemArtifactStore(base_path=base_path)

    storage = None
    if cfg.mysql:
        storage = Path(cfg.storage)
        storage = f"sqllite:///{storage}.sqlite3"

    study = optuna.create_study(
        direction="minimize", pruner=pruner, storage=storage, study_name=cfg.study_name
    )

    objective_func = partial(objective, cfg=cfg, artifact_store=artifact_store)
    study.optimize(objective_func, n_trials=cfg.n_trials)
