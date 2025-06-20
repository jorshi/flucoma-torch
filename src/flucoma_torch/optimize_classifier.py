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


def objective(
    trial, cfg: DictConfig, artifact_store: Optional[FileSystemArtifactStore] = None
):
    return 0.0


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
