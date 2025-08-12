"""
Hyperparameter Optimization with Optuna.
"""

from functools import partial
import json
import os
from pathlib import Path
from typing import Optional

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf
import optuna
from optuna.artifacts import FileSystemArtifactStore
from optuna.integration import PyTorchLightningPruningCallback

from flucoma_torch.config import OptimizeClassifierConfig
from flucoma_torch.train_classifier import setup_data, fit_model


def objective(
    trial, cfg: DictConfig, artifact_store: Optional[FileSystemArtifactStore] = None
):
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

    # Optimize on val loss if it is being used
    metric = "val_loss" if cfg.mlp.validation > 0.0 else "train_loss"

    # Callback integration to enable optuna to prune trials
    callbacks = [PyTorchLightningPruningCallback(trial, monitor=metric)]

    # Setup data and model, then train
    data = setup_data(cfg)
    fit = fit_model(cfg, data, extra_callbacks=callbacks)

    # Save model artefacts
    if artifact_store is not None:
        # Save the model json
        model_dict = fit["mlp"].model.get_as_dict()
        labels_dict = {"labels": data["labels"], "rows": len(data["labels"])}
        classifier_dict = {
            "labels": labels_dict,
            "mlp": model_dict,
        }

        model_path = "model.json"
        with open(model_path, "w") as f:
            json.dump(classifier_dict, f, indent=4)

        # Upload as optuna artefact
        artifact_id = optuna.artifacts.upload_artifact(
            artifact_store=artifact_store,
            file_path=model_path,
            study_or_trial=trial,
        )
        trial.set_user_attr("model_artifact_id", artifact_id)

    return fit["trainer"].callback_metrics[metric]


@hydra.main(version_base=None, config_name="optimize_classifier_config")
def main(cfg: OptimizeClassifierConfig) -> None:
    logger.info("Starting hyperparameter optimization with config:")
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    # Setup data to save the scaler -- we only need to save the scaler once
    data = setup_data(cfg)
    if data["scaler"]:
        source_scaler_path = f"source_{data["scaler_name"]}.json"
        with open(source_scaler_path, "w") as f:
            json.dump(data["scaler"], f, indent=4)
        logger.info(f"Source scaler saved to {source_scaler_path}")

    # Optuna pruner will cancel trials that aren't looking good after a specified
    # number of trials and model warmup steps.
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=cfg.n_startup_trials, n_warmup_steps=cfg.n_warmup_steps
    )

    # Create the artifact store
    base_path = "./artifacts"
    os.makedirs(base_path, exist_ok=True)
    artifact_store = optuna.artifacts.FileSystemArtifactStore(base_path=base_path)

    # Optional save the trial to a sqlite database
    storage = None
    if cfg.sqlite:
        storage = Path(cfg.storage_name)
        storage = f"sqlite:///{storage}.sqlite3"

    study = optuna.create_study(
        direction="minimize", pruner=pruner, storage=storage, study_name=cfg.study_name
    )

    # Run the study
    objective_func = partial(objective, cfg=cfg, artifact_store=artifact_store)
    study.optimize(objective_func, n_trials=cfg.n_trials)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Get the best trained model
    best_artifact_id = study.best_trial.user_attrs.get("model_artifact_id")
    download_path = Path("best_model.json")

    optuna.artifacts.download_artifact(
        artifact_store=artifact_store,
        artifact_id=best_artifact_id,
        file_path=download_path,
    )

    logger.info(f"Saved best model to {download_path}")

    # Remove temp model
    os.remove("model.json")

    # Save the best trial parameters
    hyperparam_path = "best_hyperparameters.json"
    with open(hyperparam_path, "w") as fp:
        json.dump(trial.params, fp, indent=4)

    logger.info(f"Saved best hyperparameters to {hyperparam_path}")
