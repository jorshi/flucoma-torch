from dataclasses import dataclass, field
from typing import List, Any, Optional

from hydra.core.config_store import ConfigStore
from hydra.conf import HydraConf, RunDir, JobConf
from omegaconf import MISSING

from .model import MLPConfig
from .scaler import ScalerConfig


regressor_defaults = ["_self_", {"mlp": "regressor"}, {"scaler": "normalize"}]
classifier_defaults = ["_self_", {"mlp": "classifier"}, {"scaler": "normalize"}]


@dataclass
class RegressorConfig:
    defaults: List[Any] = field(default_factory=lambda: regressor_defaults)
    mlp: MLPConfig = MISSING
    scaler: Optional[ScalerConfig] = None

    source: str = MISSING
    target: str = MISSING

    hydra: HydraConf = field(
        default_factory=lambda: HydraConf(
            run=RunDir(
                dir="./outputs/${hydra.job.name}/${now:%Y-%m-%d}/${now:%H-%M-%S}"
            ),
            job=JobConf(chdir=True),
        )
    )


@dataclass
class ClassifierConfig:
    defaults: List[Any] = field(default_factory=lambda: classifier_defaults)
    mlp: MLPConfig = MISSING
    scaler: Optional[ScalerConfig] = None

    source: str = MISSING
    target: str = MISSING

    hydra: HydraConf = field(
        default_factory=lambda: HydraConf(
            run=RunDir(
                dir="./outputs/${hydra.job.name}/${now:%Y-%m-%d}/${now:%H-%M-%S}"
            ),
            job=JobConf(chdir=True),
        )
    )


@dataclass
class OptimizeClassifierConfig:
    defaults: List[Any] = field(default_factory=lambda: classifier_defaults)
    mlp: MLPConfig = MISSING
    scaler: Optional[ScalerConfig] = None

    source: str = MISSING
    target: str = MISSING

    # Optuna specific config
    study_name: str = "classifier_study"
    sqlite: bool = False
    storage_name: str = "classifier_study"
    n_trials: int = 100
    n_startup_trials: int = 10  # Number trials before start checking to prune
    n_warmup_steps: int = 100  # Number warm-up steps.

    hydra: HydraConf = field(
        default_factory=lambda: HydraConf(
            run=RunDir(
                dir="./outputs/${hydra.job.name}/${now:%Y-%m-%d}/${now:%H-%M-%S}"
            ),
            job=JobConf(chdir=True),
        )
    )


@dataclass
class OptimizeRegressorConfig:
    defaults: List[Any] = field(default_factory=lambda: regressor_defaults)
    mlp: MLPConfig = MISSING
    scaler: Optional[ScalerConfig] = None

    source: str = MISSING
    target: str = MISSING

    # Optuna specific config
    study_name: str = "classifier_study"
    sqlite: bool = False
    storage_name: str = "classifier_study"
    n_trials: int = 100
    n_startup_trials: int = 10  # Number trials before start checking to prune
    n_warmup_steps: int = 100  # Number warm-up steps.

    hydra: HydraConf = field(
        default_factory=lambda: HydraConf(
            run=RunDir(
                dir="./outputs/${hydra.job.name}/${now:%Y-%m-%d}/${now:%H-%M-%S}"
            ),
            job=JobConf(chdir=True),
        )
    )


cs = ConfigStore.instance()
cs.store(name="regressor_config", node=RegressorConfig)
cs.store(name="classifier_config", node=ClassifierConfig)
cs.store(name="optimize_classifier_config", node=OptimizeClassifierConfig)
cs.store(name="optimize_regressor_config", node=OptimizeRegressorConfig)
