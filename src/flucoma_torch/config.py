from typing import List, Any

from dataclasses import dataclass, field
from omegaconf import MISSING

from hydra.core.config_store import ConfigStore
from hydra.conf import HydraConf, RunDir, JobConf


@dataclass
class MLPConfig:
    _target_: str = MISSING


@dataclass
class FluidMLPRegressorConfig(MLPConfig):
    _target_: str = "flucoma_torch.task.FluidMLPRegressor"
    input_size: int = MISSING
    output_size: int = MISSING
    activation: int = 2
    batch_size: int = 50
    hidden_layers: list[int] = field(default_factory=lambda: [3, 3])
    learn_rate: float = 0.01
    max_iter: int = 1000
    momentum: float = 0.9
    output_activation: int = 0
    validation: float = 0.2


defaults = ["_self_", {"mlp": "regressor"}]


@dataclass
class Config:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    mlp: MLPConfig = MISSING

    hydra: HydraConf = field(
        default_factory=lambda: HydraConf(
            run=RunDir(
                dir="./outputs/${hydra.job.name}/${now:%Y-%m-%d}/${now:%H-%M-%S}"
            ),
            job=JobConf(chdir=True),
        )
    )


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="mlp", name="regressor", node=FluidMLPRegressorConfig)
