from dataclasses import dataclass, field
from typing import List, Any, Optional

from hydra.core.config_store import ConfigStore
from hydra.conf import HydraConf, RunDir, JobConf
from omegaconf import MISSING

from .model import MLPConfig
from .scaler import ScalerConfig


defaults = ["_self_", {"mlp": "regressor"}, {"scaler": "normalize"}]


@dataclass
class Config:
    defaults: List[Any] = field(default_factory=lambda: defaults)
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


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
