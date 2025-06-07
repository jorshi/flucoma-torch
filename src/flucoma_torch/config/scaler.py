from dataclasses import dataclass
from omegaconf import MISSING

from hydra.core.config_store import ConfigStore


@dataclass
class ScalerConfig:
    _target_: str = MISSING


@dataclass
class FluidNormalizeConfig(ScalerConfig):
    _target_: str = "flucoma_torch.scaler.FluidNormalize"
    min: float = 0.0
    max: float = 1.0


@dataclass
class FluidStandardizeConfig(ScalerConfig):
    _target_: str = "flucoma_torch.sclaer.FluidStandardize"


@dataclass
class FluidRobustScalerConfig(ScalerConfig):
    _target_: str = "flucoma_torch.scaler.FluidRobustScaler"
    low: float = 0.25
    high: float = 0.75


cs = ConfigStore.instance()
cs.store(group="scaler", name="normalize", node=FluidNormalizeConfig)
cs.store(group="scaler", name="standardize", node=FluidStandardizeConfig)
cs.store(group="scaler", name="robustscale", node=FluidRobustScalerConfig)
