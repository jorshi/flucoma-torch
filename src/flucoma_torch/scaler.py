"""
FluCoMa Scalers
"""

from typing import Dict

import torch


class FluidBaseScaler:
    """
    Base class for FluCoMa scalers.
    """

    def fit(self, data: torch.Tensor):
        """
        Fit the scaler to the data.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Transform the data using the fitted scaler.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def get_as_dict(self) -> Dict:
        """
        Get the scaler parameters as a dictionary.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")


class FluidNormalize(FluidBaseScaler):
    """
    Normalizer scaler for FluCoMa data.
    """

    def __init__(self, min: float = 0.0, max: float = 1.0):
        """
        Initialize the normalizer with min and max values.
        """
        self.min = min
        self.max = max

    def fit(self, data: torch.Tensor):
        assert data.ndim == 2, "Data should be a 2D tensor."
        self.data_min = data.min(dim=0).values
        self.data_max = data.max(dim=0).values

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        assert data.ndim == 2, "Data should be a 2D tensor."
        normalized_data = (data - self.data_min) / (self.data_max - self.data_min)
        return normalized_data * (self.max - self.min) + self.min

    def get_as_dict(self) -> Dict:
        return {
            "cols": self.data_min.shape[0],
            "data_max": self.data_max.tolist(),
            "data_min": self.data_min.tolist(),
            "max": self.max,
            "min": self.min,
        }


class FluidStandardize(FluidBaseScaler):
    """
    Standardizer scaler for FluCoMa data.
    """

    def fit(self, data: torch.Tensor):
        assert data.ndim == 2, "Data should be a 2D tensor."
        self.mean = data.mean(dim=0)
        self.std = torch.mean(torch.square(data - self.mean), dim=0).sqrt()
        self.std[self.std < 10 * torch.finfo(self.std.dtype).smallest_normal] = 1.0

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        assert data.ndim == 2, "Data should be a 2D tensor."
        standardized_data = (data - self.mean) / self.std
        return standardized_data

    def get_as_dict(self) -> Dict:
        return {
            "cols": self.mean.shape[0],
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }
