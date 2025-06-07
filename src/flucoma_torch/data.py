"""
Dataset and DataLoader for Fluid Data
"""

from pathlib import Path
import json
from typing import Dict, Optional, Literal

import torch


class FluidDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for Fluid data.
    """

    def __init__(self, data: torch.Tensor, targets: torch.Tensor):
        """
        Initialize the dataset with data and targets.

        :param data: Input data as a tensor.
        :param targets: Target values as a tensor.
        """
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def convert_fluid_dataset_to_tensor(fluid_data: Dict):
    data = []

    # Sort the keys to ensure consistent order
    keys = sorted(list(fluid_data["data"].keys()))
    for key in keys:
        data.append(fluid_data["data"][key])

    if len(data) == 0:
        raise ValueError("No data found in the fluid dataset.")

    data = torch.tensor(data, dtype=torch.float32)
    assert data.ndim == 2, "Data should be a 2D tensor."
    assert (
        data.shape[1] == fluid_data["cols"]
    ), f"Data shape mismatch: expected {fluid_data['cols']} columns, got {data.shape[1]}."
    return data


def load_regression_dataset(
    source_filename: str,
    target_filename: str,
    scaler: Optional[Literal["standardize, normalizer, robust_scale"]] = None,
):
    """
    Load source and target datasets from JSON files and return a dataset
    TODO: Figure out validation split
    """
    source_path = Path(source_filename)
    target_path = Path(target_filename)

    if not source_path.exists():
        raise FileNotFoundError("Source file does not exist.")
    if not target_path.exists():
        raise FileNotFoundError("Target file does not exist.")

    with open(source_path, "r") as f:
        source_data = json.load(f)

    with open(target_path, "r") as f:
        target_data = json.load(f)

    source_data = convert_fluid_dataset_to_tensor(source_data)
    target_data = convert_fluid_dataset_to_tensor(target_data)

    if source_data.shape[0] != target_data.shape[0]:
        raise ValueError(
            "Source and target datasets must have the same number of samples."
        )

    # Apply scaler if needed
