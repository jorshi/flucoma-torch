"""
Dataset and DataLoader for Fluid Data
"""

from pathlib import Path
import json
from typing import Dict, Optional

import torch

from flucoma_torch.scaler import FluidBaseScaler


class FluidDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for Fluid data.
    """

    def __init__(self, source: torch.Tensor, target: torch.Tensor):
        """
        Initialize the dataset with data and targets.

        :param data: Input data as a tensor.
        :param targets: Target values as a tensor.
        """
        assert source.ndim == 2, "Source data should be a 2D tensor."
        assert target.ndim == 2, "Target data should be a 2D tensor."
        assert (
            source.shape[0] == target.shape[0]
        ), "Source and target must have the same number of samples."

        self.source = source
        self.target = target

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        return self.source[idx], self.target[idx]


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


def convert_fluid_labelset_to_tensor(fluid_data: Dict):
    """
    Create a one-hot encoded tensor from the labels.
    """
    # Assert that there is only one col in the data -- we assume that there is only
    # one label for each datapoint.
    assert fluid_data["cols"] == 1, "Expcted labelset to have one column only"

    # Sort the labeles -- this isn't exactly what FluCoMa does, but as long as the
    # order of the labels is correct in the classifier dict then we should be good.
    keys = sorted(list(fluid_data["data"].keys()))
    labels = sorted(list(set(fluid_data["data"][k][0] for k in keys)))
    assert len(labels) > 1, "Only a single label found!"

    data = []

    for key in keys:
        label_idx = labels.index(fluid_data["data"][key][0])
        onehot = torch.zeros(len(labels))
        onehot[label_idx] = 1.0
        data.append(onehot)

    data = torch.vstack(data)
    assert data.ndim == 2, "Data should be a 2D tensor."
    assert data.shape[1] == len(
        labels
    ), f"Data shape mismatch: expected {len(labels)} columns, got {data.shape[1]}."
    return data, labels


def load_regression_dataset(
    source_filename: str,
    target_filename: str,
    scaler: Optional[FluidBaseScaler] = None,
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
    source_scaler_dict = None
    target_scaler_dict = None
    if scaler is not None:
        scaler.fit(source_data)
        source_data = scaler.transform(source_data)
        source_scaler_dict = scaler.get_as_dict()

        scaler.fit(target_data)
        target_data = scaler.transform(target_data)
        target_scaler_dict = scaler.get_as_dict()

    dataset = FluidDataset(source_data, target_data)
    return dataset, source_scaler_dict, target_scaler_dict


def load_classifier_dateset(
    source_filename: str,
    target_filename: str,
    scaler: Optional[FluidBaseScaler] = None,
):
    """
    Load source and target datasets from JSON files and return a dataset
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
    target_data, target_labels = convert_fluid_labelset_to_tensor(target_data)

    if source_data.shape[0] != target_data.shape[0]:
        raise ValueError(
            "Source and target datasets must have the same number of samples."
        )

    # Apply scaler to input if needed
    source_scaler_dict = None
    if scaler is not None:
        scaler.fit(source_data)
        source_data = scaler.transform(source_data)
        source_scaler_dict = scaler.get_as_dict()

    dataset = FluidDataset(source_data, target_data)
    return dataset, source_scaler_dict, target_labels


def split_dataset_for_validation(dataset: FluidDataset, val_ratio: float):
    assert 0.0 < val_ratio < 1.0, "Expected val_ratio to be between 0.0 and 1.0"

    num_data = len(dataset)
    assert num_data > 1, "Expected a dataset with at least 2 items"

    num_val = int(num_data * val_ratio)
    idx = torch.randperm(num_data)
    val_idx = idx[:num_val]
    train_idx = idx[num_val:]
    assert len(train_idx) + len(val_idx) == num_data

    train_dataset = FluidDataset(
        source=dataset.source[train_idx], target=dataset.target[train_idx]
    )

    val_dataset = FluidDataset(
        source=dataset.source[val_idx], target=dataset.target[val_idx]
    )

    return train_dataset, val_dataset
