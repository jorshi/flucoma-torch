"""
Tests for the scaler module.
"""

import json

import torch

from flucoma_torch.scaler import FluidNormalize, FluidStandardize
from flucoma_torch.data import convert_fluid_dataset_to_tensor


def test_fluid_normalize():
    data_file = "tests/data/feature_regressor_in.json"
    with open(data_file, "r") as f:
        fluid_data = json.load(f)
    data = convert_fluid_dataset_to_tensor(fluid_data)
    scaler = FluidNormalize(min=0.0, max=1.0)
    scaler.fit(data)
    transformed_data = scaler.transform(data)

    # Check the transformed data is within the expected range
    assert (
        transformed_data.min() >= 0.0
    ), "Transformed data has values below the minimum."
    assert (
        transformed_data.max() <= 1.0
    ), "Transformed data has values above the maximum."

    # Check the transformed data matches the expected normalized data
    flucoma_normed = "tests/data/feature_regressor_in_normalized.json"
    with open(flucoma_normed, "r") as f:
        expected_data = json.load(f)
    expected_data = convert_fluid_dataset_to_tensor(expected_data)
    torch.testing.assert_close(transformed_data, expected_data)

    # Check the dictionary is correctly populated
    scaler_dict = scaler.get_as_dict()

    flucoma_file = "tests/data/feature_regressor_in_normalize.json"
    with open(flucoma_file, "r") as f:
        expected_scaler = json.load(f)

    assert scaler_dict["cols"] == expected_scaler["cols"]
    assert scaler_dict["max"] == expected_scaler["max"]
    assert scaler_dict["min"] == expected_scaler["min"]
    torch.testing.assert_close(
        torch.tensor(scaler_dict["data_max"]),
        torch.tensor(expected_scaler["data_max"]),
    )
    torch.testing.assert_close(
        torch.tensor(scaler_dict["data_min"]),
        torch.tensor(expected_scaler["data_min"]),
    )


def test_fluid_standardize():
    data_file = "tests/data/feature_regressor_in.json"
    with open(data_file, "r") as f:
        fluid_data = json.load(f)
    data = convert_fluid_dataset_to_tensor(fluid_data)
    scaler = FluidStandardize()
    scaler.fit(data)
    transformed_data = scaler.transform(data)

    # Check the transformed data has mean close to 0 and std close to 1
    torch.testing.assert_close(transformed_data.mean(dim=0), torch.zeros(data.shape[1]))
    torch.testing.assert_close(
        transformed_data.std(dim=0, unbiased=False), torch.ones(data.shape[1])
    )

    # Check the transformed data matches the expected standardized data
    flucoma_std = "tests/data/feature_regressor_in_standardized.json"
    with open(flucoma_std, "r") as f:
        expected_data = json.load(f)
    expected_data = convert_fluid_dataset_to_tensor(expected_data)
    torch.testing.assert_close(transformed_data, expected_data)

    # Check the dictionary is correctly populated
    scaler_dict = scaler.get_as_dict()
    flucoma_file = "tests/data/feature_regressor_in_standardize.json"
    with open(flucoma_file, "r") as f:
        expected_scaler = json.load(f)

    assert scaler_dict["cols"] == expected_scaler["cols"]
    torch.testing.assert_close(
        torch.tensor(scaler_dict["mean"]),
        torch.tensor(expected_scaler["mean"]),
    )
    torch.testing.assert_close(
        torch.tensor(scaler_dict["std"]),
        torch.tensor(expected_scaler["std"]),
    )
