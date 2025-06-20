import pytest
import torch
from flucoma_torch.data import (
    FluidDataset,
    load_regression_dataset,
    load_classifier_dateset,
    convert_fluid_labelset_to_tensor,
    split_dataset_for_validation,
)


def test_load_regression_dataset():
    load_regression_dataset(
        "tests/data/feature_regressor_in.json", "tests/data/feature_regressor_out.json"
    )


def test_load_classifier_dataset():
    dataset, scaler, labels = load_classifier_dateset(
        "tests/data/mlpc_help_data.json", "tests/data/mlpc_help_labels.json"
    )
    assert len(dataset) == 512

    x, y = dataset[0]
    assert x.shape == (2,)
    assert y.shape == (4,)

    assert scaler is None
    assert labels == [
        "bottom left red",
        "bottom right green",
        "top left yellow",
        "top right blue",
    ]


def test_convert_fluid_labelset_to_tensor_bad_data():
    bad_data = {"cols": 2, "data": "no_data"}
    with pytest.raises(
        AssertionError, match="Expcted labelset to have one column only"
    ):
        convert_fluid_labelset_to_tensor(bad_data)


def test_convert_fluid_labelset_to_tensor():
    labelset = {
        "cols": 1,
        "data": {
            "0": ["label_1"],
            "1": ["label_2"],
            "3": ["label_1"],
            "2": ["label_3"],
        },
    }

    data, labels = convert_fluid_labelset_to_tensor(labelset)

    # Make sure the labels returned look correct
    assert len(labels) == 3
    assert labels.index("label_1") == 0
    assert labels.index("label_2") == 1
    assert labels.index("label_3") == 2

    # Now make sure the data is correct
    onehot = torch.zeros(3)
    onehot[0] = 1.0

    assert torch.all(data[0] == onehot)
    assert torch.all(data[3] == onehot)
    assert torch.all(data[1] == onehot.roll(1))
    assert torch.all(data[2] == onehot.roll(2))


def test_split_dataset_for_validation():
    dataset = FluidDataset(
        source=torch.rand(24, 4),
        target=torch.rand(24, 1),
    )

    assert len(dataset) == 24

    # Assertion should be thrown for val sizes not 0.0 < val_size < 1.0
    with pytest.raises(AssertionError):
        split_dataset_for_validation(dataset, 0.0)

    with pytest.raises(AssertionError):
        split_dataset_for_validation(dataset, 1.0)

    with pytest.raises(AssertionError):
        split_dataset_for_validation(dataset, 2.0)

    train, val = split_dataset_for_validation(dataset, 0.25)
    assert len(train) == 18
    assert len(val) == 6

    for x, y in train:
        assert torch.sum(x == dataset.source) == 4  # Point exists once in original data
        assert torch.sum(x == val.source) == 0  # Does not exist in val
        assert torch.sum(y == dataset.target) == 1  # Exists onse in original data
        assert torch.sum(y == val.target) == 0  # Does exist in val

    for x, y in val:
        assert torch.sum(x == dataset.source) == 4
        assert torch.sum(x == train.source) == 0
        assert torch.sum(y == dataset.target) == 1
        assert torch.sum(y == train.target) == 0
