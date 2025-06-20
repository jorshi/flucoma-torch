import pytest
from flucoma_torch.data import (
    load_regression_dataset,
    load_classifier_dateset,
    convert_fluid_labelset_to_tensor,
)


def test_load_regression_dataset():
    load_regression_dataset(
        "tests/data/feature_regressor_in.json", "tests/data/feature_regressor_out.json"
    )


def test_load_classifier_dataset():
    load_classifier_dateset(
        "tests/data/mlpc_help_data.json", "tests/data/mlpc_help_labels.json"
    )


def test_convert_fluid_labelset_to_tensor_bad_data():
    bad_data = {"cols": 2, "data": "no_data"}
    with pytest.raises(
        AssertionError, match="Expcted labelset to have one column only"
    ):
        convert_fluid_labelset_to_tensor(bad_data)
