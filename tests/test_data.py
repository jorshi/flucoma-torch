from flucoma_torch.data import load_regression_dataset


def test_load_regression_dataset():
    load_regression_dataset(
        "tests/data/feature_regressor_in.json", "tests/data/feature_regressor_out.json"
    )
