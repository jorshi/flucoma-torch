# Integration test for training a regressor model
import os
from pathlib import Path
import json

from hydra import initialize_config_module, compose
import pytest
import numpy as np

from flucoma_torch import train_classifier

test_data = Path(__file__).parent.joinpath("data")


@pytest.fixture
def rundir(tmp_path):
    current_dir = os.getcwd()
    yield tmp_path
    os.chdir(current_dir)


def test_train_regressor(rundir):
    with initialize_config_module(
        version_base=None, config_module="flucoma_torch.config"
    ):
        source_arg = f"source={test_data.joinpath('mlpc_help_data.json')}"
        target_arg = f"target={test_data.joinpath('mlpc_help_labels.json')}"
        cfg = compose(
            "classifier_config",
            overrides=[
                source_arg,
                target_arg,
                "mlp.max_iter=2",
                "mlp.hidden_layers=[6]",
                "mlp.activation=1",
            ],
        )

        os.chdir(rundir)
        train_classifier.main(cfg)

        # Load a target model file to check against
        target_model_path = test_data.joinpath("mlpc_fit.json")
        with open(target_model_path) as f:
            target_model = json.load(f)

        # Load the trained model file
        assert Path("model.json").exists()
        with open("model.json") as f:
            trained_model = json.load(f)

        # Checkt the labels are correct
        assert "labels" in trained_model
        trained_labels = trained_model["labels"]["labels"]
        assert set(trained_labels) == set(target_model["labels"]["labels"])
        assert trained_model["labels"]["rows"] == target_model["labels"]["rows"]

        # Check the layers are correct
        assert "mlp" in trained_model
        assert len(trained_model["mlp"]["layers"]) == len(target_model["mlp"]["layers"])
        for trained_layer, target_layer in zip(
            trained_model["mlp"]["layers"], target_model["mlp"]["layers"]
        ):
            assert trained_layer["cols"] == target_layer["cols"]
            assert trained_layer["rows"] == target_layer["rows"]
            assert trained_layer["activation"] == target_layer["activation"]
            assert (
                np.array(trained_layer["weights"]).shape
                == np.array(target_layer["weights"]).shape
            )
            assert (
                np.array(trained_layer["biases"]).shape
                == np.array(target_layer["biases"]).shape
            )

        # Check the normalizer file is correctly saved
        assert Path("source_normalize.json").exists()
        with open("source_normalize.json") as f:
            source_scaler = json.load(f)

        assert source_scaler["cols"] == cfg.mlp.input_size
        assert source_scaler["min"] == 0.0
        assert source_scaler["max"] == 1.0
        assert len(source_scaler["data_min"]) == cfg.mlp.input_size
        assert len(source_scaler["data_max"]) == cfg.mlp.input_size
