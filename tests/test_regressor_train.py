# Integration test for training a regressor model
import os
from pathlib import Path
import json

from hydra import initialize_config_module, compose
import pytest
import numpy as np

from flucoma_torch import train_regressor

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
        source_arg = f"source={test_data.joinpath('feature_regressor_in.json')}"
        target_arg = f"target={test_data.joinpath('feature_regressor_out.json')}"
        cfg = compose(
            "regressor_config",
            overrides=[
                source_arg,
                target_arg,
                "mlp.max_iter=2",
                "mlp.hidden_layers=[8,8]",
            ],
        )

        os.chdir(rundir)
        train_regressor.main(cfg)

        # Check model file is correctly saved
        assert Path("model.json").exists()
        with open("model.json") as f:
            model = json.load(f)
        assert "layers" in model
        layer1 = model["layers"][0]
        assert np.array(layer1["weights"]).shape == (cfg.mlp.input_size, 8)
        assert np.array(layer1["biases"]).shape == (8,)
        layer2 = model["layers"][1]
        assert np.array(layer2["weights"]).shape == (8, 8)
        assert np.array(layer2["biases"]).shape == (8,)
        layer3 = model["layers"][2]
        assert np.array(layer3["weights"]).shape == (8, cfg.mlp.output_size)
        assert np.array(layer3["biases"]).shape == (cfg.mlp.output_size,)

        # Check scaler files are correctly saved
        assert Path("source_normalize.json").exists()
        with open("source_normalize.json") as f:
            source_scaler = json.load(f)

        assert source_scaler["cols"] == cfg.mlp.input_size
        assert source_scaler["min"] == 0.0
        assert source_scaler["max"] == 1.0
        assert len(source_scaler["data_min"]) == cfg.mlp.input_size
        assert len(source_scaler["data_max"]) == cfg.mlp.input_size

        assert Path("target_normalize.json").exists()
        with open("target_normalize.json") as f:
            target_scaler = json.load(f)

        assert target_scaler["cols"] == cfg.mlp.output_size
        assert target_scaler["min"] == 0.0
        assert target_scaler["max"] == 1.0
        assert len(target_scaler["data_min"]) == cfg.mlp.output_size
        assert len(target_scaler["data_max"]) == cfg.mlp.output_size
