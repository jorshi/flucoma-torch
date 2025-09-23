# Integration test for training a regressor model
import os
from pathlib import Path
import json

from hydra import initialize_config_module, compose
import pytest

from flucoma_torch import optimize_classifier

test_data = Path(__file__).parent.joinpath("data")


@pytest.fixture
def rundir(tmp_path):
    current_dir = os.getcwd()
    yield tmp_path
    os.chdir(current_dir)


def test_optimize_classifier(rundir):
    with initialize_config_module(
        version_base=None, config_module="flucoma_torch.config"
    ):
        source_arg = f"source={test_data.joinpath('mlpc_help_data.json')}"
        target_arg = f"target={test_data.joinpath('mlpc_help_labels.json')}"
        cfg = compose(
            "optimize_classifier_config",
            overrides=[
                source_arg,
                target_arg,
                "mlp.max_iter=2",
                "n_trials=2",
            ],
        )

        os.chdir(rundir)
        optimize_classifier.main(cfg)

        # Check files are present and look correct
        assert Path("source_normalize.json").exists()
        with open("source_normalize.json") as f:
            source_scaler = json.load(f)

        assert source_scaler["cols"] == cfg.mlp.input_size
        assert source_scaler["min"] == 0.0
        assert source_scaler["max"] == 1.0
        assert len(source_scaler["data_min"]) == cfg.mlp.input_size
        assert len(source_scaler["data_max"]) == cfg.mlp.input_size

        assert Path("best_model.json").exists()
        with open("best_model.json") as f:
            best_model = json.load(f)

        assert "labels" in best_model
        assert len(best_model["labels"]["labels"]) == cfg.mlp.output_size
        assert "mlp" in best_model

        assert Path("best_hyperparameters.json").exists()
