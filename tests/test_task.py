from flucoma_torch.task import FluidMLPRegressor, FluidMLPClassifier


def test_init_fluid_mlp_regressor():
    FluidMLPRegressor(
        input_size=5,
        hidden_layers=[
            2,
        ],
        output_size=1,
        activation=2,
        output_activation=1,
    )


def test_init_fluid_mlp_classifier():
    FluidMLPClassifier(
        input_size=5,
        hidden_layers=[
            2,
        ],
        output_size=1,
        activation=2,
    )
