from flucoma_torch.model import FluidMLP


def test_():
    model = FluidMLP(
        input_size=10,
        hidden_layers=[20, 30],
        output_size=5,
        activation=1,  # Assuming 1 corresponds to a valid activation function
        output_activation=0,  # Assuming 0 corresponds to a valid output activation
    )

    assert model is not None
