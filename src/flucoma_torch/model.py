"""
FluidMLP model
"""

import json
import math
from typing import Dict, List

import torch

# Define the activation functions used in the FluCoMa MLP
FLUID_ACTIVATIONS = {
    0: torch.nn.Identity,
    1: torch.nn.Sigmoid,
    2: torch.nn.ReLU,
    3: torch.nn.Tanh,
}


def get_activation(activation: int):
    return FLUID_ACTIVATIONS[activation]()


class FluidMLP(torch.nn.Module):
    """
    Regressor based on the FluCoMa MLP architecture.
    """

    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int],
        output_size: int,
        activation: int,
        output_activation: int,
    ):
        super().__init__()
        layers = []
        for i, h in enumerate(hidden_layers):
            layers.append(
                FluidLinear(input_size if i == 0 else hidden_layers[i - 1], h)
            )
            layers.append(get_activation(activation))
        layers.append(FluidLinear(hidden_layers[-1], output_size))
        layers.append(get_activation(output_activation))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def get_as_dict(self) -> Dict:
        """
        Get the model parameters as a dictionary.
        """
        layers = []
        for i, layer in enumerate(self.model.modules()):
            if isinstance(layer, torch.nn.Linear):
                layers.append(
                    {
                        "biases": layer.bias.data.tolist(),
                        "cols": layer.out_features,
                        "rows": layer.in_features,
                        "weights": layer.weight.data.T.tolist(),
                    }
                )

            if type(layer) in FLUID_ACTIVATIONS.values():
                activation = list(FLUID_ACTIVATIONS.keys())[
                    list(FLUID_ACTIVATIONS.values()).index(type(layer))
                ]
                layers[-1]["activation"] = activation

        return {"layers": layers}

    def save(self, path: str):
        """
        Save the model parameters to a Fluid dictionary format.
        """
        fluid_dict = self.get_as_dict()
        with open(path, "w") as f:
            json.dump(fluid_dict, f, indent=4)


class FluidLinear(torch.nn.Module):
    """Linear layer with custom initialization based on the FluCoMa NNLayer."""

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

        # Initialize weights
        dev = math.sqrt(6 / (input_size + output_size))
        self.linear.weight.data = (
            torch.rand_like(self.linear.weight.data) * 2.0 - 1.0
        ) * dev
        self.linear.bias.data.fill_(0.0)

    def forward(self, x):
        return self.linear(x)


def get_layer(layer: Dict, init: bool = True):
    activation = layer["activation"]
    output_dims = layer["cols"]
    input_dims = layer["rows"]
    weights = torch.tensor(layer["weights"])
    bias = torch.tensor(layer["biases"])

    activation = get_activation(activation)
    linear = torch.nn.Linear(input_dims, output_dims)
    if init:
        linear.weight.data = weights.T
        linear.bias.data = bias
    else:
        dev = math.sqrt(6.0 / (input_dims + output_dims))
        linear.weight.data = (torch.rand_like(weights.T) * 2.0 - 1.0) * dev
        linear.bias.data = torch.zeros_like(bias)

    return linear, activation


def regressor_from_dict(regressor_dict: Dict):
    model = []
    for layer in regressor_dict["layers"]:
        linear, activation = get_layer(layer)
        model.append(linear)
        model.append(activation)

    return torch.nn.Sequential(*model)
