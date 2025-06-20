"""
Lightning tasks for FluCoMa MLPs
"""

from typing import List

import lightning as L
import torch

from flucoma_torch.model import FluidMLP


class FluidMLPRegressor(L.LightningModule):
    """
    A PyTorch Lightning module for training and evaluation of a FluCoMa MLP
    """

    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int],
        output_size: int,
        activation: int,
        output_activation: int,
        learn_rate: float = 1e-3,
        max_iter: int = 1000,
        validation: float = 0.2,
        batch_size: int = 32,
        momentum: float = 0.9,
    ):
        super().__init__()
        self.model = FluidMLP(
            input_size=input_size,
            hidden_layers=hidden_layers,
            output_size=output_size,
            activation=activation,
            output_activation=output_activation,
        )
        self.learn_rate = learn_rate
        self.max_iter = max_iter
        self.validation = validation
        self.batch_size = batch_size
        self.momentum = momentum
        self.loss_function = torch.nn.MSELoss()

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.
        """
        x, y = batch

        # Forward pass
        y_hat = self.model(x)

        # Compute loss
        loss = self.loss_function(y_hat, y)

        # Log the loss
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step.
        """
        x, y = batch

        # Forward pass
        y_hat = self.model(x)

        # Compute loss
        loss = self.loss_function(y_hat, y)

        # Log the validation loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer for training.
        """
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learn_rate,
            momentum=self.momentum,
        )
        return optimizer


class FluidMLPClassifier(FluidMLPRegressor):
    """
    A PyTorch Lightning module for training and evaluation of a FluCoMa MLP Classifier.
    This is almost identical to the regressor, but the output activation is always
    a sigmoid.
    """

    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int],
        output_size: int,
        activation: int,
        learn_rate: float = 1e-3,
        max_iter: int = 1000,
        validation: float = 0.2,
        batch_size: int = 32,
        momentum: float = 0.9,
    ):
        super().__init__(
            input_size=input_size,
            hidden_layers=hidden_layers,
            output_size=output_size,
            activation=activation,
            output_activation=1,
            learn_rate=learn_rate,
            max_iter=max_iter,
            validation=validation,
            batch_size=batch_size,
            momentum=momentum,
        )
