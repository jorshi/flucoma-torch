"""
Lightning tasks for FluCoMa MLPs
"""

import lightning as L


class FluidMLPRegressor(L.LightngModule):
    """
    A PyTorch Lightning module for training and evaluation of a FluCoMa MLP
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
