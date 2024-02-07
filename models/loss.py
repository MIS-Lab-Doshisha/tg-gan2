import torch
import torch.nn as nn


class RMSELoss(nn.Module):
    def __init__(self, *args, **kwargs):
        """RMSELoss: root mean squared error
        """
        super().__init__(*args, **kwargs)
        self.mse = nn.MSELoss()

    def forward(self, y, y_hat):
        return torch.sqrt(self.mse(y, y_hat))
