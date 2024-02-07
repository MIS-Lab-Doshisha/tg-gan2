import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layer import E2EBlock, E2NBlock, N2GBlock


class BrainNetCNN(nn.Module):

    def __init__(self, n_regions: int, dr_rate=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dr_rate = dr_rate
        self.e2e_conv1 = E2EBlock(in_channels=1, out_channels=32, n_regions=n_regions, bias=True)
        self.e2e_conv2 = E2EBlock(in_channels=32, out_channels=64, n_regions=n_regions, bias=True)
        self.e2n = E2NBlock(in_channels=64, out_channels=1, n_regions=n_regions, bias=True)
        self.n2g = N2GBlock(in_channels=1, out_channels=256, n_regions=n_regions, bias=True)
        self.dense1 = nn.Linear(in_features=256, out_features=128, bias=True)
        self.dense2 = nn.Linear(in_features=128, out_features=64, bias=True)
        self.dense3 = nn.Linear(in_features=64, out_features=32, bias=True)
        self.dense4 = nn.Linear(in_features=32, out_features=1, bias=True)

    def forward(self, x: torch.Tensor):
        x = F.leaky_relu(self.e2e_conv1(x), negative_slope=0.33)
        x = F.leaky_relu(self.e2e_conv2(x), negative_slope=0.33)
        x = F.leaky_relu(self.e2n(x), negative_slope=0.33)
        x = F.dropout(F.leaky_relu(self.n2g(x), negative_slope=0.33), p=self.dr_rate)
        x = x.view(x.size(0), -1)
        x = F.dropout(F.leaky_relu(self.dense1(x), negative_slope=0.33), p=self.dr_rate)
        x = F.dropout(F.leaky_relu(self.dense2(x), negative_slope=0.33), p=self.dr_rate)
        x = F.dropout(F.leaky_relu(self.dense3(x), negative_slope=0.33), p=self.dr_rate)
        x = F.leaky_relu(self.dense4(x), negative_slope=0.33)
        return x
