import torch
import torch.nn as nn


class E2EBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_regions: int, bias=False, *args, **kwargs):
        """E2EBlock: Edge-to-Edge Convolution

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param n_regions: number of brain regions
        :param bias: bias
        """

        super().__init__(*args, **kwargs)
        self.n_regions = n_regions
        self.cnn1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, n_regions), bias=bias)
        self.cnn2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(n_regions, 1), bias=bias)

    def forward(self, x: torch.Tensor):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a] * self.n_regions, 3) + torch.cat([b] * self.n_regions, 2)


class E2NBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_regions: int, bias=False, *args, **kwargs):
        """E2NBlock: Edge-to-Node Convolution

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param n_regions: number of brain regions
        :param bias: bias
        """
        super().__init__(*args, **kwargs)
        self.cnn = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, n_regions), bias=bias)

    def forward(self, x: torch.Tensor):
        return self.cnn(x)


class N2GBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_regions: int, bias=False, *args, **kwargs):
        """N2GBlock: Node-to-Graph Convolution

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param n_regions: number of brain regions
        :param bias: bias
        """

        super().__init__(*args, **kwargs)
        self.cnn = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(n_regions, 1), bias=bias)

    def forward(self, x: torch.Tensor):
        return self.cnn(x)
