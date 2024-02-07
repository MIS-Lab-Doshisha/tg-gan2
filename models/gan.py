import sys
import torch
import torch.nn as nn
from nilearn.connectome import sym_matrix_to_vec

sys.path.append('.')
from models.layer import E2EBlock, E2NBlock, N2GBlock
from models.connectome import vec_to_sym_matrix_torch
from models.loss import RMSELoss


# ------ Critics ------ #
class Critics(nn.Module):
    def __init__(self, n_regions: int, dr_rate=0.5):
        """Critics: Critics of Wasserstein GAN

        :param n_regions: number of regions in adjacency matrix (connectivity matrix)
        :param dr_rate: drop out rate
        """
        super().__init__()
        self.model = nn.Sequential(
            E2EBlock(in_channels=1, out_channels=32, n_regions=n_regions, bias=True),
            nn.LeakyReLU(negative_slope=0.33),
            E2EBlock(in_channels=32, out_channels=64, n_regions=n_regions, bias=True),
            nn.LeakyReLU(negative_slope=0.33),
            E2NBlock(in_channels=64, out_channels=1, n_regions=n_regions),
            nn.LeakyReLU(negative_slope=0.33),
            N2GBlock(in_channels=1, out_channels=256, n_regions=n_regions),
            nn.Dropout(p=dr_rate),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.33),
            nn.Dropout(p=dr_rate),
            nn.Linear(128, 3),
            nn.LeakyReLU(negative_slope=0.33),
            nn.Linear(3, 1),
            nn.LeakyReLU(negative_slope=0.33)
        )

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        return x


# ------ Generator: Encoder ------ #
class Encoder(nn.Module):
    def __init__(self, n_features: int, n_regions: int, dr_rate=0.5):
        """Encoder: Encoder (Generator) of Wasserstein GAN

        :param n_features: number of latent features
        :param n_regions: number of regions in adjacency matrix (connectivity matrix)
        :param dr_rate: drop out rate
        """
        super().__init__()
        self.model = nn.Sequential(
            E2EBlock(in_channels=1, out_channels=32, n_regions=n_regions, bias=True),
            nn.LeakyReLU(negative_slope=0.33),
            E2EBlock(in_channels=32, out_channels=64, n_regions=n_regions, bias=True),
            nn.LeakyReLU(negative_slope=0.33),
            E2NBlock(in_channels=64, out_channels=1, n_regions=n_regions),
            nn.LeakyReLU(negative_slope=0.33),
            N2GBlock(in_channels=1, out_channels=256, n_regions=n_regions),
            nn.Dropout(p=dr_rate),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.33),
            nn.Dropout(p=dr_rate),
            nn.Linear(128, n_features),
            nn.LeakyReLU(negative_slope=0.33)
        )

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        return x


# ------ Generator: Decoder ------ #
class Decoder(nn.Module):
    def __init__(self, n_features: int, n_regions: int, diagonal='zero'):
        """Decoder: Decoder (Generator) of Wasserstein GAN

        :param n_features: number of latent features
        :param n_regions: number of regions in adjacency matrix (connectivity matrix)
        :param: diagonal: fill 0 (one) in diagonal part of adjacency matrix (connectivity matrix) if it is set as zero (one)
        """
        super().__init__()
        self.n_regions = n_regions
        self.diagonal = diagonal
        _n_out = int(n_regions * (n_regions - 1) / 2)

        self.model = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, _n_out),
        )

        self.lastLayer = nn.Sequential(
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor):
        _n_samples = x.shape[0]

        if self.diagonal == 'zero':
            _diagonal = torch.zeros(size=(_n_samples, self.n_regions))
        elif self.diagonal == 'one':
            _diagonal = torch.ones(size=(_n_samples, self.n_regions))

        x = self.model(x)
        x = self.lastLayer(x)
        x = vec_to_sym_matrix_torch(vec=x, diagonal=_diagonal)
        x = x.reshape(_n_samples, 1, self.n_regions, self.n_regions)

        return x


# ------ Trainer ------ #
def gradient(real: torch.Tensor, fake: torch.Tensor, epsilon: torch.Tensor, critics):
    """gradient: Calculate gradient

    :param real: real images
    :param fake: fake images
    :param epsilon: random number epsilon in U[0, 1]
    :param critics: critics
    :return: gradient: gradient
    """
    # Mixed images together
    mixed_images = real * epsilon + fake * (1 - epsilon)

    # Calculate the critics score on the mixed images
    mixed_scores = critics(mixed_images)

    # Take the gradient of the scores with respect to the images
    grad = torch.autograd.grad(
        # Note: You need to  take the gradient of outputs with respect to inputs
        inputs=mixed_images,
        outputs=mixed_scores,
        # These other parameters have to do with how the pytorch autograd engine works
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    return grad


def gradient_penalty(grad: torch.Tensor):
    """gradient_penalty: Calculate gradient penalty

    :param grad: gradient
    :return: penalty: gradient penalty
    """
    # Flatten the gradients so that each row captures one images
    grad = grad.view(len(grad), -1)

    # Calculate the magnitude of every row
    grad_norm = grad.norm(2, dim=-1)

    # Penalize the mean squared distance of the gradient norms from 1
    penalty = torch.mean(torch.square(grad_norm - 1))

    return penalty


def trainer_critics(critics, optimizer, real: torch.Tensor, fake: torch.Tensor, epsilon: torch.Tensor, c_lambda: float):
    """trainer_critics: trainer of critics (calculate critics loss)

    :param critics: critics
    :param optimizer: optimizer of critics
    :param real: real images
    :param fake: fake images
    :param epsilon: random number epsilon in U[0, 1]
    :param c_lambda: gradient penalty parameter
    :return: -total_error: critics error [total]
    :return: -critics_error: critics error [wasserstein loss]
    """
    optimizer.zero_grad()

    # Critics error
    error_real = critics(real).mean()  # error of critics [real images]
    error_fake = critics(fake).mean()  # error of critics [fake images]
    critics_error = - (error_real - error_fake)  # error of critics [Wasserstein loss]

    # Gradient penalty
    grad = gradient(real=real, fake=fake, epsilon=epsilon, critics=critics)
    penalty = gradient_penalty(grad)

    # Critics loss
    total_error = critics_error + c_lambda * penalty

    total_error.backward()
    optimizer.step()

    return -total_error, -critics_error


def trainer_generator(critics, encoder_optimizer, decoder_optimizer,
                      fake, target):

    """trainer_generator: trainer of generator

    :param critics: critics
    :param encoder_optimizer: optimizer of encoder
    :param decoder_optimizer: optimizer of decoder
    :param fake: fake image
    :param target: true target
    :return: error_fake: generator error
    """

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Generator error
    error_fake = -critics(fake).mean()

    error_fake.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return error_fake


def trainer_generator_regressor(critics, regressor, encoder_optimizer, decoder_optimizer, regressor_optimizer,
                                fake, target, alpha):
    """trainer_generator_regressor: trainer of generator and regressor (calculate generator and regressor loss)

    :param critics: critics
    :param regressor: regressor
    :param encoder_optimizer: optimizer of encoder
    :param decoder_optimizer: optimizer of decoder
    :param regressor_optimizer: optimizer of regressor
    :param fake: fake image
    :param target: true target
    :param alpha: rmse loss parameter
    :return: error_total: generator and regressor error
    :return: error_fake: generator error
    :return: error_rmse: regressor error
    """
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    regressor_optimizer.zero_grad()

    # Generator error
    error_fake = -critics(fake).mean()

    # Regressor error
    loss_func = RMSELoss()
    target_pred = regressor(fake).reshape(-1)
    error_mse = loss_func(target, target_pred)

    # Total error
    error_total = error_fake + alpha * error_mse

    error_total.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    regressor_optimizer.step()

    return error_total, error_fake, error_mse
