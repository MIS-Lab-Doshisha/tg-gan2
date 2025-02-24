import os
import sys
import pickle
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

sys.path.append('.')
from utils.load_dataset import HCPSC
from utils.visualizer import plot_connectivity_matrix
from models.gan import Encoder, Decoder, Critics, trainer_critics, trainer_generator

# ------ Setting ------ #
# GPU setting
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('device: ', device)

data_dir = '../data_hcp'
n_epochs = 2000
n_critics = 1
dropout_rate = [0.5, 0.4, 0.3, 0.2, 0.1]

# ------ Load NIMH datasets ------- #
transform = transforms.Compose([transforms.ToTensor()])
train_datasets = HCPSC(participants_csv='../data_hcp/hcp-cognition/hcp_fluid-intelligence/hcp_fluid-intelligence_gan_train.csv', target='fluid intelligence', transforms=transform,
                       data_dir='../data_hcp/structural_connectivity_360')
valid_datasets = HCPSC(participants_csv='../data_hcp/hcp-cognition/hcp_fluid-intelligence/hcp_fluid-intelligence_gan_valid.csv', target='fluid intelligence', transforms=transform,
                       data_dir='../data_hcp/structural_connectivity_360')

train_loader = DataLoader(dataset=train_datasets, batch_size=5, shuffle=True)
valid_loader = DataLoader(dataset=valid_datasets, batch_size=1, shuffle=False)

# ------- Wasserstein GAN ------ #
for dr in dropout_rate:
    print('dropout: ', dr)
    # ------ Setting ------- #
    # result
    result_dir = f'./data_hcp/wgan-gp-sc_hcp_fluid-intelligence/dr_{dr}'
    os.makedirs(result_dir, exist_ok=True)

    # ------ GAN ------ #
    # Initialize generator and critics
    encoder = Encoder(n_features=64, n_regions=360, dr_rate=dr)
    decoder = Decoder(n_features=64, n_regions=360)
    critics = Critics(n_regions=360, dr_rate=dr)
    encoder.to(device=device)
    decoder.to(device=device)
    critics.to(device=device)

    # Optimizer
    alpha = 0.0001
    encoder_optimizer = optim.Adam(params=encoder.parameters(), lr=alpha, betas=(0.9, 0.999))
    decoder_optimizer = optim.Adam(params=decoder.parameters(), lr=alpha, betas=(0.9, 0.999))
    critics_optimizer = optim.Adam(params=critics.parameters(), lr=alpha, betas=(0.9, 0.999))

    # Test matrix.shape
    test_matrix = [matrix for matrix, _ in valid_loader]
    test_matrix = torch.stack(test_matrix, dim=0).to(dtype=torch.float32, device=device)
    test_matrix = test_matrix.reshape((len(test_matrix), 1, 360, 360))
    test_target = [target for _, target in valid_loader]
    test_target = torch.stack(test_target, dim=0).to(dtype=torch.float32, device=device)
    test_target = test_target.reshape((len(test_target)))
    print('test_matrix: ', test_matrix.shape)
    print('test_target: ', test_target.shape)

    # Train generator and critics
    # Initialize loss array [training]
    generator_loss = []
    wasserstein_loss = []
    critics_loss = []

    # Training
    encoder.train()
    decoder.train()
    critics.train()
    for epoch in range(n_epochs):
        print('epoch: ', epoch)
        g_error, w_error, c_error = 0.0, 0.0, 0.0
        error_valid = 0.0

        iCount = 0  # Counter for generator training
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            # Real matrix and target
            matrix, target = data
            real = matrix.to(device=device).to(dtype=torch.float32)
            target = target.to(device=device).to(dtype=torch.float32)

            # ------ Train critics ------ #
            z = encoder(real)
            fake = decoder(z)

            # Update critics
            epsilon = torch.randn(len(real), 1, 1, 1, device=device, requires_grad=True)
            ec, ew = trainer_critics(critics=critics, optimizer=critics_optimizer, real=real, fake=fake,
                                     epsilon=epsilon, c_lambda=10)
            c_error += ec.item()  # critics loss
            w_error += ew.item()  # wasserstein loss

            if (i+1) % n_critics == 0:
                iCount += 1
                # ------ Train generator ------ #
                z = encoder(real)
                fake = decoder(z)

                # Update generator
                eg = trainer_generator(critics=critics, encoder_optimizer=encoder_optimizer,
                                       decoder_optimizer=decoder_optimizer, fake=fake, target=target)
                g_error += eg.item()  # generator loss

        # Calculate average loss
        w_error = w_error / len(train_loader)
        c_error = c_error / len(train_loader)
        g_error = g_error / iCount
        print('generator: ', g_error)
        print('critics: ', c_error)
        print('wasserstein: ', w_error)

        wasserstein_loss.append(w_error)
        critics_loss.append(c_error)
        generator_loss.append(g_error)

        # ------ Save ------ #
        # # Synthesized connectivity matrix
        # z = encoder(test_matrix)
        # fake = decoder(z)
        # fake = fake.cpu().detach().numpy()
        # fake = fake.reshape(len(fake), 360, 360)
        #
        # fig = plt.figure(figsize=(12, 8))
        # fig.subplots_adjust(wspace=0.3, hspace=0.3, top=0.9, bottom=0.1, left=0.1, right=0.9)
        # for i in range(12):
        #     ax = fig.add_subplot(3, 4, i + 1)
        #     ax = plot_connectivity_matrix(fig=fig, ax=ax, matrix=fake[i], cmap='gist_heat',
        #                                   vmin=0, vmax=15)
        #     ax.set_title(f'Epoch: {epoch + 1}, Sample: {i + 1}, Score: {test_target[i]}', fontsize=9)
        # fig.savefig(f'{result_dir}/structural_connectivity_{epoch + 1}.pdf', transparent=True)
        # plt.clf()
        # plt.close(fig=fig)

        # Loss
        loss = {'generator': generator_loss,
                'wasserstein': wasserstein_loss,
                'critics': critics_loss}

        with open(f'{result_dir}/loss_epoch_{epoch+1}.pkl', mode='wb') as f:
            pickle.dump(obj=loss, file=f)

        if (epoch + 1) % 500 == 0:
            torch.save(encoder.state_dict(), f'{result_dir}/checkpoint_encoder.pth')
            torch.save(decoder.state_dict(), f'{result_dir}/checkpoint_decoder.pth')
            torch.save(critics.state_dict(), f'{result_dir}/checkpoint_critics.pth')
