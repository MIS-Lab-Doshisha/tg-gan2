import sys
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

sys.path.append('.')
from utils.load_dataset import NIMHSC
from models.gan import Encoder, Decoder


# Visualize latent space
# ------ Matplotlib ------ #
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# ------ Setting ------ #
data_dir = '../data'
dropout= [0.1]

# ------ Load NIMH dataset ------ #
transform = transforms.Compose([transforms.ToTensor()])
train_datasets = NIMHSC(participants_csv='../data/nihtoolbox_disc.csv', target='nihtoolbox', transforms=transform,
                        data_dir=f'{data_dir}/structural_connectivity',
                        atlas='aal116_sift_invnodevol_radius2_count_connectivity')
# valid_datasets = NIMHSC(participants_csv='../data/nihtoolbox_disc_valid.csv', target='nihtoolbox',
#                         data_dir=f'{data_dir}/structural_connectivity',
#                         atlas='aal116_sift_invnodevol_radius2_count_connectivity', transforms=transform)
train_loader = DataLoader(dataset=train_datasets, batch_size=75, shuffle=False)
# valid_loader = DataLoader(dataset=valid_datasets, batch_size=25, shuffle=False)

for dr in dropout:
    # ------ Load Encoder/Decoder checkpoint ------ #
    checkpoint_encoder_pth = f'./data/wgan-gp-sc/dr_{dr}/checkpoint_encoder.pth'
    checkpoint_decoder_pth = f'./data/wgan-gp-sc/dr_{dr}/checkpoint_decoder.pth'
    checkpoint_encoder = torch.load(checkpoint_encoder_pth, map_location=torch.device('cpu'))
    checkpoint_decoder = torch.load(checkpoint_decoder_pth, map_location=torch.device('cpu'))

    encoder = Encoder(n_features=64, n_regions=116, dr_rate=dr)
    encoder.load_state_dict(checkpoint_encoder)
    decoder = Decoder(n_features=64, n_regions=116)
    decoder.load_state_dict(checkpoint_decoder)

    encoder.eval()
    decoder.eval()

    # ------ Synthesize connectivity matrix [Discovery] ------ #
    for i, data in enumerate(train_loader):
        # Real matrix and target
        matrix, target = data
        real = matrix.to(dtype=torch.float32)
        target = target.to(dtype=torch.float32)

        # Synthesize connectivity matrix
        z = encoder(real)
        fake = decoder(z)

    z = z.detach().numpy()
    real = real.detach().numpy()
    real = real.reshape(real.shape[0], real.shape[-1], real.shape[-1])
    fake = fake.detach().numpy()
    fake = fake.reshape(fake.shape[0], fake.shape[-1], fake.shape[-1])
    target = target.detach().numpy()
    print('real: ', real.shape)
    print('fake: ', fake.shape)

    # Principal Component Analysis
    pca = PCA()
    z_new = pca.fit_transform(X=z)
    evr = pca.explained_variance_ratio_ * 100

    # Plotting [Latent space: PCA]
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot()
    mappable = ax.scatter(z_new[:, 0], z_new[:, 1], c=target, edgecolors='#000000', cmap='OrRd', linewidth=1.0, s=50)
    ax.set_xlabel(f'PC 1 [{evr[0]:.2f}%]', fontsize=18)
    ax.set_ylabel(f'PC 2 [{evr[1]:.2f}%]', fontsize=18)
    # ax.set_title(f'[Training] Dropout: {dr}', fontsize=20)
    ax.grid(linestyle='dotted', linewidth=1.5)

    ax.spines["top"].set_linewidth(2.0)
    ax.spines["bottom"].set_linewidth(2.0)
    ax.spines["right"].set_linewidth(2.0)
    ax.spines["left"].set_linewidth(2.0)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    cbar = fig.colorbar(mappable, ax=ax)
    cbar.ax.tick_params(labelsize=15)

    fig.savefig(f'./fig/wgan-gp-sc/training_latent_space_pca_dr_{dr}.pdf', transparent=True)
    fig.savefig(f'./fig/wgan-gp-sc/training_latent_space_pca_dr_{dr}.png', dpi=700)

    # Plotting [Latent space and Objective function: PCA]
    rval, pval = pearsonr(z_new[:, 0], target)
    print(f'[Training] r: {rval:.2f}, p: {pval:.2f}')

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot()
    mappable = ax.scatter(z_new[:, 0], target, c='#DC3220', edgecolors='#000000', linewidth=1.0, s=50)
    sns.regplot(x=z_new[:, 0], y=target, scatter=False, fit_reg=True, ci=0, ax=ax, line_kws={'color': '#DC3220', 'linewidth': 3.0})
    sns.regplot(x=z_new[:, 0], y=target, scatter=False, ax=ax, line_kws={'color': '#808080', 'linewidth': 0.0})
    ax.set_xlabel(f'PC 1 [{evr[0]:.2f}%]', fontsize=18)
    ax.set_ylabel(f'Fluid intelligence', fontsize=18)
    # ax.set_title(f'[Training] Dropout: {dr}, alpha: {a}', fontsize=20)
    ax.grid(linestyle='dotted', linewidth=1.5)

    ax.spines["top"].set_linewidth(2.0)
    ax.spines["bottom"].set_linewidth(2.0)
    ax.spines["right"].set_linewidth(2.0)
    ax.spines["left"].set_linewidth(2.0)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    fig.savefig(f'./fig/wgan-gp-sc/training_latent_score_corr_pca_dr_{dr}.pdf', transparent=True)
    fig.savefig(f'./fig/wgan-gp-sc/training_latent_score_corr_pca_dr_{dr}.png', dpi=700)

plt.show()
