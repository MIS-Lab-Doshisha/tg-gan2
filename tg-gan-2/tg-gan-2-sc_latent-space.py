import sys
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
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
dropout = [0.1]
alphas = [0.6]

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
    for a in alphas:
        # ------ Load Encoder/Decoder checkpoint ------ #
        checkpoint_encoder_pth = f'./data/tg-gan-sc/dr_{dr}_alpha_{a}/checkpoint_encoder.pth'
        checkpoint_decoder_pth = f'./data/tg-gan-sc/dr_{dr}_alpha_{a}/checkpoint_decoder.pth'
        checkpoint_encoder = torch.load(checkpoint_encoder_pth, map_location=torch.device('cpu'))
        checkpoint_decoder = torch.load(checkpoint_decoder_pth, map_location=torch.device('cpu'))

        encoder = Encoder(n_features=64, n_regions=116, dr_rate=dr)
        encoder.load_state_dict(checkpoint_encoder)
        decoder = Decoder(n_features=64, n_regions=116)
        decoder.load_state_dict(checkpoint_decoder)

        encoder.eval()
        decoder.eval()

        # ------ Synthesize connectivity matrix [Discovery dataset] ------ #
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

        # # t-SNE
        # tSNE = TSNE(n_components=2, perplexity=5, random_state=0)
        # z_tsne = tSNE.fit_transform(X=z)

        # Plotting [Latent space: PCA]
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot()
        mappable = ax.scatter(z_new[:, 0], z_new[:, 1], c=target, edgecolors='#000000', cmap='Purples', linewidth=1.0, s=50)
        ax.set_xlabel(f'PC 1 [{evr[0]:.2f}%]', fontsize=18)
        ax.set_ylabel(f'PC 2 [{evr[1]:.2f}%]', fontsize=18)
        # ax.set_title(f'[Training] Dropout: {dr}, alpha: {a}', fontsize=20)
        ax.grid(linestyle='dotted', linewidth=1.5)

        ax.spines["top"].set_linewidth(2.0)
        ax.spines["bottom"].set_linewidth(2.0)
        ax.spines["right"].set_linewidth(2.0)
        ax.spines["left"].set_linewidth(2.0)

        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        cbar = fig.colorbar(mappable, ax=ax)
        cbar.ax.tick_params(labelsize=15)

        fig.savefig(f'./fig/tg-gan-2-sc/training_latent_space_pca_dr_{dr}_alpha_{a}.pdf', transparent=True)
        fig.savefig(f'./fig/tg-gan-2-sc/training_latent_space_pca_dr_{dr}_alpha_{a}.png', dpi=700)

        # Plotting [Latent space and Objective function: PCA]
        rval, pval = pearsonr(z_new[:, 0], target)
        print(f'[Training] r: {rval:.2f}, p: {pval:.2f}')

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot()
        mappable = ax.scatter(z_new[:, 0], target, c='#5D3A9B', edgecolors='#000000', linewidth=1.0, s=50)
        sns.regplot(x=z_new[:, 0], y=target, scatter=False, fit_reg=True, ax=ax, ci=0, line_kws={'color': '#5D3A9B', 'linewidth': 3.0})
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

        fig.savefig(f'./fig/tg-gan-2-sc/training_latent_score_corr_pca_dr_{dr}_alpha_{a}.pdf', transparent=True)
        fig.savefig(f'./fig/tg-gan-2-sc/training_latent_score_corr_pca_dr_{dr}_alpha_{a}.png', dpi=700)

        # # Plotting [Latent space: t-SNE]
        # fig = plt.figure(figsize=(10, 7))
        # ax = fig.add_subplot()
        # mappable = ax.scatter(z_tsne[:, 0], z_tsne[:, 1], c=target, edgecolors='#000000', cmap='Purples', linewidth=1.0, s=50)
        # ax.set_xlabel('t-SNE 1', fontsize=18)
        # ax.set_ylabel('t-SNE 2', fontsize=18)
        # # ax.set_title(f'[Training] Dropout: {dr}, alpha: {a}', fontsize=20)
        # ax.grid(linestyle='dotted', linewidth=1.5)

        # ax.spines["top"].set_linewidth(2.0)
        # ax.spines["bottom"].set_linewidth(2.0)
        # ax.spines["right"].set_linewidth(2.0)
        # ax.spines["left"].set_linewidth(2.0)
        #
        # plt.xticks(fontsize=15)
        # plt.yticks(fontsize=15)

        # cbar = fig.colorbar(mappable, ax=ax)
        # cbar.ax.tick_params(labelsize=15)
        #
        # fig.savefig(f'./fig/tg-gan-2-sc/training_latent_space_tSNE_dr_{dr}_alpha_{a}.pdf', transparent=True)

        # # ------ Synthesize connectivity matrix [Validation] ------ #
        # for i, data in enumerate(valid_loader):
        #     # Real matrix and target
        #     matrix, target = data
        #     real = matrix.to(dtype=torch.float32)
        #     target = target.to(dtype=torch.float32)
        #
        #     # Synthesize connectivity matrix
        #     z = encoder(real)
        #     fake = decoder(z)
        #
        # z = z.detach().numpy()
        # real = real.detach().numpy()
        # real = real.reshape(real.shape[0], real.shape[-1], real.shape[-1])
        # fake = fake.detach().numpy()
        # fake = fake.reshape(fake.shape[0], fake.shape[-1], fake.shape[-1])
        # target = target.detach().numpy()
        # print('real: ', real.shape)
        # print('fake: ', fake.shape)
        #
        # # Principal Component Analysis
        # pca = PCA()
        # z_new = pca.fit_transform(X=z)
        # # z_new = pca.transform(X=z)
        # evr = pca.explained_variance_ratio_ * 100
        #
        # # t-SNE
        # # tSNE = TSNE(n_components=2, perplexity=5, random_state=0)
        # # z_tsne = tSNE.fit_transform(X=z)
        #
        # # Plotting
        # fig = plt.figure(figsize=(10, 7))
        # ax = fig.add_subplot()
        # mappable = ax.scatter(z_new[:, 0], z_new[:, 1], c=target, edgecolors='#000000', cmap='Purples', linewidth=1.0, s=50)
        # ax.set_xlabel(f'PC 1 [{evr[0]:.2f}%]', fontsize=18)
        # ax.set_ylabel(f'PC 2 [{evr[1]:.2f}%]', fontsize=18)
        # # ax.set_title(f'[Validation] Dropout: {dr}, alpha: {a}', fontsize=20)
        # ax.grid(linestyle='dotted', linewidth=1.5)
        #
        # ax.spines["top"].set_linewidth(2.0)
        # ax.spines["bottom"].set_linewidth(2.0)
        # ax.spines["right"].set_linewidth(2.0)
        # ax.spines["left"].set_linewidth(2.0)
        #
        # plt.xticks(fontsize=15)
        # plt.yticks(fontsize=15)
        #
        # cbar = fig.colorbar(mappable, ax=ax)
        # cbar.ax.tick_params(labelsize=15)
        #
        # fig.savefig(f'./fig/tg-gan-2-sc/validation_latent_space_pca_dr_{dr}_alpha_{a}.pdf', transparent=True)
        # fig.savefig(f'./fig/tg-gan-2-sc/validation_latent_space_pca_dr_{dr}_alpha_{a}.png', dpi=700)
        #
        # # Plotting [Latent space and Objective function: PCA]
        # rval, pval = pearsonr(z_new[:, 0], target)
        # print(f'[Validation] r: {rval:.2f}, p: {pval:.2f}')
        #
        # fig = plt.figure(figsize=(10, 7))
        # ax = fig.add_subplot()
        # mappable = ax.scatter(z_new[:, 0], target, c='#5D3A9B', edgecolors='#000000', linewidth=1.0, s=50)
        # sns.regplot(x=z_new[:, 0], y=target, scatter=False, fit_reg=True, ci=0, ax=ax, line_kws={'color': '#5D3A9B', 'linewidth': 3.0})
        # sns.regplot(x=z_new[:, 0], y=target, scatter=False, ax=ax, line_kws={'color': '#808080', 'linewidth': 0.0})
        # ax.set_xlabel(f'PC 1 [{evr[0]:.2f}%]', fontsize=18)
        # ax.set_ylabel(f'Fluid intelligence', fontsize=18)
        # ax.grid(linestyle='dotted', linewidth=1.5)
        #
        # ax.spines["top"].set_linewidth(2.0)
        # ax.spines["bottom"].set_linewidth(2.0)
        # ax.spines["right"].set_linewidth(2.0)
        # ax.spines["left"].set_linewidth(2.0)
        #
        # plt.xticks(fontsize=15)
        # plt.yticks(fontsize=15)
        #
        # fig.savefig(f'./fig/tg-gan-2-sc/validation_latent_score_corr_pca_dr_{dr}_alpha_{a}.pdf', transparent=True)
        # fig.savefig(f'./fig/tg-gan-2-sc/validation_latent_score_corr_pca_dr_{dr}_alpha_{a}.png', dpi=700)

        # # Plotting [Latent space: t-SNE]
        # fig = plt.figure(figsize=(10, 7))
        # ax = fig.add_subplot()
        # mappable = ax.scatter(z_tsne[:, 0], z_tsne[:, 1], c=target, edgecolors='#000000', cmap='Purples', linewidth=1.0, s=50)
        # ax.set_xlabel('t-SNE 1', fontsize=18)
        # ax.set_ylabel('t-SNE 2', fontsize=18)
        # # ax.set_title(f'[Validation] Dropout: {dr}, alpha: {a}', fontsize=20)
        # ax.grid(linestyle='dotted', linewidth=1.5)
        #
        # ax.spines["top"].set_linewidth(2.0)
        # ax.spines["bottom"].set_linewidth(2.0)
        # ax.spines["right"].set_linewidth(2.0)
        # ax.spines["left"].set_linewidth(2.0)
        #
        # plt.xticks(fontsize=15)
        # plt.yticks(fontsize=15)
        #
        # cbar = fig.colorbar(mappable, ax=ax)
        # cbar.ax.tick_params(labelsize=15)

        # fig.savefig(f'./fig/tg-gan-2-sc/validation_latent_space_tSNE_dr_{dr}_alpha_{a}.pdf', transparent=True)

plt.show()
