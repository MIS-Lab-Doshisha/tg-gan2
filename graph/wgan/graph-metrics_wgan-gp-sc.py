import sys
import math
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from torch.utils.data import DataLoader
from nilearn.connectome import sym_matrix_to_vec
from bct.algorithms.centrality import betweenness_bin
from bct.algorithms.clustering import clustering_coef_bu
from scipy.stats import gaussian_kde

sys.path.append('.')
from utils.load_dataset import NIMHSC
from utils.graph import binarize_matrix, kl_div
from utils.visualizer import plot_connectivity_matrix
from models.gan import Encoder, Decoder


# Calculate and visualize graph metrics of acquired and synthesized matrices
# ------ Matplotlib ------ #
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# ------ Setting ------ #
data_dir = '../../data'
dropout = [0.1]
edge_density = [0.05, 0.10, 0.15, 0.20, 0.25]

# ------ Load NIMH dataset ------ #
transform = transforms.Compose([transforms.ToTensor()])
train_datasets = NIMHSC(participants_csv='../../data/nihtoolbox_disc.csv', target='nihtoolbox', transforms=transform,
                        data_dir=f'{data_dir}/structural_connectivity',
                        atlas='aal116_sift_invnodevol_radius2_count_connectivity')
train_loader = DataLoader(dataset=train_datasets, batch_size=75, shuffle=True)

# ------ Load NIMH dataset ------ #
transform = transforms.Compose([transforms.ToTensor()])
train_datasets = NIMHSC(participants_csv='../../data/nihtoolbox_disc.csv', target='nihtoolbox', transforms=transform,
                        data_dir=f'{data_dir}/structural_connectivity',
                        atlas='aal116_sift_invnodevol_radius2_count_connectivity')
train_loader = DataLoader(dataset=train_datasets, batch_size=75, shuffle=False)

for dr in dropout:

    kl_divergence_between_centrality = []
    kl_divergence_cluster_coefficient = []

    # ------ Load Encoder/Decoder checkpoint ------ #
    checkpoint_encoder_pth = f'../../wgan/data/wgan-gp-sc/dr_{dr}/checkpoint_encoder.pth'
    checkpoint_decoder_pth = f'../../wgan/data/wgan-gp-sc/dr_{dr}/checkpoint_decoder.pth'
    checkpoint_encoder = torch.load(checkpoint_encoder_pth, map_location=torch.device('cpu'))
    checkpoint_decoder = torch.load(checkpoint_decoder_pth, map_location=torch.device('cpu'))

    encoder = Encoder(n_features=64, n_regions=116, dr_rate=dr)
    encoder.load_state_dict(checkpoint_encoder)
    decoder = Decoder(n_features=64, n_regions=116)
    decoder.load_state_dict(checkpoint_decoder)

    encoder.eval()
    decoder.eval()

    # ------ Synthesize connectivity matrix ------ #
    for i, data in enumerate(train_loader):
        # Real matrix and target
        matrix, target = data
        real = matrix.to(dtype=torch.float32)
        target = target.to(dtype=torch.float32)

        # Synthesize connectivity matrix
        z = encoder(real)
        fake = decoder(z)

    real = real.detach().numpy()
    real = real.reshape(real.shape[0], real.shape[-1], real.shape[-1])
    fake = fake.detach().numpy()
    fake = fake.reshape(fake.shape[0], fake.shape[-1], fake.shape[-1])
    print('real: ', real.shape)
    print('fake: ', fake.shape)

    # ------ Edge strength ------ #
    real_mean = np.average(real, axis=0)
    fake_mean = np.average(fake, axis=0)
    real_mean_vector = sym_matrix_to_vec(real_mean, discard_diagonal=True)
    fake_mean_vector = sym_matrix_to_vec(fake_mean, discard_diagonal=True)
    print('real_mean_vector: ', real_mean_vector.shape)
    print('fake_mean_vector: ', fake_mean_vector.shape)

    # Execute Kernel Density Estimation (KDE)
    kde_edge_density_real = gaussian_kde(real_mean_vector)
    kde_edge_density_fake = gaussian_kde(fake_mean_vector)
    x = np.linspace(-2, 20, 1000)

    density_edge_density_real = kde_edge_density_real(x)
    density_edge_density_fake = kde_edge_density_fake(x)

    kl_edge_strength = kl_div(density_edge_density_fake, density_edge_density_real)
    print('kl_edge_strength: ', kl_edge_strength)

    # ------ Graph metrics ------ #
    between_centrality_real_concat = []
    between_centrality_fake_concat = []
    cluster_coefficient_real_concat = []
    cluster_coefficient_fake_concat = []
    for ed in edge_density:
        real_bin = binarize_matrix(real_mean, edge_density=ed)
        fake_bin = binarize_matrix(fake_mean, edge_density=ed)
        print('real_bin: ', real_bin.shape)
        print('fake_bin: ', fake_bin.shape)

        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        ax = plot_connectivity_matrix(fig=fig, ax=ax, matrix=fake_mean, cmap='gist_heat', vmin=0, vmax=10)

        ax = fig.add_subplot(1, 2, 2)
        ax = plot_connectivity_matrix(fig=fig, ax=ax, matrix=fake_bin, cmap='gray', vmin=0, vmax=1)

        fig.savefig(f'../fig/wgan-gp-sc_connectivity_matrix_edge_density_{ed}.pdf', transparent=True)

        # Between centrality
        real_between_centrality = betweenness_bin(real_bin) / ((real_bin.shape[0] - 1) * (real_bin.shape[0] - 2))
        fake_between_centrality = betweenness_bin(fake_bin) / ((fake_bin.shape[0] - 1) * (fake_bin.shape[0] - 2))
        print('real_between_centrality: ', real_between_centrality.shape)
        print('fake_between_centrality: ', fake_between_centrality.shape)

        between_centrality_real_concat.append(real_between_centrality)
        between_centrality_fake_concat.append(fake_between_centrality)

        # Execute Kernel Density Estimation (KDE)
        kde_between_centrality_real = gaussian_kde(real_between_centrality)
        kde_between_centrality_fake = gaussian_kde(fake_between_centrality)
        x = np.linspace(-0.1, 0.35, 1000)

        density_between_centrality_real = kde_between_centrality_real(x)
        density_between_centrality_fake = kde_between_centrality_fake(x)

        kl_between_centrality = kl_div(density_between_centrality_fake, density_between_centrality_real)
        kl_divergence_between_centrality.append(kl_between_centrality)

        # Cluster coefficient
        real_cluster_coefficient = clustering_coef_bu(real_bin)
        fake_cluster_coefficient = clustering_coef_bu(fake_bin)
        print('real_cluster_coefficient: ', real_cluster_coefficient.shape)
        print('fake_cluster_coefficient: ', fake_cluster_coefficient.shape)

        cluster_coefficient_real_concat.append(real_cluster_coefficient)
        cluster_coefficient_fake_concat.append(fake_cluster_coefficient)

        # Execute Kernel Density Estimation (KDE)
        kde_cluster_coefficient_real = gaussian_kde(real_cluster_coefficient)
        kde_cluster_coefficient_fake = gaussian_kde(fake_cluster_coefficient)
        x = np.linspace(-0.5, 1.50, 1000)

        density_cluster_coefficient_real = kde_cluster_coefficient_real(x)
        density_cluster_coefficient_fake = kde_cluster_coefficient_fake(x)

        kl_cluster_coefficient = kl_div(density_cluster_coefficient_fake, density_cluster_coefficient_real)
        kl_divergence_cluster_coefficient.append(kl_cluster_coefficient)

    # violin plot
    color_pallete = ['#2ca25f', '#de2d26']

    # edge strength
    df_edge_strength_real = pd.DataFrame({'edge_strength': real_mean_vector})
    df_edge_strength_fake = pd.DataFrame({'edge_strength': fake_mean_vector})
    df_edge_strength_real_melt = pd.melt(df_edge_strength_real)
    df_edge_strength_real_melt['kind'] = 'real'
    df_edge_strength_fake_melt = pd.melt(df_edge_strength_fake)
    df_edge_strength_fake_melt['kind'] = 'fake'
    df_edge_strength = pd.concat([df_edge_strength_real_melt, df_edge_strength_fake_melt], axis=0)

    fig = plt.figure(figsize=(4, 5))
    ax = fig.add_subplot()
    with sns.color_palette(color_pallete):
        ax = sns.violinplot(x='variable', y='value', data=df_edge_strength, hue='kind', split=True, ax=ax)
    ax.set_xlabel('', fontsize=14)
    ax.set_ylabel('Edge Strength', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    spines = 2.0
    ax.spines["top"].set_linewidth(spines)
    ax.spines["left"].set_linewidth(spines)
    ax.spines["bottom"].set_linewidth(spines)
    ax.spines["right"].set_linewidth(spines)

    fig.savefig('../fig/wgan-gp-sc_edge_strength.pdf', transparent=True)

    # between centrality
    df_between_centrality_real = pd.DataFrame({})
    df_between_centrality_fake = pd.DataFrame({})
    for i, ed in enumerate(edge_density):
        df_between_centrality_real[f'{ed * 100}%'] = between_centrality_real_concat[i]
        df_between_centrality_fake[f'{ed * 100}%'] = between_centrality_fake_concat[i]
    df_between_centrality_real_melt = pd.melt(df_between_centrality_real)
    df_between_centrality_real_melt['kind'] = 'real'
    df_between_centrality_fake_melt = pd.melt(df_between_centrality_fake)
    df_between_centrality_fake_melt['kind'] = 'fake'
    df_between_centrality = pd.concat([df_between_centrality_real_melt, df_between_centrality_fake_melt], axis=0)

    fig = plt.figure()
    ax = fig.add_subplot()
    with sns.color_palette(color_pallete):
        ax = sns.violinplot(x='variable', y='value', data=df_between_centrality, hue='kind', split=True, ax=ax)
    ax.set_xlabel('Edge density', fontsize=14)
    ax.set_ylabel('Between centrality', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    spines = 2.0
    ax.spines["top"].set_linewidth(spines)
    ax.spines["left"].set_linewidth(spines)
    ax.spines["bottom"].set_linewidth(spines)
    ax.spines["right"].set_linewidth(spines)

    fig.savefig('../fig/wgan-gp-sc_between-centrality.pdf', transparent=True)

    # cluster coefficient
    df_cluster_coefficient_real = pd.DataFrame({})
    df_cluster_coefficient_fake = pd.DataFrame({})
    for i, ed in enumerate(edge_density):
        df_cluster_coefficient_real[f'{ed * 100}%'] = cluster_coefficient_real_concat[i]
        df_cluster_coefficient_fake[f'{ed * 100}%'] = cluster_coefficient_fake_concat[i]
    df_cluster_coefficient_real_melt = pd.melt(df_cluster_coefficient_real)
    df_cluster_coefficient_real_melt['kind'] = 'real'
    df_cluster_coefficient_fake_melt = pd.melt(df_cluster_coefficient_fake)
    df_cluster_coefficient_fake_melt['kind'] = 'fake'
    df_cluster_coefficient = pd.concat([df_cluster_coefficient_real_melt, df_cluster_coefficient_fake_melt], axis=0)

    fig = plt.figure()
    ax = fig.add_subplot()
    with sns.color_palette(color_pallete):
        ax = sns.violinplot(x='variable', y='value', data=df_cluster_coefficient, hue='kind', split=True, ax=ax)
    ax.set_xlabel('Edge density', fontsize=14)
    ax.set_ylabel('Cluster coefficient', fontsize=14)
    ax.legend(loc='lower right')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    spines = 2.0
    ax.spines["top"].set_linewidth(spines)
    ax.spines["left"].set_linewidth(spines)
    ax.spines["bottom"].set_linewidth(spines)
    ax.spines["right"].set_linewidth(spines)

    fig.savefig('../fig/wgan-gp-sc_cluster-coefficient.pdf', transparent=True)

    graph_metrics_txt = f'./wgan-gp-sc_graph-metrics_dr_{dr}.txt'
    with open(graph_metrics_txt, mode='w') as f:
        f.write(f'edge strength: {kl_edge_strength}\n')
        for i, ed in enumerate(edge_density):
            f.write(f'edge density: {ed}\n')
            f.write(f'between centrality: {kl_divergence_between_centrality[i]}\n')
            f.write(f'cluster coefficient: {kl_divergence_cluster_coefficient[i]}\n\n')

plt.show()
