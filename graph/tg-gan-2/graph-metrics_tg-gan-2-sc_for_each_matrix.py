import sys
import math
import torch
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from torch.utils.data import DataLoader
from nilearn.connectome import sym_matrix_to_vec
from bct.algorithms.centrality import betweenness_bin
from bct.algorithms.clustering import clustering_coef_bu
from bct.algorithms.efficiency import efficiency_bin
from scipy.stats import ttest_rel
from pingouin import compute_effsize

sys.path.append('.')
from utils.load_dataset import NIMHSC
from utils.graph import binarize_matrix, kl_div
from utils.visualizer import plot_connectivity_matrix
from models.gan import Encoder, Decoder


# ------ Matplotlib ------ #
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# ------ Setting ------ #
data_dir = '../../data'
dropout = 0.1
alpha = 0.6
edge_density = [0.05, 0.10, 0.15, 0.20, 0.25]
# edge_density = [0.05]
color_pallete = ['#2ca25f', '#5D3A9B']

# ------ Load NIMH dataset ------ #
transform = transforms.Compose([transforms.ToTensor()])
train_datasets = NIMHSC(participants_csv='../../data/nihtoolbox_disc.csv', target='nihtoolbox', transforms=transform,
                        data_dir=f'{data_dir}/structural_connectivity',
                        atlas='aal116_sift_invnodevol_radius2_count_connectivity')
train_loader = DataLoader(dataset=train_datasets, batch_size=75, shuffle=False)

# ------ Load Encoder/Decoder checkpoint ------ #
checkpoint_encoder_pth = f'../../tg-gan-2/data/tg-gan-sc/dr_{dropout}_alpha_{alpha}/checkpoint_encoder.pth'
checkpoint_decoder_pth = f'../../tg-gan-2/data/tg-gan-sc/dr_{dropout}_alpha_{alpha}/checkpoint_decoder.pth'
checkpoint_encoder = torch.load(checkpoint_encoder_pth, map_location=torch.device('cpu'))
checkpoint_decoder = torch.load(checkpoint_decoder_pth, map_location=torch.device('cpu'))

encoder = Encoder(n_features=64, n_regions=116, dr_rate=dropout)
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


# ------ Graph metrics ------ #
df_between_centrality_real = pd.DataFrame({})
df_between_centrality_fake = pd.DataFrame({})
df_cluster_coefficient_real = pd.DataFrame({})
df_cluster_coefficient_fake = pd.DataFrame({})
df_local_efficiency_real = pd.DataFrame({})
df_local_efficiency_fake = pd.DataFrame({})
df_global_efficiency_real = pd.DataFrame({})
df_global_efficiency_fake = pd.DataFrame({})
df_modularity_real = pd.DataFrame({})
df_modularity_fake = pd.DataFrame({})

# ------ Cohen's d ------ #
between_centrality_d = []
cluster_coefficient_d = []
local_efficiency_d = []
global_efficiency_d = []
modularity_d = []

for ed in edge_density:
    print('edge density: ', ed)

    # betweenness centrality
    between_centrality_real = []
    between_centrality_fake = []

    # clustering coefficient
    cluster_coefficient_real = []
    cluster_coefficient_fake = []

    # local efficiency
    local_efficiency_real = []
    local_efficiency_fake = []

    # global efficiency
    global_efficiency_real = []
    global_efficiency_fake = []

    # modularity
    modularity_real = []
    modularity_fake = []

    for i in range(len(real)):  # Calculation graph theory metrics in each matrix
        # Matrix binarization
        rm = real[i, :, :]  # real matrix
        fm = fake[i, :, :]  # fake matrix
        rm_bin = binarize_matrix(matrix=rm, edge_density=ed)
        fm_bin = binarize_matrix(matrix=fm, edge_density=ed)
        # print('real: ', rm.shape)
        # print('fake: ', fm.shape)
        # print('real_bin: ', rm_bin.shape)
        # print('fake_bin: ', fm_bin.shape)

        # ------ Plot matrix for supporting information ----- #
        # fig = plt.figure()
        # ax = fig.add_subplot()
        # ax = fig.add_subplot(1, 2, 1)
        # ax = plot_connectivity_matrix(fig=fig, ax=ax, matrix=rm_bin, cmap='gray', vmin=0, vmax=1)
        #
        # ax = fig.add_subplot(1, 2, 2)
        # ax = plot_connectivity_matrix(fig=fig, ax=ax, matrix=fm_bin, cmap='gray', vmin=0, vmax=1)

        # Betweenness centrality
        real_bc = betweenness_bin(rm_bin) / ((rm_bin.shape[0] - 1) * (rm_bin.shape[0] - 2))
        fake_bc = betweenness_bin(fm_bin) / ((fm_bin.shape[0] - 1) * (fm_bin.shape[0] - 2))
        real_bc_mean = np.mean(real_bc)
        fake_bc_mean = np.mean(fake_bc)
        between_centrality_real.append(real_bc_mean)
        between_centrality_fake.append(fake_bc_mean)

        # Clustering coefficient
        real_cc = clustering_coef_bu(rm_bin)
        fake_cc = clustering_coef_bu(fm_bin)
        real_cc_mean = np.mean(real_cc)
        fake_cc_mean = np.mean(fake_cc)
        cluster_coefficient_real.append(real_cc_mean)
        cluster_coefficient_fake.append(fake_cc_mean)

        # Local efficiency
        real_le = efficiency_bin(rm_bin, local=True)
        fake_le = efficiency_bin(fm_bin, local=True)
        real_le_mean = np.mean(real_le)
        fake_le_mean = np.mean(fake_le)
        local_efficiency_real.append(real_le_mean)
        local_efficiency_fake.append(fake_le_mean)

        # Global efficiency
        real_ge = efficiency_bin(rm_bin, local=False)
        fake_ge = efficiency_bin(fm_bin, local=False)
        global_efficiency_real.append(real_ge)
        global_efficiency_fake.append(fake_ge)

        # Modularity
        real_G = nx.from_numpy_array(rm_bin)
        fake_G = nx.from_numpy_array(fm_bin)
        real_md = nx.community.modularity(real_G, nx.community.label_propagation_communities(real_G))
        fake_md = nx.community.modularity(fake_G, nx.community.label_propagation_communities(fake_G))
        modularity_real.append(real_md)
        modularity_fake.append(fake_md)

    # ----- DataFrame ----- #
    # Betweenness centrality
    df_between_centrality_real[f'{ed * 100}%'] = between_centrality_real
    df_between_centrality_fake[f'{ed * 100}%'] = between_centrality_fake

    # Clustering coefficient
    df_cluster_coefficient_real[f'{ed * 100}%'] = cluster_coefficient_real
    df_cluster_coefficient_fake[f'{ed * 100}%'] = cluster_coefficient_fake

    # Local efficiency
    df_local_efficiency_real[f'{ed * 100}%'] = local_efficiency_real
    df_local_efficiency_fake[f'{ed * 100}%'] = local_efficiency_fake

    # Global efficiency
    df_global_efficiency_real[f'{ed * 100}%'] = global_efficiency_real
    df_global_efficiency_fake[f'{ed * 100}%'] = global_efficiency_fake

    # Modularity
    df_modularity_real[f'{ed * 100}%'] = modularity_real
    df_modularity_fake[f'{ed * 100}%'] = modularity_fake

    # ------ Cohen7s d ------ #
    # Betweenness centrality
    d = compute_effsize(between_centrality_real, between_centrality_fake, paired=True, eftype='cohen')
    between_centrality_d.append(d)

    # Clustering coefficient
    d = compute_effsize(cluster_coefficient_real, cluster_coefficient_fake, paired=True, eftype='cohen')
    cluster_coefficient_d.append(d)

    # Local efficiency
    d = compute_effsize(local_efficiency_real, local_efficiency_fake, paired=True, eftype='cohen')
    local_efficiency_d.append(d)

    # Global efficiency
    d = compute_effsize(global_efficiency_real, global_efficiency_fake, paired=True, eftype='cohen')
    global_efficiency_d.append(d)

    # Modularity
    d = compute_effsize(modularity_real, modularity_fake, paired=True, eftype='cohen')
    modularity_d.append(d)

# ------ Graph metrics ------- #
df_graph_metrics_real = [df_between_centrality_real, df_cluster_coefficient_real, df_local_efficiency_real, df_global_efficiency_real, df_modularity_real]
df_graph_metrics_fake = [df_between_centrality_fake, df_cluster_coefficient_fake, df_local_efficiency_fake, df_global_efficiency_fake, df_modularity_fake]

# ------ Cohen's d------ #
df_d_between_centrality = pd.DataFrame({'edge density': edge_density, 't': between_centrality_d})
df_d_between_centrality.to_csv('./between_centrality_d.csv', encoding='utf-8-sig')

df_d_cluster_coefficient = pd.DataFrame({'edge density': edge_density, 't': cluster_coefficient_d})
df_d_cluster_coefficient.to_csv('./cluster_coefficient_d.csv', encoding='utf-8-sig')

df_d_local_efficiency = pd.DataFrame({'edge density': edge_density, 't': local_efficiency_d})
df_d_local_efficiency.to_csv('./local_efficiency_d.csv', encoding='utf-8-sig')

df_d_global_efficiency = pd.DataFrame({'edge density': edge_density, 't': global_efficiency_d})
df_d_global_efficiency.to_csv('./global_efficiency_d.csv', encoding='utf-8-sig')

df_d_modularity = pd.DataFrame({'edge density': edge_density, 't': modularity_d})
df_d_modularity.to_csv('./modularity_d.csv', encoding='utf-8-sig')

# ----- DataFrame manipulation for plotting ----- #
df_graph_metrics = []
for df_gm_r, df_gm_f in zip(df_graph_metrics_real, df_graph_metrics_fake):
    df_gm_r_melt = pd.melt(df_gm_r)
    df_gm_f_melt = pd.melt(df_gm_f)
    df_gm_r_melt['kind'] = 'acquired'
    df_gm_f_melt['kind'] = 'synthesized'
    df_gm = pd.concat([df_gm_r_melt, df_gm_f_melt], axis=0)
    df_graph_metrics.append(df_gm)
    print('df_gm: ', df_gm)

# ----- Plot ------ #
# Betweenness centrality
df_between_centrality = df_graph_metrics[0]
print(df_between_centrality)


# plotting
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot()
with sns.color_palette(color_pallete):
    ax = sns.violinplot(x='variable', y='value', data=df_between_centrality, hue='kind', split=True, ax=ax, inner=None)
ax.set_xlabel('Edge density', fontsize=14)
ax.set_ylabel('Betweeness centrality', fontsize=14)
ax.get_legend().remove()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
fig.savefig(f'../fig/graph_metrics/tg-gan-2-sc_betweenness_centrality.pdf', transparent=True)


# Clustering coefficient
df_cluster_coefficient = df_graph_metrics[1]
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot()
with sns.color_palette(color_pallete):
    ax = sns.violinplot(x='variable', y='value', data=df_cluster_coefficient, hue='kind', split=True, ax=ax, inner=None)
ax.set_xlabel('Edge density', fontsize=14)
ax.set_ylabel('Clustering coefficient', fontsize=14)
ax.get_legend().remove()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
fig.savefig(f'../fig/graph_metrics/tg-gan-2-sc_cluster_coefficient.pdf', transparent=True)

# Local efficiency
df_local_efficiency = df_graph_metrics[2]
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot()
with sns.color_palette(color_pallete):
    ax = sns.violinplot(x='variable', y='value', data=df_local_efficiency, hue='kind', split=True, ax=ax, inner=None)
ax.set_xlabel('Edge density', fontsize=14)
ax.set_ylabel('Local efficiency', fontsize=14)
ax.get_legend().remove()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
fig.savefig(f'../fig/graph_metrics/tg-gan-2-sc_local_efficiency.pdf', transparent=True)

# Global efficiency
df_global_efficiency = df_graph_metrics[3]
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot()
with sns.color_palette(color_pallete):
    ax = sns.violinplot(x='variable', y='value', data=df_global_efficiency, hue='kind', split=True, ax=ax, inner=None)
ax.set_xlabel('Edge density', fontsize=14)
ax.set_ylabel('Global efficiency', fontsize=14)
ax.get_legend().remove()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
fig.savefig(f'../fig/graph_metrics/tg-gan-2-sc_global_efficiency.pdf', transparent=True)

# Modularity
df_modularity = df_graph_metrics[4]
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot()
with sns.color_palette(color_pallete):
    ax = sns.violinplot(x='variable', y='value', data=df_modularity, hue='kind', split=True, ax=ax, inner=None)
ax.set_xlabel('Edge density', fontsize=14)
ax.set_ylabel('Modularity', fontsize=14)
ax.get_legend().remove()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
fig.savefig(f'../fig/graph_metrics/tg-gan-2-sc_modularity.pdf', transparent=True)

# plt.show()
