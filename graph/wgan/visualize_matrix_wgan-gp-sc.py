import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader

import networkx as nx

from nilearn import datasets
from nilearn import plotting
from nilearn.connectome import sym_matrix_to_vec
from nichord.coord_labeler import get_idx_to_label
from nichord.convert import convert_matrix
from nichord.chord import plot_chord

sys.path.append('.')
from utils.load_dataset import NIMHSC
from utils.graph import binarize_matrix
from utils.visualizer import plot_connectivity_matrix
from models.gan import Encoder, Decoder


# Visualize acquired and synthesized connectome in yeo's network order
# ------ Matplotlib ------ #
plt.rcParams['font.family'] = 'Arial'

# ------ Setting ------ #
data_dir = '../../data'
dropout = [0.1]
edge_density = [0.25]

DAN = '#0075C2'
DMN = '#EE7800'
FPCN = '#00A960'
Limbic = '#E8383D'
SM = '#7F1184'
VAN = '#E95388'
Visual = '#8DA0b6'
Uncertain = '#BB5535'

# ------ Load NIMH dataset ------ #
transform = transforms.Compose([transforms.ToTensor()])
train_datasets = NIMHSC(participants_csv='../../data/nihtoolbox_disc.csv', target='nihtoolbox', transforms=transform,
                        data_dir=f'{data_dir}/structural_connectivity',
                        atlas='aal116_sift_invnodevol_radius2_count_connectivity')
train_loader = DataLoader(dataset=train_datasets, batch_size=75, shuffle=False)

for dr in dropout:

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

    aal = datasets.fetch_atlas_aal(data_dir='../data')
    print('aal[maps]: ', aal['maps'])
    print('aal[labels]: ', aal['labels'])

    coordinates, labels = plotting.find_parcellation_cut_coords(labels_img=aal['maps'], return_label_names=True)
    print('coordinates: ', coordinates.shape)
    print('labels: ', labels)

    # ------ Connectome ------ #
    real_mean = np.average(real, axis=0)
    fake_mean = np.average(fake, axis=0)
    real_mean_vector = sym_matrix_to_vec(real_mean, discard_diagonal=True)
    fake_mean_vector = sym_matrix_to_vec(fake_mean, discard_diagonal=True)
    vmax_real = np.percentile(real_mean_vector, 95)
    vmax_fake = np.percentile(fake_mean_vector, 95)

    # network alignment
    network_idx = get_idx_to_label(coords=coordinates)
    print(network_idx)

    dan_node_l, dan_node_r, dmn_node_l, dmn_node_r, fpcn_node_l, fpcn_node_r, limbic_node_l, limbic_node_r, sm_node_l, sm_node_r, van_node_l, van_node_r, visual_node_l, visual_node_r, uncertain_node = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    dan_node_l_idx, dmn_node_l_idx, fpcn_node_l_idx, limbic_node_l_idx, sm_node_l_idx, van_node_l_idx, visual_node_l_idx, uncertain_node_idx = [], [], [], [], [], [], [], []
    dan_node_r_idx, dmn_node_r_idx, fpcn_node_r_idx, limbic_node_r_idx, sm_node_r_idx, van_node_r_idx, visual_node_r_idx = [], [], [], [], [], [], []
    for node, network in network_idx.items():
        label = aal['labels'][node]
        if network == 'DAN':
            if label[-1] == 'L':
                dan_node_l.append(node)
                dan_node_l_idx.append(0)
            if label[-1] == 'R':
                dan_node_r.append(node)
                dan_node_r_idx.append(0)
        elif network == 'DMN':
            if label[-1] == 'L':
                dmn_node_l.append(node)
                dmn_node_l_idx.append(1)
            if label[-1] == 'R':
                dmn_node_r.append(node)
                dmn_node_r_idx.append(1)
        elif network == 'FPCN':
            if label[-1] == 'L':
                fpcn_node_l.append(node)
                fpcn_node_l_idx.append(2)
            if label[-1] == 'R':
                fpcn_node_r.append(node)
                fpcn_node_r_idx.append(2)
        elif network == 'Limbic':
            if label[-1] == 'L':
                limbic_node_l.append(node)
                limbic_node_l_idx.append(3)
            if label[-1] == 'R':
                limbic_node_r.append(node)
                limbic_node_r_idx.append(3)
        elif network == 'SM':
            if label[-1] == 'L':
                sm_node_l.append(node)
                sm_node_l_idx.append(4)
            if label[-1] == 'R':
                sm_node_r.append(node)
                sm_node_r_idx.append(4)
        elif network == 'VAN':
            if label[-1] == 'L':
                van_node_l.append(node)
                van_node_l_idx.append(5)
            if label[-1] == 'R':
                van_node_r.append(node)
                van_node_r_idx.append(5)
        elif network == 'Visual':
            if label[-1] == 'L':
                visual_node_l.append(node)
                visual_node_l_idx.append(6)
            if label[-1] == 'R':
                visual_node_r.append(node)
                visual_node_r_idx.append(6)
        elif network == 'Uncertain':
            uncertain_node_idx.append(7)
            uncertain_node.append(node)

    network_node = dan_node_l + dmn_node_l + fpcn_node_l + limbic_node_l + sm_node_l + van_node_l + visual_node_l + dan_node_r + dmn_node_r + fpcn_node_r + limbic_node_r + sm_node_r + van_node_r + visual_node_r + uncertain_node
    network_labels = dan_node_l_idx + dmn_node_l_idx + fpcn_node_l_idx + limbic_node_l_idx + sm_node_l_idx + van_node_l_idx + visual_node_l_idx + dan_node_r_idx + dmn_node_r_idx + fpcn_node_r_idx + limbic_node_r_idx + sm_node_r_idx + van_node_r_idx + visual_node_r_idx +uncertain_node_idx

    with open('./network_label_idx.txt', 'w') as f:
        for label in network_labels:
            f.write('%s\n' % label)

    network_node_color = []
    for label in network_labels:
        if label == 0:
            network_node_color.append(DAN)
        elif label == 1:
            network_node_color.append(DMN)
        elif label == 2:
            network_node_color.append(FPCN)
        elif label == 3:
            network_node_color.append(Limbic)
        elif label == 4:
            network_node_color.append(SM)
        elif label == 5:
            network_node_color.append(VAN)
        elif label == 6:
            network_node_color.append(Visual)
        elif label == 7:
            network_node_color.append(Uncertain)
    print(len(network_node_color))

    real_mean = real_mean[:, network_node]
    real_mean = real_mean[network_node, :]
    fake_mean = fake_mean[:, network_node]
    fake_mean = fake_mean[network_node, :]

    fig = plt.figure()
    ax = fig.add_subplot()
    ax = plot_connectivity_matrix(fig=fig, ax=ax, matrix=real_mean, cmap='gist_heat', vmin=0, vmax=2.0, labels=network_labels)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax = plot_connectivity_matrix(fig=fig, ax=ax, matrix=fake_mean, cmap='gist_heat', vmin=0, vmax=2.0, labels=network_labels)
    fig.savefig('../fig/wgan-gp-sc_mean_connectivity_matrix.pdf', transparent=True)

    # # real
    # df_real_mean = pd.DataFrame(real_mean, index=network_labels)
    # df_real_mean_sum = df_real_mean.groupby(df_real_mean.index).sum()
    # df_real_mean_sum = df_real_mean_sum.T
    # df_real_mean_sum.index = network_labels
    # df_real_mean_sum = df_real_mean_sum.groupby(df_real_mean_sum.index).sum()
    # print(df_real_mean_sum)
    #
    # real_mean_sum = df_real_mean_sum.to_numpy()
    # np.fill_diagonal(real_mean_sum, 0)
    # G = nx.from_numpy_array(real_mean_sum)
    #
    # node_color = [DAN, DMN, FPCN, Limbic, SM, VAN, Visual, Uncertain]
    #
    # edge_weight = [data['weight'] for u, v, data in G.edges(data=True)]
    # node_strength = np.sum(real_mean_sum, axis=0)
    # edge_width = [10 * (ew - min(edge_weight)) / (max(edge_weight) - min(edge_weight)) for ew in edge_weight]
    # node_size = [(400 - 50) * (ns - min(node_strength)) / (max(node_strength) - min(node_strength)) + 50 for ns in
    #              node_strength]
    #
    # fig = plt.figure(figsize=(6, 6))
    # ax = fig.add_subplot()
    # nx.draw_networkx_edges(G=G, pos=nx.circular_layout(G), ax=ax, alpha=0.7,
    #                        edge_color=edge_weight, edge_cmap=plt.cm.YlOrRd, edge_vmin=0., edge_vmax=250, width=edge_width, )
    # nx.draw_networkx_nodes(G=G, pos=nx.circular_layout(G),
    #                        node_color=node_color, node_size=node_size, )
    #
    # # fake
    # df_fake_mean = pd.DataFrame(fake_mean, index=network_labels)
    # df_fake_mean_sum = df_fake_mean.groupby(df_fake_mean.index).sum()
    # df_fake_mean_sum = df_fake_mean_sum.T
    # df_fake_mean_sum.index = network_labels
    # df_fake_mean_sum = df_fake_mean_sum.groupby(df_fake_mean_sum.index).sum()
    # print(df_fake_mean_sum)
    #
    # fake_mean_sum = df_fake_mean_sum.to_numpy()
    # np.fill_diagonal(fake_mean_sum, 0)
    # G = nx.from_numpy_array(fake_mean_sum)
    #
    # node_color = [DAN, DMN, FPCN, Limbic, SM, VAN, Visual, Uncertain]
    #
    # edge_weight = [data['weight'] for u, v, data in G.edges(data=True)]
    # node_strength = np.sum(fake_mean_sum, axis=0)
    # edge_width = [10 * (ew - min(edge_weight)) / (max(edge_weight) - min(edge_weight)) for ew in edge_weight]
    # node_size = [(400 - 50) * (ns - min(node_strength)) / (max(node_strength) - min(node_strength)) + 50 for ns in
    #              node_strength]
    #
    # fig = plt.figure(figsize=(6, 6))
    # ax = fig.add_subplot()
    # nx.draw_networkx_edges(G=G, pos=nx.circular_layout(G), ax=ax, alpha=0.7,
    #                        edge_color=edge_weight, edge_cmap=plt.cm.YlOrRd, edge_vmin=0., edge_vmax=250, width=edge_width, )
    # nx.draw_networkx_nodes(G=G, pos=nx.circular_layout(G),
    #                        node_color=node_color, node_size=node_size, )
    # fig.savefig('../fig/wgan-gp-sc_network_graph.pdf', transparent=True)

plt.show()
