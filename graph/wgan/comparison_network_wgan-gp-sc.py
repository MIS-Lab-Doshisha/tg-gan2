import sys
import torch
import numpy as np
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
from utils.utils import load_cmap_mat


# Compare the acquired and synthesized connectome in adjacency matrix and brain connectome
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

cmap_mat = f'{data_dir}/colormap.mat'
cmap = load_cmap_mat(cmap_mat)

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
    network_labels = dan_node_l_idx + dmn_node_l_idx + fpcn_node_l_idx + limbic_node_l_idx + sm_node_l_idx + van_node_l_idx + visual_node_l_idx + dan_node_r_idx + dmn_node_r_idx + fpcn_node_r_idx + limbic_node_r_idx + sm_node_r_idx + van_node_r_idx + visual_node_r_idx + uncertain_node_idx

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

    difference_matrix = real - fake
    print('difference_matrix: ', difference_matrix.shape)
    difference_matrix_mean = np.average(difference_matrix, axis=0)
    print('difference_matrix_mean: ', difference_matrix_mean.shape)

    difference_matrix_mean = difference_matrix_mean[:, network_node]
    difference_matrix_mean = difference_matrix_mean[network_node, :]

    fig = plt.figure()
    ax = fig.add_subplot()
    ax = plot_connectivity_matrix(fig=fig, ax=ax, matrix=difference_matrix_mean, vmin=-0.5, vmax=0.5, cmap=cmap)
    fig.savefig('../fig/wgan-gp-sc_difference_matrix.pdf', transparent=True)

    # ------ Connectome Plot ------ #
    # Extract network
    network_labels = np.array(network_labels)
    network_123_idx = np.where((network_labels == 0) | (network_labels == 1) | (network_labels == 2))[0]
    network_4567_idx = np.where((network_labels == 3) | (network_labels == 4) | (network_labels == 5) | (network_labels == 6))[0]
    network_not_123_idx = np.where((network_labels != 0) & (network_labels != 1) & (network_labels != 2) & (network_labels != 3))[0]
    network_not_4567_idx = np.where((network_labels != 3) & (network_labels != 4) & (network_labels != 5) & (network_labels != 6))[0]
    print('network_123_idx: ', network_123_idx)
    print('network_4567_idx: ', network_4567_idx)
    print('network_not_123_idx: ', network_not_123_idx)
    print('network_not_4567_idx: ', network_not_4567_idx)

    # Extract partial network difference matrix
    difference_matrix_123_4567 = np.copy(difference_matrix_mean)
    difference_matrix_123_4567[:36, :36] = 0
    difference_matrix_123_4567[36:83, 36:83] = 0
    difference_matrix_123_4567 = difference_matrix_123_4567[:83, :83]

    fig = plt.figure()
    ax = fig.add_subplot()
    ax = plot_connectivity_matrix(fig=fig, ax=ax, matrix=difference_matrix_123_4567, vmin=-0.5, vmax=0.5, cmap=cmap)

    # ------ Connectome Plot ------ #
    # ------ Connectome Plot ------ #
    # ------ Extract partial network difference matrix [left-right]------ #
    difference_matrix_left_right = np.copy(difference_matrix_mean)
    difference_matrix_left_right[:42, :42] = 0
    difference_matrix_left_right[42:, 42:] = 0
    difference_matrix_left_right = difference_matrix_left_right[:83, :83]

    fig = plt.figure()
    ax = fig.add_subplot()
    ax = plot_connectivity_matrix(fig=fig, ax=ax, matrix=difference_matrix_left_right, vmin=-0.5, vmax=0.5, cmap=cmap)

    # Extract partial network coordinate and node color
    labels = np.array(aal['labels'])
    coordinates = np.array(coordinates)
    print('labels: ', labels)
    print('coordinates: ', coordinates.shape)

    new_labels = labels[network_node]
    new_labels = new_labels[:83]
    new_coordinates = coordinates[network_node]
    new_coordinates = new_coordinates[:83]
    new_node_color = network_node_color[:83]
    print('new_labels: ', new_labels)
    print('new_labels: ', new_labels.shape)
    print('new_coordinates: ', new_coordinates.shape)

    # Plot connectome plot
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot()
    ax = plotting.plot_connectome(adjacency_matrix=difference_matrix_left_right, node_coords=new_coordinates,
                                  edge_vmin=-0.5, edge_vmax=0.5, axes=ax, edge_kwargs={'linewidth': 2.0},
                                  node_size=30, node_color=new_node_color, node_kwargs={'edgecolors': '#000000'},
                                  edge_cmap=cmap, edge_threshold=0.4, colorbar=True)

    fig.savefig('../fig/wgan-gp-sc_difference_connectome_left-right.pdf', transparent=True)

    # ------ Extract partial network difference matrix [left-left] ------ #
    difference_matrix_left_left = np.copy(difference_matrix_mean)
    difference_matrix_left_left = difference_matrix_left_left[:42, :42]

    fig = plt.figure()
    ax = fig.add_subplot()
    ax = plot_connectivity_matrix(fig=fig, ax=ax, matrix=difference_matrix_left_left, vmin=-0.5, vmax=0.5, cmap=cmap)

    # Extract partial network coordinate and node color
    labels = np.array(aal['labels'])
    coordinates = np.array(coordinates)
    print('labels: ', labels)
    print('coordinates: ', coordinates.shape)

    new_labels = labels[network_node]
    new_labels = new_labels[:42]
    new_coordinates = coordinates[network_node]
    new_coordinates = new_coordinates[:42]
    new_node_color = network_node_color[:42]
    print('new_labels: ', new_labels)
    print('new_labels: ', new_labels.shape)
    print('new_coordinates: ', new_coordinates.shape)

    # Plot connectome plot
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot()
    ax = plotting.plot_connectome(adjacency_matrix=difference_matrix_left_left, node_coords=new_coordinates,
                                  edge_vmin=-0.5, edge_vmax=0.5, axes=ax, edge_kwargs={'linewidth': 2.0},
                                  node_size=30, node_color=new_node_color, node_kwargs={'edgecolors': '#000000'},
                                  edge_cmap=cmap, edge_threshold=0.4, colorbar=True)

    fig.savefig('../fig/wgan-gp-sc_difference_connectome_left-left.pdf', transparent=True)

    # ------ Extract partial network difference matrix [right-right] ------ #
    difference_matrix_right_right = np.copy(difference_matrix_mean)
    difference_matrix_right_right = difference_matrix_right_right[42:83, 42:83]

    fig = plt.figure()
    ax = fig.add_subplot()
    ax = plot_connectivity_matrix(fig=fig, ax=ax, matrix=difference_matrix_right_right, vmin=-0.5, vmax=0.5, cmap=cmap)

    # Extract partial network coordinate and node color
    labels = np.array(aal['labels'])
    coordinates = np.array(coordinates)
    print('labels: ', labels)
    print('coordinates: ', coordinates.shape)

    new_labels = labels[network_node]
    new_labels = new_labels[42:83]
    new_coordinates = coordinates[network_node]
    new_coordinates = new_coordinates[42:83]
    new_node_color = network_node_color[42:83]
    print('new_labels: ', new_labels)
    print('new_labels: ', new_labels.shape)
    print('new_coordinates: ', new_coordinates.shape)

    # Plot connectome plot
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot()
    ax = plotting.plot_connectome(adjacency_matrix=difference_matrix_right_right, node_coords=new_coordinates,
                                  edge_vmin=-0.5, edge_vmax=0.5, axes=ax, edge_kwargs={'linewidth': 2.0},
                                  node_size=30, node_color=new_node_color, node_kwargs={'edgecolors': '#000000'},
                                  edge_cmap=cmap, edge_threshold=0.4, colorbar=True)

    fig.savefig('../fig/wgan-gp-sc_difference_connectome_right-right.pdf', transparent=True)

    # ------ Extract partial network difference matrix [subcor-subcor] ------ #
    difference_matrix_subcor_subcor = np.copy(difference_matrix_mean)
    difference_matrix_subcor_subcor = difference_matrix_subcor_subcor[83:, 83:]

    fig = plt.figure()
    ax = fig.add_subplot()
    ax = plot_connectivity_matrix(fig=fig, ax=ax, matrix=difference_matrix_subcor_subcor, vmin=-0.5, vmax=0.5,
                                  cmap=cmap)

    # Extract partial network coordinate and node color
    labels = np.array(aal['labels'])
    coordinates = np.array(coordinates)
    print('labels: ', labels)
    print('coordinates: ', coordinates.shape)

    new_labels = labels[network_node]
    new_labels = new_labels[83:]
    new_coordinates = coordinates[network_node]
    new_coordinates = new_coordinates[83:]
    new_node_color = network_node_color[83:]
    print('new_labels: ', new_labels)
    print('new_labels: ', new_labels.shape)
    print('new_coordinates: ', new_coordinates.shape)

    # Plot connectome plot
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot()
    ax = plotting.plot_connectome(adjacency_matrix=difference_matrix_subcor_subcor, node_coords=new_coordinates,
                                  edge_vmin=-0.5, edge_vmax=0.5, axes=ax, edge_kwargs={'linewidth': 2.0},
                                  node_size=30, node_color=new_node_color, node_kwargs={'edgecolors': '#000000'},
                                  edge_cmap=cmap, edge_threshold=0.4, colorbar=True)

    fig.savefig('../fig/wgan-gp-sc_difference_connectome_subcor-subcor.pdf', transparent=True)

    # ------ Extract partial network difference matrix [non-subcor-subcor] ------ #
    difference_matrix_non_subcor_subcor = np.copy(difference_matrix_mean)
    difference_matrix_non_subcor_subcor[:83, :83] = 0
    difference_matrix_non_subcor_subcor[83:, 83:] = 0

    fig = plt.figure()
    ax = fig.add_subplot()
    ax = plot_connectivity_matrix(fig=fig, ax=ax, matrix=difference_matrix_non_subcor_subcor, vmin=-0.5, vmax=0.5,
                                  cmap=cmap)

    # Extract partial network coordinate and node color
    labels = np.array(aal['labels'])
    coordinates = np.array(coordinates)
    print('labels: ', labels)
    print('coordinates: ', coordinates.shape)

    new_labels = labels[network_node]
    new_labels = new_labels[:]
    new_coordinates = coordinates[network_node]
    new_coordinates = new_coordinates[:]
    new_node_color = network_node_color[:]
    print('new_labels: ', new_labels)
    print('new_labels: ', new_labels.shape)
    print('new_coordinates: ', new_coordinates.shape)

    # Plot connectome plot
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot()
    ax = plotting.plot_connectome(adjacency_matrix=difference_matrix_non_subcor_subcor, node_coords=new_coordinates,
                                  edge_vmin=-0.5, edge_vmax=0.5, axes=ax, edge_kwargs={'linewidth': 2.0},
                                  node_size=30, node_color=new_node_color, node_kwargs={'edgecolors': '#000000'},
                                  edge_cmap=cmap, edge_threshold=0.4, colorbar=True)

    fig.savefig('../fig/wgan-gp-sc_difference_connectome_non-subcor-subcor.pdf', transparent=True)

plt.show()
