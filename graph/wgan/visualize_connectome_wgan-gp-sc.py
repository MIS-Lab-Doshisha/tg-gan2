import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader

from nilearn import datasets
from nilearn import plotting
from nichord.coord_labeler import get_idx_to_label
from nichord.convert import convert_matrix
from nichord.chord import plot_chord

sys.path.append('.')
from utils.load_dataset import NIMHSC
from utils.graph import binarize_matrix
from utils.visualizer import plot_connectivity_matrix
from models.gan import Encoder, Decoder


# Visualize acquired and synthesized connectome in yeo's network order
# Visualize binarized matrices in each edge density
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

    # network alignment
    network_idx = get_idx_to_label(coords=coordinates)
    print(network_idx)

    dan_node_l, dan_node_r, dmn_node_l, dmn_node_r, fpcn_node_l, fpcn_node_r, limbic_node_l, limbic_node_r, sm_node_l, sm_node_r, van_node_l, van_node_r, visual_node_l, visual_node_r, uncertain_node = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    dan_node_idx, dmn_node_idx, fpcn_node_idx, limbic_node_idx, sm_node_idx, van_node_idx, visual_node_idx, uncertain_node_idx = [], [], [], [], [], [], [], []
    for node, network in network_idx.items():
        label = aal['labels'][node]
        if network == 'DAN':
            dan_node_idx.append(0)
            if label[-1] == 'L':
                dan_node_l.append(node)
            if label[-1] == 'R':
                dan_node_r.append(node)
        elif network == 'DMN':
            dmn_node_idx.append(1)
            if label[-1] == 'L':
                dmn_node_l.append(node)
            if label[-1] == 'R':
                dmn_node_r.append(node)
        elif network == 'FPCN':
            fpcn_node_idx.append(2)
            if label[-1] == 'L':
                fpcn_node_l.append(node)
            if label[-1] == 'R':
                fpcn_node_r.append(node)
        elif network == 'Limbic':
            limbic_node_idx.append(3)
            if label[-1] == 'L':
                limbic_node_l.append(node)
            if label[-1] == 'R':
                limbic_node_r.append(node)
        elif network == 'SM':
            sm_node_idx.append(4)
            if label[-1] == 'L':
                sm_node_l.append(node)
            if label[-1] == 'R':
                sm_node_r.append(node)
        elif network == 'VAN':
            van_node_idx.append(5)
            if label[-1] == 'L':
                van_node_l.append(node)
            if label[-1] == 'R':
                van_node_r.append(node)
        elif network == 'Visual':
            visual_node_idx.append(6)
            if label[-1] == 'L':
                visual_node_l.append(node)
            if label[-1] == 'R':
                visual_node_r.append(node)
        elif network == 'Uncertain':
            uncertain_node_idx.append(7)
            uncertain_node.append(node)

    network_node = dan_node_l + dmn_node_l + fpcn_node_l + limbic_node_l + sm_node_l + van_node_l + visual_node_l + dan_node_r + dmn_node_r + fpcn_node_r + limbic_node_r + sm_node_r + van_node_r + visual_node_r + uncertain_node
    network_labels = dan_node_idx + dmn_node_idx + fpcn_node_idx + limbic_node_idx + sm_node_idx + van_node_idx + visual_node_idx + uncertain_node_idx
    print('network_node')
    print(len(network_node))

    real_mean = real_mean[network_node, :]
    real_mean = real_mean[:, network_node]
    fake_mean = fake_mean[network_node, :]
    fake_mean = fake_mean[:, network_node]
    print(real_mean.shape)

    # fig = plt.figure()
    # ax = fig.add_subplot()
    # ax = plot_connectivity_matrix(fig=fig, ax=ax, matrix=fake_mean, cmap='gist_heat', vmin=0, vmax=5, labels=network_labels)
    # fig.savefig('../fig/wgan-2-sc_network_matrix.pdf', transparent=True)

    for ed in edge_density:
        # Binarize
        real_bin = binarize_matrix(real_mean, edge_density=ed)
        fake_bin = binarize_matrix(fake_mean, edge_density=ed)
        print('real_bin: ', real_bin.shape)
        print('fake_bin: ', fake_bin.shape)

        fig = plt.figure()
        ax = fig.add_subplot()
        ax = plot_connectivity_matrix(fig=fig, ax=ax, matrix=fake_bin, cmap='gray', vmin=0, vmax=1, labels=network_labels)
        fig.savefig(f'../fig/wgan-gp-sc_network_matrix_edge_density_{ed}.pdf', transparent=True)

        # Extract common edge
        common_bin = real_bin * fake_bin
        common_bin_node_strength = np.sum(common_bin, axis=0)
        common_bin_node_zero_idx = np.where(common_bin_node_strength == 0)[0]

        # Process for connctome plot
        new_common_coordinates = np.delete(coordinates, common_bin_node_zero_idx, axis=0)
        new_common = np.delete(fake_mean, common_bin_node_zero_idx, axis=0)
        new_common = np.delete(new_common, common_bin_node_zero_idx, axis=1)
        new_common_bin = np.delete(common_bin, common_bin_node_zero_idx, axis=0)
        new_common_bin = np.delete(new_common_bin, common_bin_node_zero_idx, axis=1)
        print('new_real: ', new_common.shape)
        print('new_real_bin: ', new_common_bin.shape)

        fake_network = get_idx_to_label(new_common_coordinates)
        fake_node_color = []
        for network in fake_network.values():
            if network == 'DAN':
                fake_node_color.append(DAN)
            elif network == 'DMN':
                fake_node_color.append(DMN)
            elif network == 'FPCN':
                fake_node_color.append(FPCN)
            elif network == 'Limbic':
                fake_node_color.append(Limbic)
            elif network == 'SM':
                fake_node_color.append(SM)
            elif network == 'VAN':
                fake_node_color.append(VAN)
            elif network == 'Visual':
                fake_node_color.append(Visual)
            elif network == 'Uncertain':
                fake_node_color.append(Uncertain)

        fig = plt.figure(figsize=(8, 3.5))
        ax = fig.add_subplot()
        ax = plotting.plot_connectome(adjacency_matrix=new_common, node_coords=new_common_coordinates,
                                      edge_threshold='95%', node_size=30, edge_vmin=0, edge_vmax=15,
                                      figure=fig, axes=ax,
                                      edge_cmap='YlOrRd', colorbar=True, node_color=fake_node_color,
                                      display_mode='ortho',
                                      node_kwargs={'edgecolors': '#000000'})

plt.show()
