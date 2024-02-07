import os
import sys
import pickle
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

sys.path.append('.')
from utils.load_dataset import NIMHSC
from utils.synthesizer import matrices_targets_synthesizer
from models.gan import Encoder, Decoder


# [tg-gan-2] Synthesize new adjacency matrices and objective scores
# ------ Setting ------ #
dropout = 0.1
alpha = 0.6
n_splits = 5

# ------ Load model parameters ------ #
checkpoint_encoder_pth = f'../../tg-gan-2/data/tg-gan-sc/dr_{dropout}_alpha_{alpha}/checkpoint_encoder.pth'
checkpoint_decoder_pth = f'../../tg-gan-2/data/tg-gan-sc/dr_{dropout}_alpha_{alpha}/checkpoint_decoder.pth'
checkpoint_encoder = torch.load(checkpoint_encoder_pth, map_location=torch.device('cpu'))
checkpoint_decoder = torch.load(checkpoint_decoder_pth, map_location=torch.device('cpu'))

# ------ Load dataset ------ #
transform = transforms.Compose([transforms.ToTensor()])
train_datasets = NIMHSC(participants_csv='../../data/nihtoolbox_disc.csv', target='nihtoolbox', transforms=transform,
                        data_dir='../../data/structural_connectivity',
                        atlas='aal116_sift_invnodevol_radius2_count_connectivity')
train_loader = DataLoader(dataset=train_datasets, batch_size=1, shuffle=False)

true_matrices = []
true_targets = []
for i, (mat, target) in enumerate(train_loader):
    # Connectivity matrix
    true_matrices.append(mat)

    # Target
    true_targets.append(target.item())

# ------ Generate connectivity matrices ------ #
encoder = Encoder(n_features=64, n_regions=116)
encoder.load_state_dict(checkpoint_encoder)
decoder = Decoder(n_features=64, n_regions=116)
decoder.load_state_dict(checkpoint_decoder)

encoder.eval()
decoder.eval()

synthesized_matrices, synthesized_targets = matrices_targets_synthesizer(encoder=encoder,
                                                                         decoder=decoder,
                                                                         matrices=true_matrices, targets=true_targets,
                                                                         n_splits=n_splits)
print('synthesized_matrices: ', synthesized_matrices.shape)
print('synthesized_target: ', synthesized_targets.shape)

# ------ Save ------ #
data_dir = f'../../data/tg-gan-2-sc/dr_{dropout}_alpha_{alpha}'
os.makedirs(data_dir, exist_ok=True)

synthesized_matrices_pkl = f'{data_dir}/tg-gan-2-sc_n-splits_{n_splits}_matrices.pkl'
with open(synthesized_matrices_pkl, mode='wb') as f:
    pickle.dump(synthesized_matrices, f)

synthesized_targets_pkl = f'{data_dir}/tg-gan-2-sc_n-splits_{n_splits}_targets.pkl'
with open(synthesized_targets_pkl, mode='wb') as f:
    pickle.dump(synthesized_targets, f)
