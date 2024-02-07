import os
import numpy as np
import pandas as pd
import scipy.io as io
from torch.utils.data import Dataset


def load_score(participants_csv: str, target: str):
    """load_score: load target score from participants.csv

    :param participants_csv: path of participants.csv
    :param target: target score name
    :return: score: target score
    """

    participants = pd.read_csv(participants_csv, header=0, index_col=0)
    score = participants[target].to_list()
    score = np.array(score)

    return score


def load_structural_connectivity(participants_csv: str, atlas: str, data_dir: str, suffix_mat: str):
    """load_structural_connectivity: load structural connectivity

    :param participants_csv: path of participants.csv
    :param atlas: brian atlas name
    :param data_dir: data directory containing structural connectivity
    :param suffix_mat: suffix behind sub-{subj_id}
    :return: structural_connectivity: structural_connectivity (n_samples, n_regions, n_regions)
    """

    # ------ Load participants csv file ------ #
    participants = pd.read_csv(participants_csv, header=0, index_col=0)
    participants = participants['participant_id'].to_list()

    # ------ Path (.mat) of structural connectivity ------ #
    structural_connectivity_path = [os.path.join(data_dir, f'{subj}_{suffix_mat}.mat') for subj in participants]

    # ------ Load structural connectivity ------ #
    structural_connectivity = []
    for scp in structural_connectivity_path:
        _structural_connectivity = io.loadmat(scp)[atlas]
        _structural_connectivity[_structural_connectivity == float('inf')] = 0
        structural_connectivity.append(_structural_connectivity)
    structural_connectivity = np.array(structural_connectivity)

    return structural_connectivity


class NIMHSC(Dataset):

    def __init__(self, participants_csv, data_dir: str, target: str, atlas: str, transforms=None):
        """NIMHSC: Structural connectivity and target score from NIMH dataset

        :param data_dir: data directory
        :param target: target score name
        :param atlas: brian atlas name
        :param transforms: torchvision.transforms
        """

        super().__init__()
        self.transforms = transforms

        # ------ Load dataset ------ #
        self.data = load_structural_connectivity(participants_csv=participants_csv, atlas=atlas, data_dir=data_dir,
                                                 suffix_mat='ses-01_space-T1w_desc-preproc_dhollanderconnectome')
        self.target = load_score(participants_csv=participants_csv, target=target)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # ------ Load dataset ------ #
        data = self.data[index]
        target = self.target[index]

        # ------ Transform ------ #
        if self.transforms:
            data = self.transforms(data)

        return data, target
