import os
import pickle
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


def load_functional_connectivity(participants_csv, atlas, data_dir):
    """load_functional_connectivity: load functional connectivity

    :param participants_csv: path of participants.csv
    :param atlas: brian atlas name
    :param data_dir: data directory containing functional connectivity
    :return:
    """

    # ------ Load participants csv file ------ #
    participants = pd.read_csv(participants_csv, header=0, index_col=0)
    participants = participants['participant_id'].to_list()

    # ------ Path of all functional connectivity ------ #
    functional_connectivity_path = [os.path.join(data_dir, f'{subj}_functional_connectivity_{atlas}.pkl') for subj in participants]

    # ------ Load functional connectivity ------ #
    functional_connectivity = []
    for fcp in functional_connectivity_path:
        with open(fcp, mode='rb') as f:
            _functional_connectivity = pickle.load(f)[0]
        _functional_connectivity[_functional_connectivity == float('inf')] = 0
        functional_connectivity.append(_functional_connectivity)
    functional_connectivity = np.array(functional_connectivity)

    return functional_connectivity


def load_structural_connectivity_hcp(participants_csv: str, data_dir: str):
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
    structural_connectivity_path = [os.path.join(data_dir, f'{subj}_desc-hcpmmp1_connectome.csv') for subj in participants]

    # ------ Load structural connectivity ------ #
    structural_connectivity = []
    for scp in structural_connectivity_path:
        _structural_connectivity = pd.read_csv(scp, header=None).to_numpy(dtype=np.float32)
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


class NIMHFC(Dataset):

    def __init__(self, participants_csv, data_dir: str, target: str, atlas: str, transforms=None):
        """NIMHFC: Functional connectivity and target score from NIMH dataset

        :param participants_csv:
        :param data_dir: data directory
        :param target: target score name
        :param atlas: brain atlas name
        :param transforms: torchvision.transforms
        """

        super().__init__()
        self.transforms = transforms

        # ------ Load dataset ------ #
        self.data = load_functional_connectivity(participants_csv=participants_csv, atlas=atlas, data_dir=data_dir,)
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


class HCPSC(Dataset):

    def __init__(self, participants_csv, data_dir: str, target: str, transforms=None):
        """HCPSC: Structural connectivity and target score from HCP dataset

        :param data_dir: data directory
        :param target: target score name
        :param transforms: torchvision.transforms
        """

        super().__init__()
        self.transforms = transforms

        # ------ Load dataset ------ #
        self.data = load_structural_connectivity_hcp(participants_csv=participants_csv, data_dir=data_dir)
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

