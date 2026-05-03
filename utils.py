import random
import h5py
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def getfile(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def load_h5_ModelNet(h5_filename):
    with h5py.File(h5_filename, "r") as f:
        data = f["data"][:]
        label = f["label"][:]
    return data, label


def load_h5_ModelNet40E(h5_filename):
    with h5py.File(h5_filename, "r") as f:
        data = f["data"][:]
        label = f["label"][:]
        sigma = f["sigma"][:]
        mu = f["mu"][:]
    return data, label, sigma, mu


def load_h5_ScanObjectNN(h5_filename):
    with h5py.File(h5_filename, "r") as f:
        data = f["data"][:]
        label = f["label"][:]
    return data, label


def load_h5_ScanObjectNNE(h5_filename):
    with h5py.File(h5_filename, "r") as f:
        data = f["data"][:]
        label = f["label"][:]
        sigma = f["sigma"][:]
        mu = f["mu"][:]
    return data, label, sigma, mu