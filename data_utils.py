import os
import torch
from torch.utils.data import DataLoader

from utils import (
    getfile,
    load_h5_ModelNet,
    load_h5_ModelNet40E,
    load_h5_ScanObjectNN,
    load_h5_ScanObjectNNE,
)

class PointCloudDataset_modelnet40e(torch.utils.data.Dataset):
    def __init__(self, point_clouds, labels, sigmas, mus):
        self.point_clouds = point_clouds
        self.labels = labels
        self.sigmas = sigmas
        self.mus = mus

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        return self.point_clouds[idx], self.labels[idx], self.sigmas[idx], self.mus[idx]
    


def load_data_modelnet40e(dataset, batch_size, device, noise_level, shuffle):

    if 1==1:

        if dataset == 'modelnet40':

            # Replace this with your point cloud dataset
            ntrain = 0
            point_clouds_train = []
            pids_train = []
            sigmas_train = []
            mus_train = []
            trainfiles=getfile(os.path.join('modelnet40-e/train_files.txt'))
            for i in range(len(trainfiles)):
                if noise_level=="none":
                    traindata, labels = load_h5_ModelNet('objdata/ModelNet40/'+trainfiles[i])
                else:
                    traindata, labels, sigma_values, mu_values = load_h5_ModelNet40E(f'modelnet40-e/{noise_level}/train/'+trainfiles[i])
                for j in range(len(traindata)):
                    point_clouds_train.append(torch.from_numpy(traindata[j]).float().requires_grad_(True).to(device))
                    pids_train.append(torch.from_numpy(labels[j]).to(device))
                    if noise_level=="none":
                        n_points = traindata[j].shape[0]
                        zeros_sigma = torch.zeros(n_points, device=device)
                        zeros_mu = torch.zeros(n_points, device=device)
                        sigmas_train.append(zeros_sigma)
                        mus_train.append(zeros_mu)
                    else:
                        sigmas_train.append(torch.from_numpy(sigma_values[j]).to(device))
                        mus_train.append(torch.from_numpy(mu_values[j]).to(device))
                    ntrain+=1

            # Replace this with your point cloud dataset
            ntest = 0
            point_clouds_test = []
            pids_test = []
            sigmas_test = []
            mus_test = []
            testfiles=getfile(os.path.join('modelnet40-e/test_files.txt'))
            for i in range(len(testfiles)):
                if noise_level=="none":
                    testdata, labels = load_h5_ModelNet('objdata/ModelNet40/'+testfiles[i])
                else:
                    testdata, labels, sigma_values, mu_values = load_h5_ModelNet40E(f'modelnet40-e/{noise_level}/test/'+testfiles[i])
                for j in range(len(testdata)):
                    point_clouds_test.append(torch.from_numpy(testdata[j]).float().requires_grad_(True).to(device))
                    pids_test.append(torch.from_numpy(labels[j]).to(device))
                    if noise_level=="none":
                        n_points = testdata[j].shape[0]
                        zeros_sigma = torch.zeros(n_points, device=device)
                        zeros_mu = torch.zeros(n_points, device=device)
                        sigmas_test.append(zeros_sigma)
                        mus_test.append(zeros_mu)
                    else:
                        sigmas_test.append(torch.from_numpy(sigma_values[j]).to(device))
                        mus_test.append(torch.from_numpy(mu_values[j]).to(device))
                    ntest+=1

        train_dataset = PointCloudDataset_modelnet40e(point_clouds_train, pids_train, sigmas_train, mus_train)
        dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

        test_dataset = PointCloudDataset_modelnet40e(point_clouds_test, pids_test, sigmas_test, mus_test)
        dataloader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, dataloader_train, test_dataset, dataloader_test



def load_data_ScanObjectNNe(dataset, batch_size, device, noise_level, shuffle):
    point_clouds_train = []
    pids_train = []
    point_clouds_test = []
    pids_test = []

    if dataset == 'ScanObjectNN':

        ntrain = 0
        point_clouds_train = []
        pids_train = []
        sigmas_train = []
        mus_train = []

        ntest = 0
        point_clouds_test = []
        pids_test = []
        sigmas_test = []
        mus_test = []

        if noise_level=="none":
            print('Loading ScanObjectNN')
            traindata, labels = load_h5_ScanObjectNN('h5_files/main_split/training_objectdataset_augmentedrot_scale75.h5')
            traindata = traindata[:, :, :3]
            for j in range(len(traindata)):
                point_clouds_train.append(torch.from_numpy(traindata[j]).float().requires_grad_(True).to(device))
                pids_train.append(torch.tensor(labels[j]).to(device))
                n_points = traindata[j].shape[0]
                zeros_sigma = torch.zeros(n_points, device=device)
                zeros_mu = torch.zeros(n_points, device=device)
                sigmas_train.append(zeros_sigma)
                mus_train.append(zeros_mu)
                ntrain+=1

            testdata, labels = load_h5_ScanObjectNN('h5_files/main_split/test_objectdataset_augmentedrot_scale75.h5')
            testdata = testdata[:, :, :3]
            for j in range(len(testdata)):
                point_clouds_test.append(torch.from_numpy(testdata[j]).float().requires_grad_(True).to(device))
                pids_test.append(torch.tensor(labels[j]).to(device))
                n_points = testdata[j].shape[0]
                zeros_sigma = torch.zeros(n_points, device=device)
                zeros_mu = torch.zeros(n_points, device=device)
                sigmas_test.append(zeros_sigma)
                mus_test.append(zeros_mu)
                ntest+=1

        else:
            print('Loading ScanObjectNN')
            traindata, labels, sigma_values, mu_values = load_h5_ScanObjectNNE(f'ScanObjectNN-e/{noise_level}/training/training_objectdataset_augmentedrot_scale75.h5')
            traindata = traindata[:, :, :3]
            for j in range(len(traindata)):
                point_clouds_train.append(torch.from_numpy(traindata[j]).float().requires_grad_(True).to(device))
                pids_train.append(torch.tensor(labels[j]).to(device))
                sigmas_train.append(torch.from_numpy(sigma_values[j]).to(device))
                mus_train.append(torch.from_numpy(mu_values[j]).to(device))
                ntrain+=1

            testdata, labels, sigma_values, mu_values = load_h5_ScanObjectNNE(f'ScanObjectNN-e/{noise_level}/test/test_objectdataset_augmentedrot_scale75.h5')
            testdata = testdata[:, :, :3]
            for j in range(len(testdata)):
                point_clouds_test.append(torch.from_numpy(testdata[j]).float().requires_grad_(True).to(device))
                pids_test.append(torch.tensor(labels[j]).to(device))
                sigmas_test.append(torch.from_numpy(sigma_values[j]).to(device))
                mus_test.append(torch.from_numpy(mu_values[j]).to(device))  

    else:
        raise ValueError(f"Unknown dataset {dataset}")

    train_dataset = PointCloudDataset_modelnet40e(point_clouds_train, pids_train, sigmas_train, mus_train)
    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    test_dataset = PointCloudDataset_modelnet40e(point_clouds_test, pids_test, sigmas_test, mus_test)
    dataloader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, dataloader_train, test_dataset, dataloader_test