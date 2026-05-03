import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from utils_old import *
from sklearn.neighbors import NearestNeighbors
from models import PointNet
from models import PointNet2
#from models import PTV3.model
from models import DGCNN
from models import PointMLP
from models import CurveNet
from models import SimpleView
from utils_modelnet40E import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device =', device)

set_seed(42)

all_models_calibration_curves = {}
all_models_stratified_ece = {}

dataset = "ScanObjectNN"

if dataset == "modelnet40":
    nclasses=40
elif dataset == "ScanObjectNN":
    nclasses=15

for modelname in ["PointNet", "PointNet2", "DGCNN", "PointMLP", "CurveNet", "SimpleView", "PTv3"]:

    calibration_curves = {}

    # Global accumulators for all predictions
    global_pred_probs = []
    global_correctness = []
    global_sigmas = []
    global_labels = []
    global_outputs = []   # NEW

    ECE_values = []

    per_level_sigmas = []
    per_level_outputs = []
    per_level_labels = []
    per_level_ece = []

    accs = []
    eces = []
    AUROCs = []
    pearsons_onlycorrect = []
    pearsons = []
    spearmans_onlycorrect = []
    spearmans = []

    for noise_level in ["none", "light", "moderate", "heavy"]:  # "none", "light", "moderate", "heavy"

        if modelname == "PointNet":
            model = PointNetCls(k=nclasses).to(device)
        elif modelname == "PointNet2":
            model = PointNet2_cls_msg(num_class=nclasses, normal_channel=False).to(device)
        elif modelname == "CurveNet":
            model = CurveNet(num_classes=nclasses).to(device)
        elif modelname == "PTv3":
            encoder = PointTransformerV3(
                in_channels=3,  # only XYZ
                cls_mode=False,
                enable_flash=False
            ).cuda()
            model = PointCloudTransformerAE_fullmodel_cls_false(encoder, k=nclasses).cuda()
        elif modelname == "DGCNN":
            class Args:
                def __init__(self):
                    self.k = 20
                    self.emb_dims = 1024
                    self.dropout = 0.5
                    self.leaky_relu = 1
            args = Args()
            model = DGCNN(args, output_channels=nclasses).to(device)
        elif modelname == "PointMLP":
            model = pointMLP(num_classes=nclasses).to(device)
        elif modelname == "SimpleView":
            model = MVModel(task='cls', dataset=dataset,
                            backbone='resnet18', feat_size=64).to(device)
        else:
            raise ValueError("Unknown modelname")

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        if modelname == "PTv3":
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

        if modelname == "PTv3":
            checkpoint_path = f"saved_models_modelnet40E/{dataset}e_none_ICLR_{modelname}_9_v2.pth"
        else:
            checkpoint_path = f"saved_models_modelnet40E/{dataset}e_none_ICLR_{modelname}_9_v2.pth"

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"Checkpoint loaded. Resuming from epoch {start_epoch}.")
        else:
            print("No checkpoint found.")
            start_epoch = 0

        batch_size = 32

        if dataset == "modelnet40":
            train_dataset, dataloader_train, test_dataset, dataloader_test = load_data_modelnet40e(
                dataset, batch_size, device, noise_level=noise_level, shuffle=False)
        else:
            train_dataset, dataloader_train, test_dataset, dataloader_test = load_data_ScanObjectNNe(
                dataset, batch_size, device, noise_level=noise_level, shuffle=False)

        model.eval()

        all_pred_probs = []
        all_correctness = []
        all_sigmas = []
        all_outputs = []
        all_labels = []
        total_correct = 0
        total_samples = 0
        pcwise_correlations = []

        with torch.no_grad():
            for batch_idx, (batch, labels, sigmas, mus) in enumerate(dataloader_test):
                batch = batch.to(device)
                labels = labels.to(device)

                if modelname == "PointNet":
                    batch = batch.permute(0, 2, 1)
                    outputs, trans, trans_feat = model(batch)
                    probs = torch.exp(outputs)
                elif modelname == "PointNet2":
                    batch = batch.permute(0, 2, 1)
                    outputs, _ = model(batch)
                    probs = torch.softmax(outputs, dim=1)
                elif modelname == "DGCNN":
                    batch = batch.permute(0, 2, 1)
                    outputs = model(batch)
                    probs = torch.softmax(outputs, dim=1)
                elif modelname == "PointMLP":
                    batch = batch.permute(0, 2, 1)
                    outputs = model(batch)
                    probs = torch.softmax(outputs, dim=1)
                elif modelname == "CurveNet":
                    batch = batch.permute(0, 2, 1)
                    outputs = model(batch)
                    probs = torch.softmax(outputs, dim=1)
                elif modelname == "PTv3":
                    B, N, C = batch.shape
                    pointcloud = batch.view(B * N, 3)
                    batch_indices = torch.arange(B).unsqueeze(1).repeat(1, N).view(-1)
                    data_dict = {
                        'coord': pointcloud.cuda(),
                        'batch': batch_indices.cuda(),
                        'feat': pointcloud.cuda(),
                        'grid_size': torch.tensor(0.01).cuda()
                    }
                    outputs = model(data_dict)
                    probs = torch.softmax(outputs, dim=1)
                elif modelname == "SimpleView":
                    outputs = model(batch)['logit']
                    probs = torch.softmax(outputs, dim=1)
                else:
                    raise ValueError(f"Unknown model: {modelname}")

                y_pred_prob, y_pred_label = probs.max(dim=1)
                if dataset=="modelnet40":
                    y_true = (y_pred_label == labels.squeeze(1)).long()
                else:
                    y_true = (y_pred_label == labels).long()
                

                total_correct += y_true.sum().item()
                total_samples += y_true.numel()

                all_pred_probs.append(y_pred_prob.cpu())
                all_correctness.append(y_true.cpu())
                all_sigmas.append(sigmas.mean(dim=1).cpu())
                all_outputs.append(probs.cpu())
                if dataset=="modelnet40":
                    all_labels.append(labels.squeeze(1).long().cpu())
                else:
                    all_labels.append(labels.long().cpu())

                if noise_level != "none":
                    corr = compute_uncertainty_correlation_pcwise(sigmas, y_pred_prob)
                    pcwise_correlations.append(corr)

                #if batch_idx == 5:
                #    break

        accuracy = total_correct / total_samples
        accs.append(accuracy)
        print(f"✅ Accuracy over full test set = {accuracy:.4f}")

        all_pred_probs = torch.cat(all_pred_probs, dim=0)
        all_correctness = torch.cat(all_correctness, dim=0)
        all_sigmas = torch.cat(all_sigmas, dim=0)
        all_outputs = torch.cat(all_outputs, dim=0)  # [N, C]
        all_labels = torch.cat(all_labels, dim=0)    # [N]

        ece, bin_confidences, bin_accuracies, bin_counts = compute_ece_torch(
            all_correctness, all_pred_probs, n_bins=5
        )
        eces.append(ece)
        print(f"✅ ECE over full test set = {ece:.4f}")

        ECE_values.append(ece)

        per_level_sigmas.append(all_sigmas)     # [N_level]
        per_level_outputs.append(all_outputs)   # [N_level, C]
        per_level_labels.append(all_labels)     # [N_level]
        per_level_ece.append(ece)               # float

        corr_all, corr_correct, corr_incorrect = compute_uncertainty_correlation_stratified(
            all_sigmas, all_pred_probs, all_correctness
        )
        print(f"Pearson (all): {corr_all:.4f}, correct-only: {corr_correct:.4f}, incorrect-only: {corr_incorrect:.4f}")

        nll = compute_nll(all_outputs, all_labels)
        brier = compute_brier(all_outputs, all_labels, num_classes=nclasses)
        auroc = compute_auroc_error_detection(all_pred_probs, all_correctness)
        print(f"NLL={nll:.4f}, Brier={brier:.4f}, AUROC-error={auroc:.4f}")
        AUROCs.append(auroc)

        calibration_curves[noise_level] = {
            "bin_confidences": bin_confidences,
            "bin_accuracies": bin_accuracies
        }

        if noise_level != "none":
            p_max, all_preds = all_outputs.max(dim=1)
            corr_onlycorrect = compute_uncertainty_correlation_correct_only(all_sigmas, all_pred_probs, all_preds, all_labels)
            corr = compute_uncertainty_correlation_torch(all_sigmas, all_pred_probs)
            pearsons_onlycorrect.append(corr_onlycorrect)
            pearsons.append(corr)
            print(f"✅ Correlation between σ and predicted uncertainty (only correct): {corr_onlycorrect:.4f}")
            print(f"✅ Correlation between σ and predicted uncertainty: {corr:.4f}")

            corr_onlycorrect = compute_uncertainty_spearman_correct_only(all_sigmas, all_pred_probs, all_preds, all_labels)
            corr = compute_uncertainty_spearman_torch(all_sigmas, all_pred_probs)
            spearmans_onlycorrect.append(corr_onlycorrect)
            spearmans.append(corr)
            print(f"✅ Spearman (only correct): {corr_onlycorrect:.4f}")
            print(f"✅ Spearman: {corr:.4f}")

            print(f"✅ PCwise Correlation between σ and predicted uncertainty:{np.array(pcwise_correlations).mean()}")
        
            adj_awareness, raw_corr, penalty = compute_adjusted_awareness(
                sigmas.mean(dim=1).cpu(),        # move to CPU
                probs.cpu(),         # move to CPU
                labels.cpu(),        # move to CPU
                ece_none=ECE_values[0],
                ece_noise=ece,
                eps=1e-6
            )
            print(f"Adjusted awareness (correct only): {adj_awareness:.4f} | "
                f"Raw corr={raw_corr:.4f}, Penalty={penalty:.4f}")

            # Append to global
            global_pred_probs.append(all_pred_probs)
            global_correctness.append(all_correctness)
            global_sigmas.append(all_sigmas)
            global_labels.append(all_labels)
            global_outputs.append(all_outputs)   # NEW

    all_models_calibration_curves[modelname] = calibration_curves

    # =============================================
    # Global stratified ECE across all noise levels
    # =============================================

    global_pred_probs = torch.cat(global_pred_probs, dim=0)
    global_correctness = torch.cat(global_correctness, dim=0)
    global_sigmas = torch.cat(global_sigmas, dim=0)
    global_labels = torch.cat(global_labels, dim=0)
    global_outputs = torch.cat(global_outputs, dim=0)  # [N, C]  # NEW

    p_max, global_preds = global_outputs.max(dim=1)

    global_corr_onlycorrect = compute_uncertainty_correlation_correct_only(global_sigmas, global_pred_probs, global_preds, global_labels)
    global_corr = compute_uncertainty_correlation_torch(global_sigmas, global_pred_probs)
    global_corr_bins = compute_binned_uncertainty_correlation(global_sigmas, global_pred_probs, n_bins=30, plot=True)
    print(f"🌍 Global correlation across all noise levels (only correct) = {global_corr_onlycorrect:.4f}")
    print(f"🌍 Global correlation across all noise levels = {global_corr:.4f}")
    print(f"🌍 Global correlation across all noise levels bins = {global_corr_bins:.4f}")

    global_corr_correct, n_used = compute_awareness_correct_only(
        global_sigmas, global_outputs, global_labels
    )
    print(f"🌍 Global awareness (correct only): {global_corr_correct:.4f} (n={n_used})")

    ECE_values = np.array(ECE_values)

    sigmas_np = global_sigmas.numpy()
    pred_probs_np = global_pred_probs.numpy()
    correctness_np = global_correctness.numpy()

    bins = np.quantile(sigmas_np, [0.0, 0.25, 0.5, 0.75, 1.0])
    print("Global σ bins (quartiles):", bins)

    bin_indices = np.digitize(sigmas_np, bins, right=False) - 1

    print("\n✅ Stratified ECE by σ quartiles (all noise levels combined):")
    stratified_ece = {}
    n_bins_ece = 10

    for i in range(4):
        mask = bin_indices == i
        count = mask.sum()
        if count == 0:
            print(f"Bin {i}: no samples.")
            continue

        bin_preds = pred_probs_np[mask]
        bin_correct = correctness_np[mask]

        ece, bin_confidences, bin_accuracies, bin_counts = compute_ece_torch(
            torch.from_numpy(bin_correct),
            torch.from_numpy(bin_preds),
            n_bins=n_bins_ece
        )

        stratified_ece[i] = {
            "ece": ece,
            "bin_confidences": bin_confidences,
            "bin_accuracies": bin_accuracies,
            "count": count
        }

        print(f"Bin {i}: count={count}, ECE={ece:.4f}")

    print("\n\n")
    print(f"final results {modelname}")
    print("accs =", np.array2string(100*np.array(accs), precision=2))
    print("eces =", np.array2string(np.array(eces), precision=4))
    print("AUROCs =", np.array2string(np.array(AUROCs), precision=4))
    print("Pearsons (correct only) =", np.array2string(np.array(pearsons_onlycorrect), precision=4))
    print("Pearsons =", np.array2string(np.array(pearsons), precision=4))
    print(f"uncertainty awareness (correct only) = {global_corr_onlycorrect:.4f}")
    print(f"uncertainty awareness = {global_corr:.4f}")
    print("\n\n")