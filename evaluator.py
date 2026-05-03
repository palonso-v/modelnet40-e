import os
import torch
import numpy as np
from metrics import (
    compute_accuracy,
    compute_ece,
    compute_auroc_error_detection,
    compute_pearson_uncertainty,
    compute_pearson_uncertainty_correct_only,
)
from data_utils import load_data_modelnet40e, load_data_ScanObjectNNe
from models import build_model, forward_model


NOISE_LEVELS = ["none", "light", "moderate", "heavy"]


def get_num_classes(dataset):
    if dataset == "modelnet40":
        return 40
    if dataset == "ScanObjectNN":
        return 15
    raise ValueError(f"Unknown dataset: {dataset}")


def load_checkpoint(model, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    print(f"Loaded checkpoint: {checkpoint_path}")


def get_test_loader(dataset, batch_size, device, noise_level):
    if dataset == "modelnet40":
        _, _, _, test_loader = load_data_modelnet40e(
            dataset,
            batch_size,
            device,
            noise_level=noise_level,
            shuffle=False,
        )
    elif dataset == "ScanObjectNN":
        _, _, _, test_loader = load_data_ScanObjectNNe(
            dataset,
            batch_size,
            device,
            noise_level=noise_level,
            shuffle=False,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return test_loader


def collect_predictions(model, loader, model_name, dataset, device):
    model.eval()

    all_probs = []
    all_labels = []
    all_sigmas = []

    with torch.no_grad():
        for batch, labels, sigmas, mus in loader:
            batch = batch.to(device)
            labels = labels.to(device)

            outputs = forward_model(model, batch, model_name, device)
            probs = torch.softmax(outputs, dim=1)

            if dataset == "modelnet40":
                labels = labels.squeeze(1).long()
            else:
                labels = labels.long()

            all_probs.append(probs.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_sigmas.append(sigmas.mean(dim=1).detach().cpu())

    probs = torch.cat(all_probs, dim=0)
    labels = torch.cat(all_labels, dim=0)
    sigmas = torch.cat(all_sigmas, dim=0)

    return probs, labels, sigmas


def evaluate_one_noise_level(model, config, noise_level, device):
    loader = get_test_loader(
        dataset=config.dataset,
        batch_size=config.batch_size,
        device=device,
        noise_level=noise_level,
    )

    probs, labels, sigmas = collect_predictions(
        model=model,
        loader=loader,
        model_name=config.model,
        dataset=config.dataset,
        device=device,
    )

    results = {
        "accuracy": compute_accuracy(probs, labels),
        "ece": compute_ece(probs, labels, n_bins=config.n_bins),
        "auroc_error": compute_auroc_error_detection(probs, labels),
    }

    if noise_level != "none":
        results["pearson_correct"] = compute_pearson_uncertainty_correct_only(
            probs, labels, sigmas
        )
        results["pearson_all"] = compute_pearson_uncertainty(
            probs, sigmas
        )
    else:
        results["pearson_correct"] = np.nan
        results["pearson_all"] = np.nan

    return results, probs, labels, sigmas


def evaluate_classification(config):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print("device =", device)

    num_classes = get_num_classes(config.dataset)

    model = build_model(
        model_name=config.model,
        dataset=config.dataset,
        num_classes=num_classes,
        device=device,
    )

    load_checkpoint(model, config.checkpoint, device)

    if config.noise_level == "all":
        noise_levels = NOISE_LEVELS
    else:
        noise_levels = [config.noise_level]

    all_results = {}

    global_probs = []
    global_labels = []
    global_sigmas = []

    for noise_level in noise_levels:
        print(f"\nEvaluating noise level: {noise_level}")

        results, probs, labels, sigmas = evaluate_one_noise_level(
            model=model,
            config=config,
            noise_level=noise_level,
            device=device,
        )

        all_results[noise_level] = results

        if noise_level != "none":
            global_probs.append(probs)
            global_labels.append(labels)
            global_sigmas.append(sigmas)

        print(
            f"{noise_level:>8s} | "
            f"acc: {100 * results['accuracy']:.2f} | "
            f"ECE: {results['ece']:.4f} | "
            f"AUROC: {results['auroc_error']:.4f} | "
            f"Pearson correct: {results['pearson_correct']:.4f} | "
            f"Pearson all: {results['pearson_all']:.4f}"
        )

    if len(global_probs) > 0:
        global_probs = torch.cat(global_probs, dim=0)
        global_labels = torch.cat(global_labels, dim=0)
        global_sigmas = torch.cat(global_sigmas, dim=0)

        global_pearson_correct = compute_pearson_uncertainty_correct_only(
            global_probs, global_labels, global_sigmas
        )
        global_pearson_all = compute_pearson_uncertainty(
            global_probs, global_sigmas
        )

        print("\nGlobal uncertainty awareness across noisy levels:")
        print(f"Pearson correct-only: {global_pearson_correct:.4f}")
        print(f"Pearson all:          {global_pearson_all:.4f}")

    return all_results