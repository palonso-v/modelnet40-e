import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def compute_accuracy(probs, labels):
    """
    Classification accuracy.

    Args:
        probs: Tensor [N, C], softmax probabilities.
        labels: Tensor [N], ground-truth labels.

    Returns:
        float
    """
    preds = probs.argmax(dim=1)
    return (preds == labels).float().mean().item()


def compute_ece(probs, labels, n_bins=10):
    """
    Expected Calibration Error.

    Args:
        probs: Tensor [N, C], softmax probabilities.
        labels: Tensor [N], ground-truth labels.
        n_bins: number of confidence bins.

    Returns:
        float
    """
    confidences, preds = probs.max(dim=1)
    correctness = (preds == labels).float()

    bins = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        if i == 0:
            mask = (confidences >= bins[i]) & (confidences <= bins[i + 1])
        else:
            mask = (confidences > bins[i]) & (confidences <= bins[i + 1])

        count = mask.sum().item()
        if count == 0:
            continue

        bin_confidence = confidences[mask].mean()
        bin_accuracy = correctness[mask].mean()

        ece += (count / len(labels)) * torch.abs(bin_confidence - bin_accuracy).item()

    return float(ece)


def compute_calibration_bins(probs, labels, n_bins=10):
    """
    Return calibration curve data.

    Returns:
        ece, bin_confidences, bin_accuracies, bin_counts
    """
    confidences, preds = probs.max(dim=1)
    correctness = (preds == labels).float()

    bins = torch.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []

    for i in range(n_bins):
        if i == 0:
            mask = (confidences >= bins[i]) & (confidences <= bins[i + 1])
        else:
            mask = (confidences > bins[i]) & (confidences <= bins[i + 1])

        count = mask.sum().item()

        if count > 0:
            bin_confidence = confidences[mask].mean().item()
            bin_accuracy = correctness[mask].mean().item()
            ece += (count / len(labels)) * abs(bin_confidence - bin_accuracy)
        else:
            bin_confidence = 0.0
            bin_accuracy = 0.0

        bin_confidences.append(bin_confidence)
        bin_accuracies.append(bin_accuracy)
        bin_counts.append(count)

    return float(ece), bin_confidences, bin_accuracies, bin_counts


def compute_nll(probs, labels):
    """
    Negative log-likelihood from probabilities.

    Args:
        probs: Tensor [N, C], softmax probabilities.
        labels: Tensor [N].
    """
    log_probs = torch.log(probs + 1e-12)
    return F.nll_loss(log_probs, labels, reduction="mean").item()


def compute_brier(probs, labels, num_classes):
    """
    Multiclass Brier score.

    Args:
        probs: Tensor [N, C].
        labels: Tensor [N].
        num_classes: int.
    """
    one_hot = F.one_hot(labels, num_classes=num_classes).float()
    return ((probs - one_hot) ** 2).sum(dim=1).mean().item()


def compute_auroc_error_detection(probs, labels):
    """
    AUROC for error detection using predicted uncertainty = 1 - p_max.

    Incorrect predictions are treated as the positive class.
    """
    confidences, preds = probs.max(dim=1)
    uncertainty = 1.0 - confidences
    errors = (preds != labels).long()

    try:
        return roc_auc_score(
            errors.detach().cpu().numpy(),
            uncertainty.detach().cpu().numpy(),
        )
    except ValueError:
        return float("nan")


def _pearson(x, y):
    """
    Pearson correlation between two 1D tensors.
    """
    x = x.detach().cpu().float().view(-1)
    y = y.detach().cpu().float().view(-1)

    if x.numel() < 2:
        return float("nan")

    if x.std() < 1e-6 or y.std() < 1e-6:
        return float("nan")

    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()

    return (x * y).mean().item()


def compute_pearson_uncertainty(probs, sigmas):
    """
    Pearson correlation between ground-truth noise level and predicted uncertainty.

    Args:
        probs: Tensor [N, C], softmax probabilities.
        sigmas: Tensor [N], per-sample mean sigma.

    Returns:
        float
    """
    confidences, _ = probs.max(dim=1)
    predicted_uncertainty = 1.0 - confidences

    return _pearson(sigmas, predicted_uncertainty)


def compute_pearson_uncertainty_correct_only(probs, labels, sigmas):
    """
    Pearson correlation between sigma and predicted uncertainty,
    restricted to correctly classified samples.

    Args:
        probs: Tensor [N, C], softmax probabilities.
        labels: Tensor [N].
        sigmas: Tensor [N].

    Returns:
        float
    """
    confidences, preds = probs.max(dim=1)
    predicted_uncertainty = 1.0 - confidences

    mask = preds == labels

    if mask.sum().item() == 0:
        return float("nan")

    return _pearson(sigmas[mask], predicted_uncertainty[mask])


def compute_spearman_uncertainty(probs, sigmas):
    """
    Spearman correlation between sigma and predicted uncertainty.
    Optional metric; not required for main paper tables.
    """
    confidences, _ = probs.max(dim=1)
    predicted_uncertainty = 1.0 - confidences

    sigma_rank = torch.argsort(torch.argsort(sigmas.detach().cpu()))
    uncertainty_rank = torch.argsort(torch.argsort(predicted_uncertainty.detach().cpu()))

    return _pearson(sigma_rank.float(), uncertainty_rank.float())


def compute_spearman_uncertainty_correct_only(probs, labels, sigmas):
    """
    Spearman correlation restricted to correctly classified samples.
    Optional metric; not required for main paper tables.
    """
    confidences, preds = probs.max(dim=1)
    predicted_uncertainty = 1.0 - confidences

    mask = preds == labels

    if mask.sum().item() == 0:
        return float("nan")

    sigma_rank = torch.argsort(torch.argsort(sigmas[mask].detach().cpu()))
    uncertainty_rank = torch.argsort(torch.argsort(predicted_uncertainty[mask].detach().cpu()))

    return _pearson(sigma_rank.float(), uncertainty_rank.float())