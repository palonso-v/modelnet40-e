import os
import time
import torch
import torch.optim as optim

from models import build_model, forward_model
from data_utils import load_data_modelnet40e, load_data_ScanObjectNNe

def get_num_classes(dataset):
    if dataset == "modelnet40":
        return 40
    if dataset == "ScanObjectNN":
        return 15
    raise ValueError(f"Unknown dataset: {dataset}")


def get_default_batch_size(model_name):
    if model_name == "PTv3":
        return 8
    return 32


def get_default_lr(model_name):
    if model_name == "PTv3":
        return 5e-4
    return 1e-3


def get_criterion(model_name):
    if model_name in ["PointNet", "PointNet2"]:
        return torch.nn.NLLLoss()
    return torch.nn.CrossEntropyLoss()


def prepare_labels(labels, dataset, device):
    labels = labels.to(device)
    if dataset == "modelnet40":
        return labels.squeeze(1).long()
    return labels.long()


def get_data_loaders(dataset, batch_size, device, noise_level):
    if dataset == "modelnet40":
        return load_data_modelnet40e(
            dataset,
            batch_size,
            device,
            noise_level=noise_level,
            shuffle=True,
        )

    if dataset == "ScanObjectNN":
        return load_data_ScanObjectNNe(
            dataset,
            batch_size,
            device,
            noise_level=noise_level,
            shuffle=True,
        )

    raise ValueError(f"Unknown dataset: {dataset}")


def train_one_epoch(model, train_loader, criterion, optimizer, config, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch, labels, sigmas, mus in train_loader:
        batch = batch.to(device)
        labels = prepare_labels(labels, config.dataset, device)

        optimizer.zero_grad()

        outputs = forward_model(model, batch, config.model, device)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def validate(model, test_loader, criterion, config, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch, labels, sigmas, mus in test_loader:
            batch = batch.to(device)
            labels = prepare_labels(labels, config.dataset, device)

            outputs = forward_model(model, batch, config.model, device)
            loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def save_checkpoint(model, optimizer, epoch, val_loss, val_acc, config):
    os.makedirs(config.save_dir, exist_ok=True)

    prefix = f"{config.dataset}e_{config.noise_level}_{config.model}"

    last_path = os.path.join(config.save_dir, f"{prefix}_last.pth")

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "dataset": config.dataset,
            "model": config.model,
            "noise_level": config.noise_level,
        },
        last_path,
    )

    periodic_path = None
    if (epoch + 1) % config.save_every == 0:
        periodic_path = os.path.join(
            config.save_dir,
            f"{prefix}_epoch{epoch + 1}.pth",
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "dataset": config.dataset,
                "model": config.model,
                "noise_level": config.noise_level,
            },
            periodic_path,
        )

    return last_path, periodic_path


def load_training_checkpoint(model, optimizer, checkpoint_path, device):
    if not checkpoint_path:
        raise ValueError("--resume was set, but --checkpoint is empty.")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint.get("epoch", 0)

    print(f"Resumed training from checkpoint: {checkpoint_path}")
    print(f"Starting at epoch: {start_epoch}")

    return start_epoch


def train_classification(config):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print("device =", device)

    num_classes = get_num_classes(config.dataset)
    batch_size = config.batch_size or get_default_batch_size(config.model)
    lr = config.lr or get_default_lr(config.model)

    _, train_loader, _, test_loader = get_data_loaders(
        dataset=config.dataset,
        batch_size=batch_size,
        device=device,
        noise_level=config.noise_level,
    )

    model = build_model(
        model_name=config.model,
        dataset=config.dataset,
        num_classes=num_classes,
        device=device,
    )

    criterion = get_criterion(config.model)

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=config.weight_decay,
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.step_size,
        gamma=config.gamma,
    )

    start_epoch = 0

    if config.resume:
        start_epoch = load_training_checkpoint(
            model=model,
            optimizer=optimizer,
            checkpoint_path=config.checkpoint,
            device=device,
        )

    print(f"Dataset: {config.dataset}")
    print(f"Model: {config.model}")
    print(f"Noise level: {config.noise_level}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {config.epochs}")
    print(f"Learning rate: {lr}")

    start_time = time.time()

    best_acc = 0.0

    for epoch in range(start_epoch, config.epochs):
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            config=config,
            device=device,
        )

        val_loss, val_acc = validate(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            config=config,
            device=device,
        )

        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(
                config.save_dir,
                f"{config.dataset}e_{config.noise_level}_{config.model}_best.pth",
            )
            os.makedirs(config.save_dir, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "dataset": config.dataset,
                    "model": config.model,
                    "noise_level": config.noise_level,
                },
                best_path,
            )

        last_path, periodic_path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            val_loss=val_loss,
            val_acc=val_acc,
            config=config,
        )

        print(
            f"Epoch {epoch + 1}/{config.epochs} | "
            f"train loss: {train_loss:.4f} | train acc: {train_acc:.4f} | "
            f"val loss: {val_loss:.4f} | val acc: {val_acc:.4f}"
        )

        if periodic_path is not None:
            print(f"Saved checkpoint: {periodic_path}")

    end_time = time.time()
    print(f"Training finished in {(end_time - start_time) / 60:.2f} minutes")
    print(f"Best validation accuracy: {best_acc:.4f}")

    return 0