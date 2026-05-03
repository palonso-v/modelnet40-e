import argparse
from utils import set_seed
from trainer import train_classification


def get_args():
    parser = argparse.ArgumentParser(
        description="Train point cloud classifiers on ModelNet40-E or ScanObjectNN-E"
    )

    parser.add_argument("--dataset", type=str, default="modelnet40",
                        choices=["modelnet40", "ScanObjectNN"])
    parser.add_argument("--model", type=str, required=True,
                        choices=["PointNet", "PointNet2", "DGCNN", "PointMLP", "CurveNet", "SimpleView", "PTv3"])
    parser.add_argument("--noise_level", type=str, default="none",
                        choices=["none", "light", "moderate", "heavy"])

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--step_size", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.7)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_every", type=int, default=10)

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="")

    return parser.parse_args()


if __name__ == "__main__":
    config = get_args()
    set_seed(config.seed)
    train_classification(config)