import argparse
from utils import set_seed
from evaluator import evaluate_classification


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate point cloud classifiers on ModelNet40-E or ScanObjectNN-E"
    )

    parser.add_argument("--dataset", type=str, default="modelnet40",
                        choices=["modelnet40", "ScanObjectNN"])
    parser.add_argument("--model", type=str, required=True,
                        choices=["PointNet", "PointNet2", "DGCNN", "PointMLP", "CurveNet", "SimpleView", "PTv3"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--noise_level", type=str, default="all",
                        choices=["none", "light", "moderate", "heavy", "all"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_bins", type=int, default=10)

    return parser.parse_args()


if __name__ == "__main__":
    config = get_args()
    set_seed(config.seed)
    evaluate_classification(config)