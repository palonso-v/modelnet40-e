import argparse
from utils import set_seed
from dataset_generator import generate_noisy_dataset


def get_args():
    parser = argparse.ArgumentParser(
        description="Generate ModelNet40-E or ScanObjectNN-E with LiDAR-like noise"
    )

    parser.add_argument("--dataset", type=str, required=True,
                        choices=["modelnet40", "ScanObjectNN"])
    parser.add_argument("--input_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)

    parser.add_argument("--severity", type=str, default="all",
                        choices=["light", "moderate", "heavy", "all"])
    parser.add_argument("--split", type=str, default="all",
                        help="Use 'all' to generate all splits.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sensor_radius", type=float, default=2.0)
    parser.add_argument("--normal_k", type=int, default=10)

    return parser.parse_args()


if __name__ == "__main__":
    config = get_args()
    set_seed(config.seed)
    generate_noisy_dataset(config)