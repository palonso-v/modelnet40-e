import os
import h5py
import numpy as np
from sklearn.neighbors import NearestNeighbors
from utils import getfile, load_h5_ModelNet, load_h5_ScanObjectNN

SEVERITIES = ["light", "moderate", "heavy"]
MODELNET_SPLITS = ["train", "test"]
SCANOBJECTNN_SPLITS = ["training", "test"]

SCANOBJECTNN_FILE_TEMPLATE = "{split}_objectdataset_augmentedrot_scale75.h5"

BOUNDING_BOX = ((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))


def sample_params(severity, rng):
    if severity == "light":
        a = rng.uniform(0.002, 0.004)
        b = rng.uniform(0.0005, 0.0015)
        c = rng.uniform(1.0, 2.0)
        bias_k = rng.uniform(0.0025, 0.0075)
        outlier_prob = rng.uniform(0.005, 0.015)

    elif severity == "moderate":
        a = rng.uniform(0.003, 0.007)
        b = rng.uniform(0.001, 0.003)
        c = rng.uniform(1.5, 2.5)
        bias_k = rng.uniform(0.005, 0.015)
        outlier_prob = rng.uniform(0.01, 0.03)

    elif severity == "heavy":
        a = rng.uniform(0.005, 0.015)
        b = rng.uniform(0.002, 0.004)
        c = rng.uniform(2.0, 4.0)
        bias_k = rng.uniform(0.01, 0.025)
        outlier_prob = rng.uniform(0.04, 0.08)

    else:
        raise ValueError(f"Unknown severity: {severity}")

    return a, b, c, bias_k, outlier_prob


def sample_sensor_position(rng, radius):
    azimuth = rng.uniform(0, 2 * np.pi)
    elevation = rng.uniform(-np.pi / 4, np.pi / 4)

    x = radius * np.cos(elevation) * np.cos(azimuth)
    y = radius * np.cos(elevation) * np.sin(azimuth)
    z = radius * np.sin(elevation)

    return np.array([x, y, z], dtype=np.float32)


def estimate_normals(points, normal_k):
    nbrs = NearestNeighbors(n_neighbors=normal_k, algorithm="kd_tree").fit(points)
    _, indices = nbrs.kneighbors(points)

    normals = np.zeros_like(points, dtype=np.float32)

    for i in range(points.shape[0]):
        neighbors = points[indices[i]]
        cov = np.cov(neighbors.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normals[i] = eigvecs[:, 0]

    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return normals


def simulate_lidar_noise(
    points,
    sensor_position,
    a,
    b,
    c,
    bias_k,
    outlier_prob,
    normal_k=10,
    bounding_box=BOUNDING_BOX,
    rng=None,
):
    """
    Simulate LiDAR-like noise.

    This function intentionally preserves the original implementation used
    to generate the reported benchmark results.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_points = points.shape[0]
    noisy_points = np.zeros_like(points)

    vec_to_point = points - sensor_position[None, :]
    ranges = np.linalg.norm(vec_to_point, axis=1)

    normals = estimate_normals(points, normal_k=normal_k)

    vec_norm = vec_to_point / ranges[:, None]
    cos_theta = np.sum(vec_norm * normals, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(np.abs(cos_theta))

    # IMPORTANT:
    # Keep this formula unchanged for reproducibility with the reported results.
    sigma_range = a + b * ranges
    sigma_angle = 1 + c * (1 - np.cos(theta))
    sigma = sigma_range * sigma_angle

    mu = bias_k * (1 - np.cos(theta))

    epsilons = rng.normal(loc=mu, scale=sigma)
    delta = epsilons[:, None] * vec_norm
    noisy_points = points + delta

    if bounding_box is not None:
        outlier_mask = rng.random(n_points) < outlier_prob
        for dim in range(3):
            noisy_points[outlier_mask, dim] = rng.uniform(
                bounding_box[dim][0],
                bounding_box[dim][1],
                size=np.sum(outlier_mask),
            )

    return noisy_points.astype(np.float32), sigma.astype(np.float32), mu.astype(np.float32)


def save_h5_with_metadata(filename, data, label, sigma, mu):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with h5py.File(filename, "w") as f:
        f.create_dataset("data", data=data, compression="gzip",
                         compression_opts=4, dtype="float32")
        f.create_dataset("label", data=label, compression="gzip",
                         compression_opts=1, dtype="int64")
        f.create_dataset("sigma", data=sigma, compression="gzip",
                         compression_opts=4, dtype="float32")
        f.create_dataset("mu", data=mu, compression="gzip",
                         compression_opts=4, dtype="float32")


def generate_file(data, label, output_path, severity, config, rng):
    noisy_data = np.zeros_like(data, dtype=np.float32)
    sigma_all = np.zeros((data.shape[0], data.shape[1]), dtype=np.float32)
    mu_all = np.zeros((data.shape[0], data.shape[1]), dtype=np.float32)

    for j, pc in enumerate(data):
        if j % 100 == 0:
            print(f"  sample {j}/{len(data)}")

        sensor_position = sample_sensor_position(
            rng=rng,
            radius=config.sensor_radius,
        )

        a, b, c, bias_k, outlier_prob = sample_params(severity, rng)

        noisy_pc, sigma, mu = simulate_lidar_noise(
            points=pc,
            sensor_position=sensor_position,
            a=a,
            b=b,
            c=c,
            bias_k=bias_k,
            outlier_prob=outlier_prob,
            normal_k=config.normal_k,
            bounding_box=BOUNDING_BOX,
            rng=rng,
        )

        noisy_data[j] = noisy_pc
        sigma_all[j] = sigma
        mu_all[j] = mu

    save_h5_with_metadata(
        filename=output_path,
        data=noisy_data,
        label=label,
        sigma=sigma_all,
        mu=mu_all,
    )

    print(f"Saved: {output_path}")
    print(f"Mean sigma: {sigma_all.mean():.6f}")


def get_requested_severities(config):
    if config.severity == "all":
        return SEVERITIES
    return [config.severity]


def get_requested_splits(config):
    if config.dataset == "modelnet40":
        valid_splits = MODELNET_SPLITS
    elif config.dataset == "ScanObjectNN":
        valid_splits = SCANOBJECTNN_SPLITS
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    if config.split == "all":
        return valid_splits

    if config.split not in valid_splits:
        raise ValueError(
            f"Invalid split '{config.split}' for {config.dataset}. "
            f"Valid options are: {valid_splits}"
        )

    return [config.split]


def generate_modelnet40e(config, rng):
    severities = get_requested_severities(config)
    splits = get_requested_splits(config)

    for split in splits:
        file_list_path = os.path.join(config.input_root, f"{split}_files.txt")
        file_list = getfile(file_list_path)

        for severity in severities:
            for file_name in file_list:
                input_path = os.path.join(config.input_root, file_name)
                output_path = os.path.join(
                    config.output_root,
                    severity,
                    split,
                    file_name,
                )

                print(f"\nProcessing: {input_path}")
                print(f"Severity: {severity}")

                data, label = load_h5_ModelNet(input_path)

                generate_file(
                    data=data,
                    label=label,
                    output_path=output_path,
                    severity=severity,
                    config=config,
                    rng=rng,
                )


def generate_scanobjectnne(config, rng):
    severities = get_requested_severities(config)
    splits = get_requested_splits(config)

    for split in splits:
        file_name = SCANOBJECTNN_FILE_TEMPLATE.format(split=split)
        input_path = os.path.join(config.input_root, file_name)

        print(f"\nLoading: {input_path}")
        data, label = load_h5_ScanObjectNN(input_path)

        for severity in severities:
            output_path = os.path.join(
                config.output_root,
                severity,
                split,
                file_name,
            )

            print(f"\nProcessing: {input_path}")
            print(f"Severity: {severity}")

            generate_file(
                data=data,
                label=label,
                output_path=output_path,
                severity=severity,
                config=config,
                rng=rng,
            )


def generate_noisy_dataset(config):
    rng = np.random.default_rng(config.seed)

    print(f"Dataset: {config.dataset}")
    print(f"Input root: {config.input_root}")
    print(f"Output root: {config.output_root}")
    print(f"Severity: {config.severity}")
    print(f"Split: {config.split}")
    print(f"Seed: {config.seed}")

    if config.dataset == "modelnet40":
        generate_modelnet40e(config, rng)

    elif config.dataset == "ScanObjectNN":
        generate_scanobjectnne(config, rng)

    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")