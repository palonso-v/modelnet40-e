import torch

from .pointnet import PointNetCls
from .pointnet2 import PointNet2_cls_msg
from .dgcnn import DGCNN
from .pointmlp import pointMLP
from .curvenet import CurveNet
from .simpleview import MVModel

# Optional:
# from .ptv3 import PointTransformerV3, PointCloudTransformerAE_fullmodel_cls_false


class DGCNNArgs:
    def __init__(self):
        self.k = 20
        self.emb_dims = 1024
        self.dropout = 0.5
        self.leaky_relu = 1


def build_model(model_name, dataset, num_classes, device):
    if model_name == "PointNet":
        model = PointNetCls(k=num_classes)

    elif model_name == "PointNet2":
        model = PointNet2_cls_msg(num_class=num_classes, normal_channel=False)

    elif model_name == "DGCNN":
        model = DGCNN(DGCNNArgs(), output_channels=num_classes)

    elif model_name == "PointMLP":
        model = pointMLP(num_classes=num_classes)

    elif model_name == "CurveNet":
        model = CurveNet(num_classes=num_classes)

    elif model_name == "SimpleView":
        model = MVModel(
            task="cls",
            dataset=dataset,
            backbone="resnet18",
            feat_size=64,
        )

    elif model_name == "PTv3":
        raise NotImplementedError(
            "PTv3 support should be added locally in models/__init__.py."
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(device)


def forward_model(model, batch, model_name, device):
    if model_name == "PointNet":
        batch = batch.permute(0, 2, 1)
        outputs, _, _ = model(batch)
        return outputs

    if model_name == "PointNet2":
        batch = batch.permute(0, 2, 1)
        outputs, _ = model(batch)
        return outputs

    if model_name in ["DGCNN", "PointMLP", "CurveNet"]:
        batch = batch.permute(0, 2, 1)
        return model(batch)

    if model_name == "SimpleView":
        return model(batch)["logit"]

    if model_name == "PTv3":
        batch = batch.to(device)
        b, n, _ = batch.shape
        pointcloud = batch.reshape(b * n, 3)
        batch_indices = torch.arange(b, device=device).unsqueeze(1).repeat(1, n).reshape(-1)

        data_dict = {
            "coord": pointcloud,
            "batch": batch_indices,
            "feat": pointcloud,
            "grid_size": torch.tensor(0.01, device=device),
        }

        return model(data_dict)

    raise ValueError(f"Unknown model: {model_name}")