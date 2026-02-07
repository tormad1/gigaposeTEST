import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

import src.megapose.utils.tensor_collection as tc
from src.dataloader.template import TemplateSet
from src.models.gigaPose import GigaPose
from src.models.matching import LocalSimilarity
from src.models.network.ae_net import AENet, descriptor_sizes
from src.models.network.ist_net import ISTNet, Regressor
from src.models.network.resnet import ResNet
from src.utils.crop import CropResizePad


NORM_MEAN = [0.48145466, 0.4578275, 0.40821073]
NORM_STD = [0.26862954, 0.26130258, 0.27577711]


class InferenceTransforms:
    def __init__(self, crop_size: int = 224):
        self.crop_transform = CropResizePad(target_size=crop_size)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(NORM_MEAN, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(NORM_STD, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std


class DummyOptimConfig:
    warm_up_steps = 1


def parse_args():
    p = argparse.ArgumentParser("RGB-only GigaPose golden test")
    p.add_argument("--weights", required=True, help="Path to .ckpt or state_dict file")
    p.add_argument("--root_dir", required=True, help="Dataset root that contains <dataset_name>/models/models_info.json")
    p.add_argument("--template_dir", required=True, help="Templates root (contains per-dataset folders)")
    p.add_argument("--dataset_name", required=True, help="Dataset name key used for template loading, e.g. lmo")
    p.add_argument("--label", type=int, required=True, help="Object id label (1-based for most datasets)")
    p.add_argument("--K", required=True, help="Camera intrinsics as fx,fy,cx,cy")
    p.add_argument("--output_dir", default="./outputs_golden")

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--crop_rgb", help="Path to pre-cropped RGB image")
    group.add_argument("--image_rgb", help="Path to full RGB image (requires --bbox)")

    p.add_argument("--bbox", help="x1,y1,x2,y2 in original image coordinates (required with --image_rgb)")
    p.add_argument("--crop_mask", help="Optional mask path aligned with --crop_rgb")

    p.add_argument("--num_templates", type=int, default=162)
    p.add_argument("--pose_name", default="object_poses/OBJECT_ID.npy")
    p.add_argument("--scale_factor", type=float, default=1.0)
    p.add_argument("--pose_distribution", default="all")
    p.add_argument("--level_templates", type=int, default=1)

    p.add_argument("--ae_model", default="dinov2_vitl14")
    p.add_argument("--max_batch_size", type=int, default=64)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--sim_threshold", type=float, default=0.5)
    p.add_argument("--patch_threshold", type=int, default=3)
    p.add_argument("--max_num_dets_per_forward", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_rgb(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def load_mask(path: str, h: int, w: int) -> torch.Tensor:
    if path is None:
        return torch.ones((h, w), dtype=torch.float32)
    m = Image.open(path).convert("L")
    arr = (np.array(m) > 0).astype(np.float32)
    return torch.from_numpy(arr)


def parse_k(k_str: str) -> torch.Tensor:
    fx, fy, cx, cy = [float(x.strip()) for x in k_str.split(",")]
    K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32)
    return K


def parse_bbox(bbox_str: str) -> torch.Tensor:
    xyxy = [int(x.strip()) for x in bbox_str.split(",")]
    if len(xyxy) != 4:
        raise ValueError("--bbox must be x1,y1,x2,y2")
    return torch.tensor(xyxy, dtype=torch.long)


def build_model(args) -> GigaPose:
    dinov2_model = torch.hub.load("facebookresearch/dinov2", args.ae_model)
    ae_net = AENet(
        model_name=args.ae_model,
        dinov2_model=dinov2_model,
        descriptor_size=descriptor_sizes[args.ae_model],
        max_batch_size=args.max_batch_size,
    )

    backbone = ResNet(
        config={
            "n_heads": 0,
            "input_dim": 3,
            "input_size": 256,
            "initial_dim": 128,
            "block_dims": [128, 192, 256, 512],
            "descriptor_size": 256,
        }
    )
    regressor = Regressor(
        descriptor_size=256,
        hidden_dim=256,
        use_tanh_act=True,
        normalize_output=True,
    )
    ist_net = ISTNet(
        model_name="resnet",
        backbone=backbone,
        regressor=regressor,
        max_batch_size=args.max_batch_size,
    )

    testing_metric = LocalSimilarity(
        k=args.top_k,
        sim_threshold=args.sim_threshold,
        patch_threshold=args.patch_threshold,
    )

    model = GigaPose(
        model_name="golden",
        ae_net=ae_net,
        ist_net=ist_net,
        training_loss={},
        testing_metric=testing_metric,
        optim_config=DummyOptimConfig(),
        log_interval=10**9,
        log_dir=args.output_dir,
        max_num_dets_per_forward=args.max_num_dets_per_forward,
        test_setting="detection",
    )
    return model


def load_checkpoint(model: GigaPose, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        cleaned[nk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")


def build_template_set(args, transforms):
    template_config = SimpleNamespace(
        dir=args.template_dir,
        level_templates=args.level_templates,
        pose_distribution=args.pose_distribution,
        scale_factor=args.scale_factor,
        num_templates=args.num_templates,
        pose_name=args.pose_name,
    )
    return TemplateSet(
        root_dir=args.root_dir,
        dataset_name=args.dataset_name,
        template_config=template_config,
        transforms=transforms,
    )


def build_target_inputs(args, transforms, device):
    if args.image_rgb is not None:
        if args.bbox is None:
            raise ValueError("--bbox is required when --image_rgb is used")
        full = load_rgb(args.image_rgb).unsqueeze(0)
        bbox = parse_bbox(args.bbox).unsqueeze(0)

        crop_out = transforms.crop_transform(bbox.to(full.device), full)
        crop_rgb = crop_out["images"][:, :3]
        M = crop_out["M"]

        mask_full = torch.ones((1, 1, full.shape[2], full.shape[3]), dtype=torch.float32)
        mask_out = transforms.crop_transform(bbox.to(mask_full.device), mask_full)
        crop_mask = mask_out["images"][:, 0]
    else:
        crop = load_rgb(args.crop_rgb).unsqueeze(0)
        crop_rgb = F.interpolate(crop, size=(224, 224), mode="bilinear", align_corners=False)
        if args.crop_mask is None:
            crop_mask = torch.ones((1, 224, 224), dtype=torch.float32)
        else:
            m = load_mask(args.crop_mask, crop_rgb.shape[-2], crop_rgb.shape[-1]).unsqueeze(0)
            crop_mask = F.interpolate(m.unsqueeze(1), size=(224, 224), mode="nearest")[:, 0]
        M = torch.eye(3).unsqueeze(0)

    tar_img = transforms.normalize(crop_rgb)
    tar_mask = (crop_mask > 0.5).float()
    tar_K = parse_k(args.K).unsqueeze(0)

    return tar_img.to(device), tar_mask.to(device), tar_K.to(device), M.to(device)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "predictions"), exist_ok=True)

    device = torch.device(args.device)
    transforms = InferenceTransforms(crop_size=224)

    model = build_model(args)
    load_checkpoint(model, args.weights)

    template_set = build_template_set(args, transforms)
    model.template_datasets = {args.dataset_name: template_set}
    model.test_dataset_name = args.dataset_name
    model.to(device)
    model.eval()

    tar_img, tar_mask, tar_K, tar_M = build_target_inputs(args, transforms, device)

    infos = pd.DataFrame(
        {
            "scene_id": [0],
            "view_id": [0],
            "label": [int(args.label)],
        }
    )
    test_infos = pd.DataFrame(
        {
            "obj_id": [int(args.label)],
            "inst_count": [1],
            "detection_time": [0.0],
        }
    )

    batch = tc.PandasTensorCollection(
        tar_img=tar_img,
        tar_mask=tar_mask,
        tar_K=tar_K,
        tar_M=tar_M,
        infos=infos,
        test_list=tc.PandasTensorCollection(infos=test_infos),
    )

    with torch.no_grad():
        model.eval_retrieval(batch=batch, idx_batch=0, dataset_name=args.dataset_name)

    out_npz = Path(args.output_dir) / "predictions" / "0.npz"
    if not out_npz.exists():
        raise RuntimeError(f"Expected output not found: {out_npz}")

    data = np.load(out_npz)
    print(f"Saved: {out_npz}")
    print(f"poses shape: {data['poses'].shape}")
    print(f"scores shape: {data['scores'].shape}")
    print("top-1 pose:\n", data["poses"][0, 0])


if __name__ == "__main__":
    main()
