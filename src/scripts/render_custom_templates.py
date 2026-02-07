import argparse
import multiprocessing as mp
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.lib3d.template_transform import get_obj_poses_from_template_level
from src.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser("Render custom templates for onboarding")
    p.add_argument("--root_dir", required=True, help="Root dataset dir containing <dataset_name>/models")
    p.add_argument("--dataset_name", required=True, help="Custom dataset name (folder under root_dir)")
    p.add_argument("--num_workers", type=int, default=max(1, os.cpu_count() // 2))
    p.add_argument("--num_gpus", type=int, default=1)
    p.add_argument("--level_templates", type=int, default=1)
    p.add_argument("--pose_distribution", default="all", choices=["all", "upper"])
    p.add_argument("--translation_scale", type=float, default=0.4, help="Template camera distance scale")
    p.add_argument("--disable_output", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def render_one(job):
    idx, cad_path, obj_pose_path, output_dir, num_gpus, disable_output = job

    if output_dir.exists() and not any(output_dir.glob("*.png")):
        pass
    elif output_dir.exists() and any(output_dir.glob("*.png")):
        # already rendered
        return True

    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_id = idx % max(1, num_gpus)
    cmd = [
        "blenderproc",
        "run",
        "./src/lib3d/blenderproc.py",
        str(cad_path),
        str(obj_pose_path),
        str(output_dir),
        str(gpu_id),
        "true" if disable_output else "false",
        "true",
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        logger.exception(f"Render failed for {cad_path}")
        return False

    num_png = len(list(output_dir.glob("*.png")))
    expected = len(np.load(obj_pose_path)) * 2  # rgb + depth
    ok = num_png == expected
    if not ok:
        logger.warning(f"Rendered {num_png}/{expected} images for {cad_path}")
    return ok


def main():
    args = parse_args()

    root_dir = Path(args.root_dir)
    dataset_name = args.dataset_name
    cad_dir = root_dir / dataset_name / "models"
    save_root = root_dir / "templates" / dataset_name
    pose_dir = save_root / "object_poses"

    if not cad_dir.exists():
        raise FileNotFoundError(f"CAD directory not found: {cad_dir}")

    if args.overwrite and save_root.exists():
        shutil.rmtree(save_root)

    save_root.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)

    template_poses = get_obj_poses_from_template_level(
        level=args.level_templates,
        pose_distribution=args.pose_distribution,
    )
    template_poses[:, :3, 3] *= args.translation_scale

    cad_paths = sorted(list(cad_dir.glob("*.ply")) + list(cad_dir.glob("*.obj")))
    if len(cad_paths) == 0:
        raise RuntimeError(f"No CAD files found in {cad_dir} (.ply/.obj)")

    jobs = []
    for idx, cad_path in enumerate(cad_paths):
        stem = cad_path.stem
        if stem.startswith("obj_"):
            obj_id = int(stem.split("obj_")[-1])
        else:
            obj_id = int(stem)

        out_dir = save_root / f"{obj_id:06d}"
        pose_path = pose_dir / f"{obj_id:06d}.npy"
        np.save(pose_path, template_poses)
        jobs.append((idx, cad_path, pose_path, out_dir, args.num_gpus, args.disable_output))

    logger.info(f"Rendering {len(jobs)} objects from {cad_dir}")
    logger.info(f"Saving templates to {save_root}")

    t0 = time.time()
    with mp.Pool(processes=args.num_workers) as pool:
        values = list(tqdm(pool.imap_unordered(render_one, jobs), total=len(jobs)))

    num_ok = sum(1 for v in values if v)
    dt = time.time() - t0
    logger.info(f"Finished: {num_ok}/{len(jobs)} objects rendered correctly in {dt:.1f}s")


if __name__ == "__main__":
    main()
