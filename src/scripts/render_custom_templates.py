import glob
import multiprocessing
import os
import subprocess
import time
from functools import partial
from pathlib import Path

import hydra
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from src.lib3d.template_transform import get_obj_poses_from_template_level
from src.utils.logging import get_logger

logger = get_logger(__name__)


def parse_object_id(path: Path, fallback_idx=None):
    stem = path.stem
    if stem.startswith("obj_") and stem[4:].isdigit():
        return int(stem[4:])
    if stem.isdigit():
        return int(stem)
    if fallback_idx is not None:
        return fallback_idx + 1
    raise ValueError(
        f"Cannot parse object id from '{stem}'. Use obj_000001.* or numeric file names."
    )


def convert_fbx_to_obj(fbx_path: Path, obj_path: Path):
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "blender",
        "-b",
        "--python",
        "./src/lib3d/convert_fbx_to_obj.py",
        "--",
        str(fbx_path),
        str(obj_path),
    ]
    subprocess.run(cmd, check=True)
    return obj_path


def call_render(
    idx_obj,
    list_cad_path,
    list_pose_path,
    list_output_dir,
    disable_output,
    num_gpus,
    use_blenderProc,
):
    output_dir = list_output_dir[idx_obj]
    cad_path = list_cad_path[idx_obj]
    pose_path = list_pose_path[idx_obj]

    if os.path.exists(output_dir):
        os.system(f"rm -r {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    gpus_device = idx_obj % max(1, num_gpus)
    if use_blenderProc:
        command = (
            f"blenderproc run ./src/lib3d/blenderproc.py "
            f"{cad_path} {pose_path} {output_dir} {gpus_device}"
        )
    else:
        command = (
            f"python -m src.custom_megapose.call_panda3d "
            f"{cad_path} {pose_path} {output_dir} {gpus_device}"
        )

    command += " true" if disable_output else " false"
    command += " true"  # scale translation to meter
    os.system(command)

    num_images = len(glob.glob(f"{output_dir}/*.png"))
    expected = len(np.load(pose_path)) * 2  # rgb + depth
    if num_images == expected:
        return True

    logger.info(f"Found only {num_images}/{expected} for {cad_path} {pose_path}")
    return False


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def render(cfg) -> None:
    num_gpus = 4
    disable_output = True

    OmegaConf.set_struct(cfg, False)
    root_dir = Path(cfg.data.test.root_dir)
    root_save_dir = root_dir / "templates"
    template_poses = get_obj_poses_from_template_level(level=1, pose_distribution="all")
    template_poses[:, :3, 3] *= 0.4  # zoom to object
    dataset_name = cfg.custom_dataset_name

    dataset_save_dir = root_save_dir / dataset_name
    logger.info(f"Rendering templates for {dataset_name}")
    dataset_save_dir.mkdir(parents=True, exist_ok=True)

    obj_pose_dir = dataset_save_dir / "object_poses"
    obj_pose_dir.mkdir(parents=True, exist_ok=True)

    converted_dir = dataset_save_dir / "_converted"
    converted_dir.mkdir(parents=True, exist_ok=True)

    cad_dir = root_dir / dataset_name / "models"
    cad_paths = sorted(
        list(cad_dir.glob("*.ply")) + list(cad_dir.glob("*.obj")) + list(cad_dir.glob("*.fbx"))
    )
    logger.info(f"Found {len(cad_paths)} objects in {cad_dir}")

    final_cad_paths = []
    pose_paths = []
    output_dirs = []

    for idx, cad_path in enumerate(cad_paths):
        object_id = parse_object_id(cad_path, fallback_idx=idx)

        if cad_path.suffix.lower() == ".fbx":
            converted_obj = converted_dir / f"obj_{object_id:06d}.obj"
            logger.info(f"Converting FBX -> OBJ: {cad_path} -> {converted_obj}")
            cad_path = convert_fbx_to_obj(cad_path, converted_obj)

        final_cad_paths.append(cad_path)
        output_dirs.append(dataset_save_dir / f"{object_id:06d}")

        pose_path = obj_pose_dir / f"{object_id:06d}.npy"
        np.save(pose_path, template_poses)
        pose_paths.append(pose_path)

    logger.info(f"Start rendering for {len(final_cad_paths)} objects")
    start_time = time.time()

    call_render_ = partial(
        call_render,
        list_cad_path=final_cad_paths,
        list_pose_path=pose_paths,
        list_output_dir=output_dirs,
        disable_output=disable_output,
        num_gpus=num_gpus,
        use_blenderProc=True if dataset_name in ["tless", "itodd"] else False,
    )

    with multiprocessing.Pool(processes=int(cfg.machine.num_workers)) as pool:
        values = list(
            tqdm(
                pool.imap_unordered(call_render_, range(len(final_cad_paths))),
                total=len(final_cad_paths),
            )
        )

    correct_values = [val for val in values if val]
    finish_time = time.time()
    logger.info(f"Finished for {len(correct_values)}/{len(final_cad_paths)} objects")
    logger.info(f"Total time {len(final_cad_paths)}: {finish_time - start_time}")


if __name__ == "__main__":
    render()
