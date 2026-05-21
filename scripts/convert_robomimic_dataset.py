"""
Convert RoboMimic HDF5 datasets to OAT zarr format.
"""

if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import pathlib
import shutil

import click
import h5py
import zarr

from oat.common.input_util import wait_user_input
from oat.env.robomimic.dataset_conversion import (
    DEFAULT_REQUIRED_OBS_KEYS,
    convert_robomimic_hdf5_to_zarr_streaming,
    infer_task_name,
)


@click.command()
@click.option("--root_dir", type=str, default="data/robomimic")
@click.option("--hdf5_dir_name", type=str, default="hdf5_datasets")
@click.option("-n", "--num_sample_demo", type=int, default=None)
@click.option("--seed", type=int, default=42)
@click.option("--compression_level", type=int, default=5, show_default=True)
@click.option("--chunk_size", type=int, default=1024, show_default=True)
@click.option("--verify_sample_size", type=int, default=128, show_default=True)
@click.option(
    "--required_obs_key",
    type=str,
    multiple=True,
    default=DEFAULT_REQUIRED_OBS_KEYS,
    help="Required obs keys in each demo['obs']. Repeat this option to override defaults.",
)
def convert_all_robomimic_datasets(
    root_dir: str,
    hdf5_dir_name: str,
    num_sample_demo: int,
    seed: int,
    compression_level: int,
    chunk_size: int,
    verify_sample_size: int,
    required_obs_key: tuple[str, ...],
):
    hdf5_root = pathlib.Path(root_dir) / hdf5_dir_name
    hdf5_paths = sorted(hdf5_root.glob("*.hdf5"))
    if not hdf5_paths:
        raise FileNotFoundError(f"No .hdf5 files found in {hdf5_root}")

    for hdf5_path in hdf5_paths:
        hdf5_path_str = str(hdf5_path)
        print(f"Converting {hdf5_path_str}...")

        task_name = infer_task_name(hdf5_path_str)
        with h5py.File(hdf5_path_str, "r") as f:
            all_demo_keys = [k for k in f["data"].keys() if k.startswith("demo_")]
            expected_n_demo = len(all_demo_keys) if num_sample_demo is None else min(num_sample_demo, len(all_demo_keys))
        save_path = pathlib.Path(root_dir) / f"{task_name}_N{expected_n_demo}.zarr"

        if save_path.exists():
            keypress = wait_user_input(
                valid_input=lambda key: key in ["", "y", "n"],
                prompt=f"{save_path} already exists. Overwrite? [y/`n`]: ",
                default="n",
            )
            if keypress == "n":
                print("Skip existing export.")
                continue
            shutil.rmtree(save_path)

        compressor = zarr.Blosc(cname="zstd", clevel=compression_level, shuffle=1)
        stats = convert_robomimic_hdf5_to_zarr_streaming(
            hdf5_path=hdf5_path_str,
            zarr_path=str(save_path),
            sample_ndemo=num_sample_demo,
            required_obs_keys=required_obs_key,
            seed=seed,
            compressor=compressor,
            chunk_size=chunk_size,
            verify_sample_size=verify_sample_size,
        )
        print(
            f"Verification passed: episodes={stats['episodes']}, "
            f"steps={stats['steps']}, action_dim={stats['action_dim']}"
        )

    print("All done.")


if __name__ == "__main__":
    convert_all_robomimic_datasets()
