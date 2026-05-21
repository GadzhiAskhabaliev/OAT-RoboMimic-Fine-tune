import pathlib
from typing import Dict, List, Optional, Sequence

import h5py
import numpy as np
import tqdm
import zarr

from oat.common.replay_buffer import ReplayBuffer


OBS_KEY_MAPPING = {
    "agentview_image": "agentview_rgb",
    "robot0_eye_in_hand_image": "robot0_eye_in_hand_rgb",
    "robot0_eef_pos": "robot0_eef_pos",
    "robot0_eef_quat": "robot0_eef_quat",
    "robot0_gripper_qpos": "robot0_gripper_qpos",
    "robot0_joint_pos": "robot0_joint_pos",
}

DEFAULT_REQUIRED_OBS_KEYS = (
    "agentview_image",
    "robot0_eye_in_hand_image",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)


def _sorted_demo_keys(data_group: h5py.Group) -> List[str]:
    def demo_sort_key(name: str):
        if name.startswith("demo_"):
            suffix = name[len("demo_") :]
            if suffix.isdigit():
                return (0, int(suffix))
        return (1, name)

    return sorted([k for k in data_group.keys() if k.startswith("demo_")], key=demo_sort_key)


def _to_uint8_image(images: np.ndarray) -> np.ndarray:
    if images.dtype == np.uint8:
        return images
    if np.issubdtype(images.dtype, np.floating):
        maxv = np.nanmax(images)
        if maxv <= 1.0:
            images = images * 255.0
    images = np.clip(images, 0.0, 255.0)
    return images.astype(np.uint8)


def infer_task_name(hdf5_path: str) -> str:
    stem = pathlib.Path(hdf5_path).stem.lower()
    for task_name in ("lift", "can", "square"):
        if task_name in stem:
            return task_name
    return stem


def _select_demo_keys(
    data_group: h5py.Group,
    sample_ndemo: Optional[int],
    seed: int,
) -> List[str]:
    demo_keys = _sorted_demo_keys(data_group)
    if not demo_keys:
        raise ValueError("No demos found under 'data'.")
    if sample_ndemo is None:
        return demo_keys
    sample_ndemo = min(sample_ndemo, len(demo_keys))
    rng = np.random.default_rng(seed)
    selected_indices = rng.choice(len(demo_keys), sample_ndemo, replace=False)
    return [demo_keys[i] for i in selected_indices]


def _extract_episode(
    demo: h5py.Group,
    required_obs_keys: Sequence[str],
    expected_action_dim: Optional[int],
) -> tuple[Dict[str, np.ndarray], int, int]:
    if "actions" not in demo:
        raise KeyError("Demo missing 'actions' dataset.")
    if "obs" not in demo:
        raise KeyError("Demo missing 'obs' group.")

    action = demo["actions"][:].astype(np.float32)
    if action.ndim != 2:
        raise ValueError(f"Action shape {action.shape} should be [T, A].")
    action_dim = action.shape[-1]
    if expected_action_dim is not None and action_dim != expected_action_dim:
        raise ValueError(
            f"Inconsistent action dim: expected {expected_action_dim}, got {action_dim}."
        )
    if action_dim != 7:
        raise ValueError(f"Expected 7D actions for OAT policy, got {action_dim}.")

    obs_group = demo["obs"]
    missing_obs_keys = [k for k in required_obs_keys if k not in obs_group]
    if missing_obs_keys:
        raise KeyError(f"Missing required obs keys: {missing_obs_keys}")

    episode: Dict[str, np.ndarray] = {"action": action}
    for src_key, dst_key in OBS_KEY_MAPPING.items():
        if src_key not in obs_group:
            continue
        values = obs_group[src_key][:]
        if src_key.endswith("_image"):
            values = _to_uint8_image(values)
        elif np.issubdtype(values.dtype, np.floating):
            values = values.astype(np.float32)
        episode[dst_key] = values
    return episode, action_dim, len(action)


def convert_robomimic_hdf5_to_zarr(
    hdf5_path: str,
    sample_ndemo: Optional[int] = None,
    required_obs_keys: Sequence[str] = DEFAULT_REQUIRED_OBS_KEYS,
    seed: int = 42,
) -> ReplayBuffer:
    replay_buffer = ReplayBuffer.create_empty_zarr()

    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            raise KeyError(f"{hdf5_path} does not contain 'data' group.")
        data_group = f["data"]
        selected_demo_keys = _select_demo_keys(data_group, sample_ndemo=sample_ndemo, seed=seed)

        action_dim: Optional[int] = None
        total_steps = 0
        for demo_key in tqdm.tqdm(selected_demo_keys, desc="Converting RoboMimic Dataset"):
            demo = data_group[demo_key]
            episode, action_dim, episode_steps = _extract_episode(
                demo=demo,
                required_obs_keys=required_obs_keys,
                expected_action_dim=action_dim,
            )
            replay_buffer.add_episode(episode)
            total_steps += episode_steps

    print("-" * 50)
    print(
        f"Task: {infer_task_name(hdf5_path)} | "
        f"Episodes: {replay_buffer.n_episodes} | "
        f"Steps: {total_steps} | Action dim: {action_dim}"
    )
    print(replay_buffer)
    return replay_buffer


def convert_robomimic_hdf5_to_zarr_streaming(
    hdf5_path: str,
    zarr_path: str,
    sample_ndemo: Optional[int] = None,
    required_obs_keys: Sequence[str] = DEFAULT_REQUIRED_OBS_KEYS,
    seed: int = 42,
    compressor: Optional[zarr.Blosc] = None,
    chunk_size: int = 1024,
    verify_sample_size: int = 128,
) -> Dict[str, int]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if verify_sample_size < 0:
        raise ValueError(f"verify_sample_size must be >= 0, got {verify_sample_size}")
    if compressor is None:
        compressor = zarr.Blosc(cname="zstd", clevel=5, shuffle=1)

    replay_buffer = ReplayBuffer.create_empty_zarr(
        storage=zarr.DirectoryStore(zarr_path),
    )
    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            raise KeyError(f"{hdf5_path} does not contain 'data' group.")
        data_group = f["data"]
        selected_demo_keys = _select_demo_keys(data_group, sample_ndemo=sample_ndemo, seed=seed)

        action_dim: Optional[int] = None
        total_steps = 0
        chunks: Optional[Dict[str, tuple]] = None
        for demo_key in tqdm.tqdm(selected_demo_keys, desc="Streaming RoboMimic to Zarr"):
            demo = data_group[demo_key]
            episode, action_dim, episode_steps = _extract_episode(
                demo=demo,
                required_obs_keys=required_obs_keys,
                expected_action_dim=action_dim,
            )
            if chunks is None:
                chunks = {
                    key: (chunk_size,) + value.shape[1:]
                    for key, value in episode.items()
                }
            replay_buffer.add_episode(episode, chunks=chunks, compressors=compressor)
            total_steps += episode_steps

    # Lightweight post-check to catch incomplete writes.
    zgroup = zarr.open(zarr_path, mode="r")
    if "data" not in zgroup or "meta" not in zgroup:
        raise RuntimeError(f"Invalid zarr layout at {zarr_path}: expected data/meta groups.")
    if "action" not in zgroup["data"]:
        raise RuntimeError(f"Invalid zarr layout at {zarr_path}: missing data/action.")
    if "episode_ends" not in zgroup["meta"]:
        raise RuntimeError(f"Invalid zarr layout at {zarr_path}: missing meta/episode_ends.")
    if int(zgroup["meta"]["episode_ends"][-1]) != int(zgroup["data"]["action"].shape[0]):
        raise RuntimeError("Zarr integrity check failed: episode_ends final index != action length.")
    action_arr = zgroup["data"]["action"]
    total_action_steps = int(action_arr.shape[0])
    if total_action_steps == 0:
        raise RuntimeError("Zarr integrity check failed: no actions were written.")
    if verify_sample_size > 0:
        sample_n = min(verify_sample_size, total_action_steps)
        # evenly-spaced deterministic sample to avoid RNG-related reproducibility issues
        sample_idx = np.linspace(0, total_action_steps - 1, sample_n, dtype=np.int64)
        sample_actions = action_arr.get_orthogonal_selection((sample_idx, slice(None)))
        if not np.all(np.isfinite(sample_actions)):
            raise RuntimeError("Zarr integrity check failed: sampled action values contain NaN/Inf.")

    stats = {
        "episodes": int(replay_buffer.n_episodes),
        "steps": int(total_steps),
        "action_dim": int(zgroup["data"]["action"].shape[-1]),
    }
    print("-" * 50)
    print(
        f"Task: {infer_task_name(hdf5_path)} | "
        f"Episodes: {stats['episodes']} | Steps: {stats['steps']} | Action dim: {stats['action_dim']}"
    )
    print(f"Saved to {zarr_path}")
    return stats
