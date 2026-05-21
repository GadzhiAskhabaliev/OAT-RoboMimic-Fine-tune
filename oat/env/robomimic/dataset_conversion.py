import pathlib
from typing import Dict, List, Optional, Sequence

import h5py
import numpy as np
import tqdm

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


def convert_robomimic_hdf5_to_zarr(
    hdf5_path: str,
    sample_ndemo: Optional[int] = None,
    required_obs_keys: Sequence[str] = DEFAULT_REQUIRED_OBS_KEYS,
    seed: int = 42,
) -> ReplayBuffer:
    replay_buffer = ReplayBuffer.create_empty_zarr()
    rng = np.random.default_rng(seed)

    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            raise KeyError(f"{hdf5_path} does not contain 'data' group.")
        data_group = f["data"]
        demo_keys = _sorted_demo_keys(data_group)
        if not demo_keys:
            raise ValueError(f"No demos found under 'data' in {hdf5_path}.")

        if sample_ndemo is None:
            selected_demo_keys = demo_keys
        else:
            sample_ndemo = min(sample_ndemo, len(demo_keys))
            selected_indices = rng.choice(len(demo_keys), sample_ndemo, replace=False)
            selected_demo_keys = [demo_keys[i] for i in selected_indices]

        action_dim: Optional[int] = None
        total_steps = 0
        for demo_key in tqdm.tqdm(selected_demo_keys, desc="Converting RoboMimic Dataset"):
            demo = data_group[demo_key]
            if "actions" not in demo:
                raise KeyError(f"Demo '{demo_key}' missing 'actions' dataset.")
            if "obs" not in demo:
                raise KeyError(f"Demo '{demo_key}' missing 'obs' group.")

            action = demo["actions"][:].astype(np.float32)
            if action.ndim != 2:
                raise ValueError(f"Demo '{demo_key}' action shape {action.shape} should be [T, A].")
            if action_dim is None:
                action_dim = action.shape[-1]
            elif action.shape[-1] != action_dim:
                raise ValueError(
                    f"Inconsistent action dim in '{demo_key}': expected {action_dim}, got {action.shape[-1]}."
                )
            if action_dim != 7:
                raise ValueError(f"Expected 7D actions for OAT policy, got {action_dim}.")

            obs_group = demo["obs"]
            missing_obs_keys = [k for k in required_obs_keys if k not in obs_group]
            if missing_obs_keys:
                raise KeyError(f"Demo '{demo_key}' missing required obs keys: {missing_obs_keys}")

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

            replay_buffer.add_episode(episode)
            total_steps += len(action)

    print("-" * 50)
    print(
        f"Task: {infer_task_name(hdf5_path)} | "
        f"Episodes: {replay_buffer.n_episodes} | "
        f"Steps: {total_steps} | Action dim: {action_dim}"
    )
    print(replay_buffer)
    return replay_buffer
