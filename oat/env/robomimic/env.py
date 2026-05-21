import numpy as np
import gymnasium
import robosuite as suite

from robosuite.controllers import load_controller_config
from typing import Dict, List, Optional


TASK_NAME_TO_ROBOSUITE_ENV = {
    "lift": "Lift",
    "can": "Can",
    "square": "Square",
}


class RoboMimicEnv(gymnasium.Env):
    def __init__(
        self,
        task_name: str,
        image_size: int = 84,
        seed: int = 42,
        robot: str = "Panda",
        camera_names: List[str] = ["agentview", "robot0_eye_in_hand"],
        state_ports: List[str] = [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ],
        video_camera: str = "agentview",
        video_resolution: int = 512,
        max_episode_steps: int = 400,
        control_freq: int = 20,
        enable_render: bool = True,
    ):
        super().__init__()
        task_key = task_name.lower()
        if task_key not in TASK_NAME_TO_ROBOSUITE_ENV:
            raise ValueError(f"Unsupported RoboMimic task '{task_name}'. Supported: {list(TASK_NAME_TO_ROBOSUITE_ENV)}")

        controller_cfg = load_controller_config(default_controller="OSC_POSE")
        env = suite.make(
            env_name=TASK_NAME_TO_ROBOSUITE_ENV[task_key],
            robots=robot,
            controller_configs=controller_cfg,
            has_renderer=False,
            has_offscreen_renderer=enable_render,
            use_camera_obs=enable_render,
            reward_shaping=False,
            control_freq=control_freq,
            horizon=max_episode_steps,
            ignore_done=False,
            hard_reset=False,
            camera_names=list(set(camera_names + [video_camera])),
            camera_heights=image_size,
            camera_widths=image_size,
        )
        if hasattr(env, "seed"):
            env.seed(seed)

        self.env = env
        self.task_name = task_key
        self.camera_names = camera_names
        self.state_ports = state_ports
        self.video_camera = video_camera
        self.video_resolution = video_resolution
        self.max_episode_steps = max_episode_steps
        self.done = False
        self.cur_step = 0

        obs_dict = env.reset()
        observation_space = gymnasium.spaces.Dict({})
        for port in self.state_ports:
            if port not in obs_dict:
                raise KeyError(f"State port '{port}' not found in environment observations.")
            observation_space.spaces[port] = gymnasium.spaces.Box(
                low=-np.inf, high=np.inf, shape=obs_dict[port].shape, dtype=np.float32
            )
        for cam_name in self.camera_names:
            observation_space.spaces[f"{cam_name}_rgb"] = gymnasium.spaces.Box(
                low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8
            )
        self.observation_space = observation_space
        self.action_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(env.action_dim,), dtype=np.float32
        )

    def _extract_obs(self, raw_obs: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
        if raw_obs is None:
            raw_obs = self.env._get_observations()

        obs_dict = {}
        for port in self.state_ports:
            obs_dict[port] = raw_obs[port].astype(np.float32)
        for cam_name in self.camera_names:
            obs_dict[f"{cam_name}_rgb"] = np.flip(raw_obs[f"{cam_name}_image"], axis=0).astype(np.uint8)
        return obs_dict

    def _check_success(self) -> bool:
        if hasattr(self.env, "_check_success"):
            return bool(self.env._check_success())
        if hasattr(self.env, "check_success"):
            return bool(self.env.check_success())
        return False

    def step(self, action: np.ndarray):
        obs, _, terminated, info = self.env.step(action)
        self.cur_step += 1
        success = self._check_success()
        reward = 1.0 if success else 0.0
        self.done = self.done or terminated or success or (self.cur_step >= self.max_episode_steps)
        return self._extract_obs(obs), reward, self.done, False, info

    def reset(self, seed=None, options=None):
        if seed is not None and hasattr(self.env, "seed"):
            self.env.seed(seed)
        obs = self.env.reset()
        self.done = False
        self.cur_step = 0
        return self._extract_obs(obs), {}

    def render(self, mode="rgb_array"):
        assert mode == "rgb_array"
        frame = np.flip(
            self.env.sim.render(
                height=self.video_resolution,
                width=self.video_resolution,
                camera_name=self.video_camera,
            ),
            axis=0,
        ).astype(np.uint8)
        return frame

    def close(self):
        self.env.close()
