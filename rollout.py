import os
import time
from dataclasses import dataclass
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import torch
import tyro
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import load_stats
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots import make_robot_from_config
from lerobot.robots.so101_follower import SO101FollowerConfig

from easy_bc.policies.factory import (
    make_policy,
    make_policy_config,
    make_pre_post_processors,
)

MAX_STEPS_PER_EPISODE = 20


@dataclass
class RolloutCfg:
    repo_id: str = tyro.MISSING

    checkpoint_path: str = tyro.MISSING
    checkpoint: int = tyro.MISSING

    robot_port: str = tyro.MISSING

    policy: str = tyro.MISSING

    seed: int = 43


def main(cfg: RolloutCfg):
    init_rng = jax.random.PRNGKey(cfg.seed)
    policy_rngs = nnx.Rngs(jax.random.fold_in(init_rng, 0))
    rollout_rng = jax.random.fold_in(init_rng, 1)

    camera_config = {
        "front": OpenCVCameraConfig(
            index_or_path=2, width=640, height=480, fps=30, fourcc="MJPG"
        ),
        "static": OpenCVCameraConfig(
            index_or_path=4, width=640, height=480, fps=30, fourcc="MJPG"
        ),
    }

    robot_config = SO101FollowerConfig(
        cfg.robot_port,
        id="so101_follower_arm",
        cameras=camera_config,  # pyright: ignore
    )

    policy_config = make_policy_config(
        cfg.policy, pretrained_path=cfg.checkpoint_path, device="cuda"
    )
    policy = make_policy(policy_config, rngs=policy_rngs)

    checkpoint_dir = Path(os.path.abspath(cfg.checkpoint_path))
    dataset_stats = load_stats(checkpoint_dir)

    if dataset_stats:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_config, dataset_stats
        )
    else:
        raise ValueError(
            "Dataset stats are required for preprocessing and postprocessing."
        )

    policy_gd, policy_state = nnx.split(policy)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_names=("policy", "optimizer", "step"),
    )
    restored = mngr.restore(
        cfg.checkpoint,
        args=ocp.args.Composite(
            policy=ocp.args.StandardRestore(policy_state),  # pyright: ignore
        ),
    )

    policy_state = restored["policy"]
    policy = nnx.merge(policy_gd, policy_state)

    policy.eval()

    jit_sample_action = jax.jit(policy.sample_action)

    robot = make_robot_from_config(robot_config)

    robot.connect()

    ds_meta = LeRobotDatasetMetadata(cfg.repo_id)

    while True:
        rollout_rng, step_rng = jax.random.split(rollout_rng)

        obs = robot.get_observation()
        obs_frame = build_inference_frame(
            obs, device=torch.device("cpu"), ds_features=ds_meta.features
        )

        observation = preprocessor(obs_frame)
        observation = {
            k: jax.device_put(jnp.asarray(v))
            for k, v in observation.items()
            if isinstance(v, (np.ndarray, jax.Array, torch.Tensor))
        }

        action_chunk = jit_sample_action(observation, rng=step_rng)
        action_chunk = torch.tensor(np.array(action_chunk), device="cpu")

        action_chunk = postprocessor(action_chunk)

        assert policy.n_action_steps <= action_chunk.shape[1], (
            "Policy n_action_steps must be <= the action horizon"
        )
        print("#" * 20)
        print(action_chunk)

        for i in range(policy.n_action_steps):
            action = make_robot_action(action_chunk[:, i], ds_meta.features)
            robot.send_action(action)
            time.sleep(0.1)

    robot.disconnect()


if __name__ == "__main__":
    tyro.cli(main)
