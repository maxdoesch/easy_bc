import os
import torch
import tyro
import imageio
from pathlib import Path
import numpy as np
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp

from easy_bc.modeling_regression import RegressionPolicy
from easy_bc.configuration_regression import RegressionConfig
from easy_bc.processors_regression import make_processors_regression_pre_post_processors


from lerobot.envs.factory import make_env, make_env_config
from lerobot.datasets.utils import load_stats
from lerobot.configs.types import FeatureType
from lerobot.envs.utils import env_to_policy_features, preprocess_observation

ENV_NAME = "pusht"


@dataclass
class EvalConfig:
    checkpoint_path: str
    checkpoint: int


def main(cfg: EvalConfig):
    env_cfg = make_env_config(ENV_NAME)
    envs_dict = make_env(env_cfg)

    suite_name = next(iter(envs_dict))
    env = envs_dict[suite_name][0]

    # dict_keys(['action', 'observation.state', 'observation.image'])
    features = env_to_policy_features(env_cfg)

    output_features = {
        key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION
    }
    input_features = {
        key: ft for key, ft in features.items() if key not in output_features
    }

    regression_config = RegressionConfig(
        input_features=input_features, output_features=output_features, device="cuda"
    )

    checkpoint_dir = Path(os.path.abspath(cfg.checkpoint_path))
    dataset_stats = load_stats(checkpoint_dir)

    preprocessor, postprocessor = make_processors_regression_pre_post_processors(
        regression_config,
        dataset_stats=dataset_stats,  # pyright: ignore
    )

    policy = RegressionPolicy(config=regression_config, rngs=nnx.Rngs(0))
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

    # dict_keys(['agent_pos', 'pixels'])
    obs, info = env.reset()
    done = np.zeros(env.num_envs, dtype=bool)
    total_reward = np.zeros(env.num_envs)

    images = []
    while not done.all():
        observation = preprocess_observation(obs)
        observation = preprocessor(observation)

        observation = jax.tree_util.tree_map(jnp.asarray, observation)

        action = policy(observation)

        action = torch.tensor(np.array(action), device="cpu")

        action = postprocessor(action)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated | truncated

        images.append(obs["pixels"])

    print(f"Average reward: {total_reward.mean():.2f}")
    env.close()

    # save video
    video = np.concatenate(images, axis=0)  # (N, H, W, C)
    imageio.mimwrite("eval_video.mp4", video, fps=10)  # pyright: ignore


if __name__ == "__main__":
    tyro.cli(main)
