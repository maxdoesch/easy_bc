import tqdm
import torch
import numpy as np
from typing import Callable
from dataclasses import dataclass
from gymnasium.vector.vector_env import VectorEnv

import jax
import jax.numpy as jnp
from flax import nnx


@dataclass
class EvaluatorConfig:
    n_episodes: int = 10


class Evaluator:
    def __init__(self, envs: VectorEnv, cfg: EvaluatorConfig):
        self.cfg = cfg
        self.envs = envs

    def evaluate(
        self, policy: nnx.Module, preprocessor: Callable, postprocessor: Callable
    ) -> tuple[float, list[np.ndarray]]:
        total_returns = []
        episodes_finished = 0
        obs, info = self.envs.reset()
        done = np.zeros(self.envs.num_envs, dtype=bool)
        episode_return = np.zeros(self.envs.num_envs, dtype=np.float32)

        jit_policy = jax.jit(policy)

        pbar = tqdm.tqdm(total=self.cfg.n_episodes, desc="Evaluating", unit="ep")

        episode_images: list[list[np.ndarray]] = []
        frames_per_env: list[list[np.ndarray]] = [[] for _ in range(self.envs.num_envs)]
        while episodes_finished < self.cfg.n_episodes:
            observation = preprocessor(obs)
            observation = jax.tree_util.tree_map(jnp.asarray, observation)

            action = jit_policy(observation)
            action = torch.tensor(np.array(action), device="cpu")
            action = postprocessor(action)

            obs, reward, terminated, truncated, info = self.envs.step(action)
            episode_return += reward
            done = terminated | truncated

            frames = self.envs.render()
            for env_i, frame in enumerate(frames):  # pyright: ignore
                frames_per_env[env_i].append(frame)

            if done.any():
                done_idxs = np.nonzero(done)[0]
                for i in done_idxs:
                    if episodes_finished >= self.cfg.n_episodes:
                        break
                    total_returns.append(float(episode_return[i]))
                    episodes_finished += 1
                    pbar.update(1)

                    episode_images.append(frames_per_env[i])
                    frames_per_env[i] = []

                episode_return[done] = 0.0
        pbar.close()

        # n_envs, F, H, W, C
        episode_images_tensor = [np.stack(episode) for episode in episode_images]

        return np.mean(total_returns).item(), episode_images_tensor
