from dataclasses import dataclass
from typing import Callable

import chex
import jax
import jax.numpy as jnp
import numpy as np
import torch
import tqdm
from gymnasium.vector.vector_env import VectorEnv

from easy_bc.policies.policy import BasePolicy


@dataclass
class EvaluatorConfig:
    n_episodes: int = 10
    seed: int = 42


class Evaluator:
    def __init__(self, envs: VectorEnv, cfg: EvaluatorConfig):
        self.cfg = cfg
        self.envs = envs

    def evaluate(
        self,
        policy: BasePolicy,
        preprocessor: Callable,
        postprocessor: Callable,
        eval_rng: chex.PRNGKey,
    ) -> tuple[float, list[np.ndarray]]:
        total_returns = []
        episodes_finished = 0
        obs, info = self.envs.reset(seed=self.cfg.seed)
        done = np.zeros(self.envs.num_envs, dtype=bool)
        episode_return = np.zeros(self.envs.num_envs, dtype=np.float32)
        episode_steps = np.zeros(self.envs.num_envs, dtype=np.int32)

        jit_sample_action = jax.jit(policy.sample_action)

        pbar = tqdm.tqdm(total=self.cfg.n_episodes, desc="Evaluating", unit="ep")

        episode_images: list[list[np.ndarray]] = []
        frames_per_env: list[list[np.ndarray]] = [[] for _ in range(self.envs.num_envs)]
        while episodes_finished < self.cfg.n_episodes:
            eval_rng, step_rng = jax.random.split(eval_rng)

            observation = preprocessor(obs)
            observation = jax.tree_util.tree_map(jnp.asarray, observation)

            action = jit_sample_action(observation, rng=step_rng)
            action = torch.tensor(np.array(action), device="cpu")

            action = postprocessor(action)

            assert policy.n_action_steps <= action.shape[1], (
                "Policy n_action_steps must be <= the action horizon"
            )

            for i in range(policy.n_action_steps):
                obs, reward, terminated, truncated, info = self.envs.step(action[:, i])

                episode_return += reward
                episode_steps += 1
                done = terminated | truncated

                frames = self.envs.render()
                for env_i, frame in enumerate(frames):  # pyright: ignore
                    frames_per_env[env_i].append(frame)

                if done.any():
                    done_idxs = np.nonzero(done)[0]
                    for done_idx in done_idxs:
                        if episodes_finished >= self.cfg.n_episodes:
                            break
                        total_returns.append(
                            episode_return[done_idx] / episode_steps[done_idx]
                        )
                        episodes_finished += 1
                        pbar.update(1)

                        episode_images.append(frames_per_env[done_idx])
                        frames_per_env[done_idx] = []

                    episode_return[done] = 0.0
        pbar.close()

        # n_envs, F, H, W, C
        episode_frames = [np.stack(episode) for episode in episode_images]

        return np.mean(total_returns).item() * 100, episode_frames
