from dataclasses import dataclass
from typing import Callable, TypedDict

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


class EvalMetrics(TypedDict):
    sum_rewards: list[float]
    max_rewards: list[float]
    successes: list[bool]

    video_frames: list[np.ndarray]


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
    ) -> EvalMetrics:
        sum_rewards = []
        max_rewards = []
        successes = []
        video_frames = []

        current_sum_rewards = np.zeros(self.envs.num_envs, dtype=np.float32)
        current_max_rewards = np.full(self.envs.num_envs, -np.inf, dtype=np.float32)
        frames_per_env: list[list[np.ndarray]] = [[] for _ in range(self.envs.num_envs)]

        obs, info = self.envs.reset(seed=self.cfg.seed)
        done = np.zeros(self.envs.num_envs, dtype=bool)

        jit_sample_action = jax.jit(policy.sample_action)

        pbar = tqdm.tqdm(total=self.cfg.n_episodes, desc="Evaluating", unit="ep")

        episodes_finished = 0
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

                current_sum_rewards += reward
                current_max_rewards = np.maximum(current_max_rewards, reward)
                is_success = info.get(
                    "is_success", np.array([False] * self.envs.num_envs)
                )

                done = terminated | truncated

                frames = self.envs.render()
                for env_i, frame in enumerate(frames):  # pyright: ignore
                    frames_per_env[env_i].append(frame)

                if done.any():
                    done_idxs = np.nonzero(done)[0]

                    sum_rewards.extend(current_sum_rewards[done].tolist())
                    max_rewards.extend(current_max_rewards[done].tolist())
                    successes.extend(is_success[done].tolist())

                    video_frames.extend(
                        [np.stack(frames_per_env[i]) for i in done_idxs]
                    )

                    episodes_finished += len(done_idxs)
                    pbar.update(len(done_idxs))

                    current_sum_rewards[done] = 0.0
                    current_max_rewards[done] = 0.0

                    for i in done_idxs:
                        frames_per_env[i] = []

        pbar.close()

        return EvalMetrics(
            sum_rewards=sum_rewards,
            max_rewards=max_rewards,
            successes=successes,
            video_frames=video_frames,
        )
