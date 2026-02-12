import json
from dataclasses import dataclass, field
from typing import List, Optional, TypedDict, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
import torch
import tqdm
import tyro
from gymnasium.vector import VectorEnv
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs import EnvConfig
from lerobot.envs.factory import make_env, make_env_config, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION

from easy_bc.policies.policy import BasePolicy


@dataclass
class EvaluatorConfig:
    n_episodes: int = 10
    seed: int = 42

    env_id: str = tyro.MISSING
    env_kwargs: Optional[str] = None

    task_ids: list[int] = field(default_factory=lambda: [0])
    num_envs: int = 1

    def __post_init__(self) -> None:
        if self.env_kwargs is None:
            self.env_kwargs_dict = {}
        elif isinstance(self.env_kwargs, str):
            self.env_kwargs_dict = json.loads(self.env_kwargs)
            if not isinstance(self.env_kwargs_dict, dict):
                raise ValueError("env_kwargs JSON must decode to a dict")


class EvalMetrics(TypedDict):
    sum_rewards: list[float]
    max_rewards: list[float]
    successes: list[bool]

    video_frames: list[np.ndarray]


class EnvConfigWrapper:
    def __init__(self, env_cfg: EnvConfig, task_ids: List[int], **kwargs):
        self.env_cfg = env_cfg
        self.task_ids = task_ids
        self._override_kwargs = kwargs

    def __getattr__(self, name):
        return getattr(self.env_cfg, name)

    @property
    def gym_kwargs(self) -> dict:
        return {
            **self.env_cfg.gym_kwargs,
            **self._override_kwargs,
            "task_ids": self.task_ids,
        }


class Evaluator:
    def __init__(self, cfg: EvaluatorConfig, policy_cfg: PreTrainedConfig):
        self.cfg = cfg

        self.per_task_envs: List[VectorEnv] | None = None
        self.env_cfg = None

        if self.cfg.env_id:
            self.env_cfg = make_env_config(env_type=self.cfg.env_id)

            # hack to pass custom kwargs to environment construction
            self.env_cfg = cast(
                EnvConfig,
                EnvConfigWrapper(
                    self.env_cfg, self.cfg.task_ids, **self.cfg.env_kwargs_dict
                ),
            )
            eval_envs_dict = make_env(self.env_cfg, n_envs=cfg.num_envs)

            suite_name = next(iter(eval_envs_dict))
            self.per_task_envs = [
                eval_envs_dict[suite_name][task_id] for task_id in self.cfg.task_ids
            ]

            self.env_preprocessor, self.env_postprocessor = (
                make_env_pre_post_processors(self.env_cfg, policy_cfg)
            )

    def evaluate(
        self,
        policy: BasePolicy,
        policy_preprocessor: PolicyProcessorPipeline,
        policy_postprocessor: PolicyProcessorPipeline,
        eval_rng: chex.PRNGKey,
    ) -> EvalMetrics | None:
        if self.per_task_envs is None:
            return None

        sum_rewards = []
        max_rewards = []
        successes = []
        video_frames = []

        for envs in self.per_task_envs:
            current_sum_rewards = np.zeros(envs.num_envs, dtype=np.float32)
            current_max_rewards = np.full(envs.num_envs, -np.inf, dtype=np.float32)
            frames_per_env: list[list[np.ndarray]] = [[] for _ in range(envs.num_envs)]

            obs, info = envs.reset(seed=self.cfg.seed)
            max_steps = envs.call("_max_episode_steps")[0]  # pyright: ignore
            done = np.zeros(envs.num_envs, dtype=bool)
            steps = np.zeros(envs.num_envs, dtype=int)

            jit_sample_action = jax.jit(policy.sample_action)

            pbar = tqdm.tqdm(total=self.cfg.n_episodes, desc="Evaluating", unit="ep")

            episodes_finished = 0
            while episodes_finished < self.cfg.n_episodes:
                eval_rng, step_rng = jax.random.split(eval_rng)

                observation = preprocess_observation(obs)
                observation = self.env_preprocessor(observation)
                observation = policy_preprocessor(observation)
                observation = jax.tree_util.tree_map(
                    lambda x: jax.device_put(jnp.asarray(x)),
                    observation,
                )

                action = jit_sample_action(observation, rng=step_rng)
                action = torch.tensor(np.array(action), device="cpu")

                action = policy_postprocessor(action)
                action = self.env_postprocessor({ACTION: action})[ACTION]

                action_np: np.ndarray = action.to("cpu").numpy()

                assert policy.n_action_steps <= action_np.shape[1], (
                    "Policy n_action_steps must be <= the action horizon"
                )

                for i in range(policy.n_action_steps):
                    obs, reward, terminated, truncated, info = envs.step(
                        action_np[:, i]
                    )
                    steps += 1

                    current_sum_rewards += reward
                    current_max_rewards = np.maximum(current_max_rewards, reward)

                    final_info = info.get("final_info")
                    if final_info is not None and not isinstance(final_info, dict):
                        raise RuntimeError(
                            "Unsupported `final_info` format: \
                            expected dict (Gymnasium >= 1.0). "
                        )

                    is_success = np.asarray(
                        (final_info or {}).get(
                            "is_success", np.zeros(envs.num_envs, dtype=bool)
                        ),
                        dtype=bool,
                    )

                    done = terminated | truncated | (steps >= max_steps)

                    frames = envs.render()
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
                        steps[done] = 0

                        for i in done_idxs:
                            frames_per_env[i] = []

                        for idx in done_idxs:
                            envs.envs[idx].reset()  # type: ignore

            pbar.close()

        return EvalMetrics(
            sum_rewards=sum_rewards,
            max_rewards=max_rewards,
            successes=successes,
            video_frames=video_frames,
        )

    def close(self) -> None:
        if self.per_task_envs:
            for envs in self.per_task_envs:
                envs.close()

    @property
    def fps(self) -> int:
        if self.env_cfg:
            return self.env_cfg.fps
        return 1
