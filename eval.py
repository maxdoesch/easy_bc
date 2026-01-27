import os
from dataclasses import dataclass
from pathlib import Path

import imageio
import jax
import numpy as np
import orbax.checkpoint as ocp
import tyro
from flax import nnx
from lerobot.datasets.utils import load_stats
from lerobot.envs.factory import PushtEnv, make_env
from lerobot.envs.utils import preprocess_observation

from easy_bc.evaluation.evaluation import Evaluator, EvaluatorConfig
from easy_bc.policies.factory import (
    make_policy,
    make_policy_config,
    make_pre_post_processors,
)


@dataclass
class EvalCfg:
    checkpoint_path: str
    checkpoint: int
    evaluator: EvaluatorConfig
    n_envs: int = 1

    policy: str = tyro.MISSING

    seed: int = 43


def main(cfg: EvalCfg):
    init_rng = jax.random.PRNGKey(cfg.seed)
    policy_rngs = nnx.Rngs(jax.random.fold_in(init_rng, 0))
    eval_rng = jax.random.fold_in(init_rng, 1)

    env_cfg = PushtEnv()
    envs_dict = make_env(env_cfg, n_envs=cfg.n_envs)

    suite_name = next(iter(envs_dict))
    envs = envs_dict[suite_name][0]

    # dict_keys(['action', 'observation.state', 'observation.image'])
    policy_config = make_policy_config(cfg.policy, env_cfg=env_cfg)
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

    evaluator = Evaluator(envs=envs, cfg=cfg.evaluator)
    total_returns, evaluation_images = evaluator.evaluate(
        policy=policy,
        preprocessor=lambda x: preprocessor(preprocess_observation(x)),
        postprocessor=postprocessor,
        eval_rng=eval_rng,
    )

    imageio.mimsave(
        "evaluation.mp4",
        [img for episode in evaluation_images for img in episode],
        fps=env_cfg.fps,
    )

    print(f"Average reward: {np.mean(total_returns):.2f}")
    envs.close()


if __name__ == "__main__":
    tyro.cli(main)
