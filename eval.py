import os
from dataclasses import dataclass, field
from pathlib import Path

import jax
import numpy as np
import orbax.checkpoint as ocp
import tyro
from flax import nnx
from lerobot.datasets.utils import load_stats

from easy_bc.evaluation.evaluation import Evaluator, EvaluatorConfig
from easy_bc.policies.factory import (
    make_policy,
    make_policy_config,
    make_pre_post_processors,
)
from train import write_video_spawn


@dataclass
class EvalCfg:
    checkpoint_path: str
    checkpoint: int

    evaluator: EvaluatorConfig = field(default_factory=lambda: EvaluatorConfig())

    policy: str = tyro.MISSING

    seed: int = 43


def main(cfg: EvalCfg):
    init_rng = jax.random.PRNGKey(cfg.seed)
    policy_rngs = nnx.Rngs(jax.random.fold_in(init_rng, 0))
    eval_rng = jax.random.fold_in(init_rng, 1)

    policy_config = make_policy_config(
        cfg.policy, pretrained_path=cfg.checkpoint_path, device="cuda"
    )
    policy = make_policy(policy_config, rngs=policy_rngs)

    checkpoint_dir = Path(os.path.abspath(cfg.checkpoint_path))
    dataset_stats = load_stats(checkpoint_dir)

    if dataset_stats:
        policy_preprocessor, policy_postprocessor = make_pre_post_processors(
            policy_config, dataset_stats
        )
    else:
        raise ValueError(
            "Dataset stats are required for preprocessing and postprocessing."
        )

    evaluator = Evaluator(cfg=cfg.evaluator, policy_cfg=policy_config)

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

    eval_metrics = evaluator.evaluate(
        policy=policy,
        policy_preprocessor=policy_preprocessor,
        policy_postprocessor=policy_postprocessor,
        eval_rng=eval_rng,
    )

    evaluator.close()

    sum_rewards = np.mean(eval_metrics["sum_rewards"], axis=0)
    max_rewards = np.mean(eval_metrics["max_rewards"], axis=0)
    successes = np.mean(eval_metrics["successes"], axis=0)
    video_frames = eval_metrics["video_frames"]

    video_dir = Path("eval_videos")
    video_dir.mkdir(parents=True, exist_ok=True)
    for i in range(len(video_frames)):
        video_path = video_dir / f"eval_{cfg.checkpoint}_{i}.mp4"
        write_video_spawn(video_path, video_frames[i], fps=evaluator.fps)

    print(f"Eval results at checkpoint {cfg.checkpoint}:")
    print(f"  Average Sum Reward: {sum_rewards}")
    print(f"  Average Max Reward: {max_rewards}")
    print(f"  Success Rate: {successes}")
    print(f"  Evaluation video saved at: {video_dir}")


if __name__ == "__main__":
    tyro.cli(main)
