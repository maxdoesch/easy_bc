import os
import warnings

try:
    from pydantic.warnings import UnsupportedFieldAttributeWarning

    warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)
except Exception:
    pass

import dataclasses
import chex
import tyro
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import wandb
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass, field

import jax
import flax.nnx as nnx
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp

from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset
from lerobot.datasets.utils import write_stats
from lerobot.envs.factory import make_env, make_env_config
from lerobot.envs.utils import preprocess_observation
from lerobot.datasets.factory import resolve_delta_timestamps

from easy_bc.policies.policy import BasePolicy
from easy_bc.evaluation.evaluation import Evaluator, EvaluatorConfig

from easy_bc.policies.factory import (
    make_policy_config,
    make_policy,
    make_pre_post_processors,
)

from pydantic.warnings import UnsupportedFieldAttributeWarning

warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)


@dataclass()
class TrainConfig:
    project_name: str = "easy_bc"
    exp_name: str = tyro.MISSING

    policy: str = tyro.MISSING

    repo_id: str = tyro.MISSING
    train_steps: int = 10_000
    log_freq: int = 100
    eval_freq: int = 5_000

    checkpoint_freq: int = 1_000
    checkpoint_dir: str = "checkpoints/"

    batch_size: int = 32

    seed: int = 42

    env_id: Optional[str] = None
    num_envs: int = 1
    evaluator: EvaluatorConfig = field(default_factory=lambda: EvaluatorConfig())


def _write_video_worker(path_str: str, frames: np.ndarray, fps: int):
    from pathlib import Path as _Path
    from lerobot.utils.io_utils import write_video as _write_video

    _write_video(_Path(path_str), frames, fps=fps)


def write_video_spawn(video_path: Path, frames: np.ndarray, fps: int):
    ctx = mp.get_context("spawn")

    p = ctx.Process(
        target=_write_video_worker,
        args=(str(video_path), frames, int(fps)),
    )
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(f"write_video worker failed with exit code {p.exitcode}")


@nnx.jit
def train_step(
    policy: BasePolicy,
    batch: Dict[str, jnp.ndarray],
    optimizer: nnx.Optimizer,
    rng: chex.PRNGKey,
) -> float:
    def loss_fn(
        policy: BasePolicy, batch: Dict[str, jnp.ndarray], rng: chex.PRNGKey
    ) -> jnp.ndarray:
        chunked_loss = policy.compute_loss(batch, rng)

        return chunked_loss.mean()

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=False)
    loss, grads = grad_fn(policy, batch, rng)
    optimizer.update(policy, grads)

    return loss


def main(cfg: TrainConfig):
    wandb.init(
        project=cfg.project_name, name=cfg.exp_name, config=dataclasses.asdict(cfg)
    )

    init_rng = jax.random.PRNGKey(cfg.seed)
    policy_rngs = nnx.Rngs(jax.random.fold_in(init_rng, 0))
    train_rng = jax.random.fold_in(init_rng, 1)
    eval_rng = jax.random.fold_in(init_rng, 2)

    dataset_metadata = LeRobotDatasetMetadata(cfg.repo_id)

    policy_config = make_policy_config(
        cfg.policy, dataset_metadata=dataset_metadata, device="cuda"
    )
    policy = make_policy(policy_config, rngs=policy_rngs)

    if dataset_metadata.stats:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_config, dataset_metadata.stats
        )
    else:
        raise ValueError("Dataset stats are required for training.")

    dataset = LeRobotDataset(
        repo_id=cfg.repo_id,
        delta_timestamps=resolve_delta_timestamps(policy_config, dataset_metadata),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        multiprocessing_context="spawn",
        persistent_workers=True,
    )

    evaluator = None
    if cfg.env_id:
        env_cfg = make_env_config(cfg.env_id)
        eval_envs_dict = make_env(env_cfg, n_envs=cfg.num_envs)

        suite_name = next(iter(eval_envs_dict))
        eval_envs = eval_envs_dict[suite_name][0]

        evaluator = Evaluator(envs=eval_envs, cfg=cfg.evaluator)

    scheduler = optax.cosine_decay_schedule(
        init_value=1e-4, decay_steps=cfg.train_steps, alpha=1e-5
    )

    optimizer = nnx.Optimizer(
        policy, optax.adamw(learning_rate=scheduler), wrt=nnx.Param
    )

    policy.train()

    ckpt_dir = (
        Path(os.path.abspath(cfg.checkpoint_dir))
        / cfg.policy
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # save dataset stats
    write_stats(dataset_metadata.stats, ckpt_dir)  # pyright: ignore

    options = ocp.CheckpointManagerOptions(
        save_interval_steps=cfg.checkpoint_freq,
        create=True,
    )

    losses = []
    i = 0
    pbar = tqdm(total=cfg.train_steps, desc="Training")
    with ocp.CheckpointManager(str(ckpt_dir), options=options) as mngr:
        while True:
            for batch in dataloader:
                batch = preprocessor(batch)
                keep_keys = set(policy_config.input_features) | set(
                    policy_config.output_features
                )
                filtered = {k: v for k, v in batch.items() if k in keep_keys}

                batch = jax.tree_util.tree_map(jnp.asarray, filtered)

                train_rng, step_rng = jax.random.split(train_rng)

                # TODO: jax.block_until_ready to avoid profiling warning
                loss = train_step(policy, batch, optimizer, step_rng)
                losses.append(loss)

                if i % cfg.log_freq == 0 or i == cfg.train_steps - 1:
                    wandb.log({"train/loss": loss}, step=i)

                if evaluator and (i % cfg.eval_freq == 0 and i > 0):
                    policy.eval()
                    total_returns, episode_frames = evaluator.evaluate(
                        policy=policy,
                        preprocessor=lambda x: preprocessor(preprocess_observation(x)),
                        postprocessor=postprocessor,
                        eval_rng=eval_rng,
                    )
                    avg_return = jnp.mean(total_returns)
                    policy.train()

                    wandb.log(
                        {
                            "eval/return": avg_return,
                        },
                        step=i,
                    )

                    video_dir = ckpt_dir / "videos"
                    video_dir.mkdir(parents=True, exist_ok=True)
                    video_path = video_dir / f"eval_{i:08d}.mp4"
                    write_video_spawn(video_path, episode_frames[0], fps=env_cfg.fps)
                    wandb.log(
                        {
                            "eval/video": wandb.Video(
                                str(video_path),
                                format="mp4",
                            )
                        },
                        step=i,
                    )

                i += 1
                pbar.set_postfix(loss=f"{float(loss):.4f}")
                pbar.update(1)

                policy_gd, policy_state = nnx.split(policy)
                opt_gd, opt_state = nnx.split(optimizer)

                mngr.save(
                    i,
                    args=ocp.args.Composite(
                        policy=ocp.args.StandardSave(policy_state),  # pyright: ignore
                        optimizer=ocp.args.StandardSave(opt_state),  # pyright: ignore
                        step=ocp.args.JsonSave(i),  # pyright: ignore
                    ),
                )

                if i >= cfg.train_steps:
                    break

            if i >= cfg.train_steps:
                break


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    tyro.cli(main)
