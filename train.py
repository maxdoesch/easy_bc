import os
import tyro
from tqdm import tqdm
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

from easy_bc.policies.regression.modeling_regression import RegressionPolicy
from easy_bc.evaluation.evaluation import Evaluator, EvaluatorConfig

from easy_bc.policies.factory import (
    make_policy_config,
    make_policy,
    make_pre_post_processors,
)


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

    env_id: Optional[str] = None
    num_envs: int = 1
    evaluator: EvaluatorConfig = field(default_factory=lambda: EvaluatorConfig())


@nnx.jit
def train_step(
    policy: RegressionPolicy, batch: Dict[str, jnp.ndarray], optimizer: nnx.Optimizer
) -> float:
    def loss_fn(policy: RegressionPolicy, batch: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        chunked_loss = policy.compute_loss(batch)

        return chunked_loss.mean()

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=False)
    loss, grads = grad_fn(policy, batch)
    optimizer.update(policy, grads)

    return loss


def main(cfg: TrainConfig):
    wandb.init(
        project=cfg.project_name,
        name=cfg.exp_name,
    )

    dataset_metadata = LeRobotDatasetMetadata(cfg.repo_id)

    policy_config = make_policy_config(
        cfg.policy, dataset_metadata=dataset_metadata, device="cuda"
    )
    policy = make_policy(policy_config, rngs=nnx.Rngs(0))

    preprocessor, postprocessor = make_pre_post_processors(
        policy_config, dataset_metadata
    )

    dataset = LeRobotDataset(
        repo_id=cfg.repo_id,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )

    evaluator = None
    if cfg.env_id:
        env_cfg = make_env_config(cfg.env_id)
        eval_envs_dict = make_env(env_cfg, n_envs=cfg.num_envs)

        suite_name = next(iter(eval_envs_dict))
        eval_envs = eval_envs_dict[suite_name][0]

        evaluator = Evaluator(envs=eval_envs, cfg=cfg.evaluator)

    optimizer = nnx.Optimizer(policy, optax.adamw(learning_rate=1e-3), wrt=nnx.Param)

    policy.train()

    ckpt_dir = Path(os.path.abspath(cfg.checkpoint_dir)) / datetime.now().strftime(
        "%Y%m%d_%H%M%S"
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

                loss = train_step(policy, batch, optimizer)
                losses.append(loss)

                if i % cfg.log_freq == 0 or i == cfg.train_steps - 1:
                    wandb.log({"train/loss": loss}, step=i)

                if evaluator and (i % cfg.eval_freq == 0 and i > 0):
                    policy.eval()
                    total_returns, _ = evaluator.evaluate(
                        policy=policy,
                        preprocessor=lambda x: preprocessor(preprocess_observation(x)),
                        postprocessor=postprocessor,
                    )
                    avg_return = jnp.mean(total_returns)
                    policy.train()

                    wandb.log(
                        {
                            "eval/return": avg_return,
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
    tyro.cli(main)
