import os
import tyro
from pathlib import Path
from datetime import datetime
from typing import Dict
from dataclasses import dataclass

from matplotlib import pyplot as plt

import jax
import flax.nnx as nnx
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp

from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType

from easy_bc.configuration_regression import RegressionConfig
from easy_bc.modeling_regression import RegressionPolicy
from easy_bc.processors_regression import make_processors_regression_pre_post_processors


@dataclass
class TrainConfig:
    repo_id: str
    train_steps: int = 10_000
    log_freq: int = 100

    checkpoint_freq: int = 1_000

    checkpoint_dir: str = "checkpoints/"

    batch_size: int = 1


def loss_fn(policy: RegressionPolicy, batch: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    pred_actions = policy(batch)
    actions = batch["action"]

    loss = optax.l2_loss(predictions=pred_actions, targets=actions).mean()

    return loss


@nnx.jit
def train_step(
    policy: RegressionPolicy, batch: Dict[str, jnp.ndarray], optimizer: nnx.Optimizer
) -> float:
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=False)
    loss, grads = grad_fn(policy, batch)
    optimizer.update(policy, grads)

    return loss


def main(cfg: TrainConfig):
    dataset_metadata = LeRobotDatasetMetadata(cfg.repo_id)
    features = dataset_to_policy_features(dataset_metadata.features)

    output_features = {
        key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION
    }
    input_features = {
        key: ft for key, ft in features.items() if key not in output_features
    }

    regression_config = RegressionConfig(
        input_features=input_features, output_features=output_features, device="cuda"
    )

    policy = RegressionPolicy(config=regression_config, rngs=nnx.Rngs(0))
    pre_processor, _ = make_processors_regression_pre_post_processors(
        regression_config,
        dataset_stats=dataset_metadata.stats,  # pyright: ignore
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

    optimizer = nnx.Optimizer(policy, optax.adamw(learning_rate=1e-3), wrt=nnx.Param)

    policy.train()

    ckpt_dir = Path(os.path.abspath(cfg.checkpoint_dir)) / datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    options = ocp.CheckpointManagerOptions(
        save_interval_steps=cfg.checkpoint_freq,
        create=True,
    )

    losses = []
    i = 0
    with ocp.CheckpointManager(str(ckpt_dir), options=options) as mngr:
        while True:
            for batch in dataloader:
                batch = pre_processor(batch)
                keep_keys = set(regression_config.input_features) | set(
                    regression_config.output_features
                )
                filtered = {k: v for k, v in batch.items() if k in keep_keys}

                batch = jax.tree_util.tree_map(jnp.asarray, filtered)

                loss = train_step(policy, batch, optimizer)
                losses.append(loss)

                if i % cfg.log_freq == 0 or i == cfg.train_steps - 1:
                    print(f"{i}: Loss: {loss}")

                policy_gd, policy_state = nnx.split(policy)
                opt_gd, opt_state = nnx.split(optimizer)

                i += 1

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

    # plot losses
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig("training_loss.png")
    plt.close()


if __name__ == "__main__":
    tyro.cli(main)
