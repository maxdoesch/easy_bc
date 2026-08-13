# easy_bc

Behavior cloning policies in JAX/Flax NNX for LeRobot datasets and environments.

## What It Provides

- Policy implementations for `regression` and `flow_unet`.
- LeRobot dataset training with observation/action preprocessing from dataset stats.
- Optional online evaluation during training through LeRobot environments.
- Orbax checkpointing with saved policy config, dataset stats, optimizer state, and step.
- Standalone checkpoint evaluation with video export.
- Hardware rollout for an SO-101 follower arm using LeRobot robot APIs.

## Setup

This project uses `uv` and Python 3.11+.

```bash
uv sync
```

For development checks:

```bash
uv run ruff format --check .
uv run ruff check .
uv run basedpyright
uv run pytest
```

## Train

Train on a LeRobot dataset repo id:

```bash
uv run python train.py \
  --cfg.exp-name my_run \
  --cfg.policy flow_unet \
  --cfg.repo-id <dataset_repo_id> \
  --cfg.evaluator.env-id <lerobot_env_id>
```

Useful options:

```bash
--cfg.train-steps 10000
--cfg.batch-size 32
--cfg.checkpoint-dir checkpoints/
--cfg.checkpoint-freq 1000
--cfg.eval-freq 5000
--cfg.evaluator.n-episodes 10
--cfg.evaluator.task-ids 0
--cfg.evaluator.num-envs 1
--cfg.no-wandb-enabled
```

Checkpoints are written under:

```text
<checkpoint_dir>/<policy>/<YYYYMMDD_HHMMSS>/
```

## Evaluate

Run evaluation from a saved checkpoint directory:

```bash
uv run python eval.py \
  --cfg.checkpoint-path checkpoints/flow_unet/20260101_120000 \
  --cfg.checkpoint 10000 \
  --cfg.policy flow_unet \
  --cfg.evaluator.env-id <lerobot_env_id>
```

Evaluation prints mean rewards and success rate, and writes videos to `eval_videos/`.

## Roll Out On Robot

Run a trained policy on the configured SO-101 follower arm:

```bash
uv run python rollout.py \
  --cfg.repo-id <dataset_repo_id> \
  --cfg.checkpoint-path checkpoints/flow_unet/20260101_120000 \
  --cfg.checkpoint 10000 \
  --cfg.policy flow_unet \
  --cfg.robot-port /dev/ttyACM0
```

`rollout.py` currently assumes two OpenCV cameras named `front` and `static`, with device indices `2` and `4`.

## CLI Help

All entry points are powered by `tyro`; use `--help` for the full option list:

```bash
uv run python train.py --help
uv run python eval.py --help
uv run python rollout.py --help
```
