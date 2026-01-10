import os
import tyro
import imageio
from pathlib import Path
import numpy as np
from dataclasses import dataclass

from flax import nnx
import orbax.checkpoint as ocp

from easy_bc.policies.regression.modeling_regression import RegressionPolicy
from easy_bc.policies.regression.configuration_regression import RegressionConfig
from easy_bc.policies.regression.processors_regression import (
    make_processors_regression_pre_post_processors,
)
from easy_bc.evaluation.evaluation import EvaluatorConfig, Evaluator


from lerobot.envs.factory import make_env
from lerobot.datasets.utils import load_stats
from lerobot.configs.types import FeatureType
from lerobot.envs.utils import env_to_policy_features, preprocess_observation
from lerobot.envs.factory import PushtEnv


@dataclass
class EvalCfg:
    checkpoint_path: str
    checkpoint: int
    evaluator: EvaluatorConfig
    n_envs: int = 1


def main(cfg: EvalCfg):
    env_cfg = PushtEnv()
    envs_dict = make_env(env_cfg, n_envs=cfg.n_envs)

    suite_name = next(iter(envs_dict))
    envs = envs_dict[suite_name][0]

    # dict_keys(['action', 'observation.state', 'observation.image'])
    features = env_to_policy_features(env_cfg)

    output_features = {
        key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION
    }
    input_features = {
        key: ft for key, ft in features.items() if key not in output_features
    }

    regression_config = RegressionConfig(
        input_features=input_features, output_features=output_features, device="cuda"
    )

    checkpoint_dir = Path(os.path.abspath(cfg.checkpoint_path))
    dataset_stats = load_stats(checkpoint_dir)

    preprocessor, postprocessor = make_processors_regression_pre_post_processors(
        regression_config,
        dataset_stats=dataset_stats,  # pyright: ignore
    )

    policy = RegressionPolicy(config=regression_config, rngs=nnx.Rngs(0))
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
