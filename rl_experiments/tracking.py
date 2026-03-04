"""MLFlow tracking for RL experiments."""

import logging
from contextlib import contextmanager
from typing import Any

import gymnasium as gym
import mlflow
import numpy as np
from mlflow.tracking import MlflowClient
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from rl_experiments.config import RunConfig
from rl_experiments.runner import ALGO_MAP, RunResult

logger = logging.getLogger(__name__)


def setup_tracking(tracking_uri: str = "./mlruns") -> None:
    """Set up local file-based MLFlow tracking."""
    mlflow.set_tracking_uri(tracking_uri)


def _run_params(config: RunConfig) -> dict[str, Any]:
    """Extract the params dict we log and query for a run."""
    return {
        "env_id": config.env_id,
        "algo_name": config.algo_name,
        "algo_class": config.algo_class,
        "seed": str(config.seed),
        **{f"algo.{k}": str(v) for k, v in config.algo_kwargs.items()},
    }


def run_exists(config: RunConfig) -> bool:
    """Check if a run with matching params already exists in MLFlow."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(config.experiment_name)
    if experiment is None:
        return False

    filter_parts = [
        f"params.env_id = '{config.env_id}'",
        f"params.algo_name = '{config.algo_name}'",
        f"params.seed = '{config.seed}'",
    ]
    filter_string = " and ".join(filter_parts)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string,
        max_results=1,
    )
    return len(runs) > 0


class MlflowEvalCallback(EvalCallback):
    """EvalCallback that also logs eval metrics to the active MLFlow run."""

    def _on_step(self) -> bool:
        result = super()._on_step()

        # After parent runs evaluation, it stores results in evaluations_
        # Check if a new evaluation was recorded this step
        if hasattr(self, "evaluations_timesteps") and len(self.evaluations_timesteps) > 0:
            last_timestep = self.evaluations_timesteps[-1]
            if last_timestep == self.num_timesteps:
                mean_reward = float(np.mean(self.evaluations_results[-1]))
                std_reward = float(np.std(self.evaluations_results[-1]))
                mlflow.log_metrics(
                    {
                        "eval/mean_reward": mean_reward,
                        "eval/std_reward": std_reward,
                    },
                    step=self.num_timesteps,
                )

        return result


@contextmanager
def start_mlflow_run(config: RunConfig):
    """Context manager that opens an MLFlow run and logs params."""
    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run(
        run_name=f"{config.algo_name}_{config.env_id}_seed{config.seed}"
    ):
        mlflow.log_params(_run_params(config))
        mlflow.log_param("total_timesteps", config.total_timesteps)
        yield


def log_run_artifacts(result: RunResult) -> None:
    """Log post-training artifacts and final eval to the active MLFlow run."""
    config = result.config

    # Final evaluation of the trained model
    algo_cls = ALGO_MAP[config.algo_class]
    model = algo_cls.load(result.model_path)

    eval_env = gym.make(config.env_id)
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=config.n_eval_episodes
    )
    eval_env.close()

    mlflow.log_metrics({
        "eval/mean_reward": mean_reward,
        "eval/std_reward": std_reward,
    })

    # Log artifacts
    model_zip = result.model_path + ".zip"
    mlflow.log_artifact(model_zip)

    evaluations_path = f"{result.log_dir}/evaluations.npz"
    try:
        mlflow.log_artifact(evaluations_path)
    except Exception:
        logger.warning(f"Could not log evaluations artifact: {evaluations_path}")

    logger.info(
        f"Logged {config.algo_name}/{config.env_id}/seed{config.seed}: "
        f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}"
    )
