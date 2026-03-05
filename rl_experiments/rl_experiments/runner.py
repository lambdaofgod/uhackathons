"""Experiment runner for single RL training runs."""

import logging
import os
from dataclasses import dataclass

import gymnasium as gym
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.monitor import Monitor

from rl_experiments.config import RunConfig

# Type alias -- callers can pass MlflowEvalCallback (a subclass) via this param
EvalCallbackClass = type[EvalCallback]

logger = logging.getLogger(__name__)

ALGO_MAP = {
    "dqn": DQN,
    "ppo": PPO,
    "a2c": A2C,
    "reinforce": A2C,
}


@dataclass
class RunResult:
    config: RunConfig
    log_dir: str
    model_path: str


def run_experiment(
    config: RunConfig,
    eval_callback_class: EvalCallbackClass = EvalCallback,
) -> RunResult:
    """Execute a single training run from a RunConfig."""
    os.makedirs(config.log_dir, exist_ok=True)

    algo_class = ALGO_MAP[config.algo_class]

    env = Monitor(gym.make(config.env_id))
    eval_env = Monitor(gym.make(config.env_id))

    model = algo_class(
        env=env,
        seed=config.seed,
        verbose=0,
        **config.algo_kwargs,
    )

    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=5,
        verbose=1,
    )

    eval_callback = eval_callback_class(
        eval_env,
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        log_path=config.log_dir,
        best_model_save_path=config.log_dir,
        callback_after_eval=stop_callback,
    )

    logger.info(
        f"Training {config.algo_name} on {config.env_id} (seed={config.seed})"
    )

    model.learn(
        total_timesteps=config.total_timesteps,
        callback=eval_callback,
    )

    model_path = os.path.join(config.log_dir, "model")
    model.save(model_path)

    env.close()
    eval_env.close()

    logger.info(f"Saved model to {model_path}")
    return RunResult(config=config, log_dir=config.log_dir, model_path=model_path)
