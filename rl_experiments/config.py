"""Config parsing for the RL experiment runner."""

import inspect
import logging
import warnings
from pathlib import Path
from typing import Any

import gymnasium as gym
import yaml
from pydantic import BaseModel
from stable_baselines3 import A2C, DQN, PPO

logger = logging.getLogger(__name__)

ALGO_CLASS_MAP = {
    "dqn": "dqn",
    "ppo": "ppo",
    "a2c": "a2c",
    "reinforce": "a2c",
}

SB3_CLASSES = {
    "dqn": DQN,
    "ppo": PPO,
    "a2c": A2C,
}


class RunConfig(BaseModel):
    experiment_name: str
    env_id: str
    algo_name: str
    algo_class: str  # actual SB3 class key (e.g. "a2c" for reinforce)
    algo_kwargs: dict[str, Any]  # passed to SB3 constructor (includes policy)
    seed: int
    total_timesteps: int
    eval_freq: int
    n_eval_episodes: int
    log_dir: str


class TrainingConfig(BaseModel):
    total_timesteps: int
    n_eval_episodes: int
    eval_freq: int
    n_seeds: int
    log_dir: str


class ProblemConfig(BaseModel):
    env_id: str
    total_timesteps: int | None = None


class ExperimentGroup(BaseModel):
    name: str
    problems: list[str | ProblemConfig]
    algorithms: list[str]


class FullConfig(BaseModel):
    experiments: list[ExperimentGroup]
    training: TrainingConfig
    algorithms: dict[str, dict[str, Any]]


def load_config(path: str | Path) -> FullConfig:
    """Load, parse, and validate the YAML config file."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    config = FullConfig(**raw)
    validate_algo_kwargs(config)
    return config


def resolve_algo_config(
    algo_name: str, algorithms: dict[str, dict[str, Any]]
) -> tuple[str, dict[str, Any]]:
    """Resolve _base inheritance and return (algo_class, kwargs).

    Returns the SB3 class key and the merged kwargs dict (with _base removed).
    """
    if algo_name not in algorithms:
        raise ValueError(
            f"Unknown algorithm '{algo_name}'. "
            f"Defined algorithms: {list(algorithms.keys())}"
        )
    config = dict(algorithms[algo_name])
    base_name = config.pop("_base", None)

    if base_name:
        if base_name not in algorithms:
            raise ValueError(
                f"Algorithm '{algo_name}' has _base '{base_name}' "
                f"which is not defined in the algorithms section"
            )
        base_config = dict(algorithms[base_name])
        base_config.pop("_base", None)
        base_config.update(config)
        config = base_config

    algo_class = ALGO_CLASS_MAP.get(algo_name)
    if algo_class is None:
        if base_name and base_name in ALGO_CLASS_MAP:
            algo_class = ALGO_CLASS_MAP[base_name]
        else:
            raise ValueError(
                f"Unknown algorithm '{algo_name}'. "
                f"Known algorithms: {list(ALGO_CLASS_MAP.keys())}"
            )

    return algo_class, config


def validate_algo_kwargs(config: FullConfig) -> None:
    """Validate all algorithm kwargs against SB3 constructor signatures.

    Raises ValueError if any kwarg is not accepted by the resolved SB3 class.
    """
    # Params managed by the runner, not user config
    runner_managed = {"env", "seed", "verbose"}

    for algo_name in {
        algo for group in config.experiments for algo in group.algorithms
    }:
        algo_class_key, kwargs = resolve_algo_config(algo_name, config.algorithms)
        sb3_class = SB3_CLASSES[algo_class_key]
        sig = inspect.signature(sb3_class.__init__)
        valid_params = set(sig.parameters.keys()) - {"self"}

        user_params = set(kwargs.keys())
        invalid = user_params - valid_params - runner_managed
        if invalid:
            raise ValueError(
                f"Algorithm '{algo_name}' (resolved to {sb3_class.__name__}) "
                f"has invalid kwargs: {sorted(invalid)}. "
                f"Valid params: {sorted(valid_params - runner_managed)}"
            )


def is_compatible(env_id: str, algo_name: str) -> bool:
    """Check if an algorithm is compatible with an environment's action space."""
    env = gym.make(env_id)
    is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    env.close()
    if algo_name == "dqn" and not is_discrete:
        return False
    return True


def expand_matrix(config: FullConfig) -> list[RunConfig]:
    """Expand the full config into a flat list of RunConfigs."""
    training = config.training
    run_configs: list[RunConfig] = []

    for group in config.experiments:
        for problem in group.problems:
            if isinstance(problem, ProblemConfig):
                env_id = problem.env_id
                timesteps = problem.total_timesteps or training.total_timesteps
            else:
                env_id = problem
                timesteps = training.total_timesteps

            for algo_name in group.algorithms:
                if not is_compatible(env_id, algo_name):
                    warnings.warn(
                        f"Skipping incompatible combo: "
                        f"{algo_name} + {env_id}",
                        stacklevel=2,
                    )
                    continue

                algo_class, algo_kwargs = resolve_algo_config(
                    algo_name, config.algorithms
                )

                for seed in range(training.n_seeds):
                    log_dir = (
                        f"{training.log_dir}/{group.name}/"
                        f"{algo_name}_{env_id}_seed{seed}"
                    )
                    run_configs.append(
                        RunConfig(
                            experiment_name=group.name,
                            env_id=env_id,
                            algo_name=algo_name,
                            algo_class=algo_class,
                            algo_kwargs=algo_kwargs,
                            seed=seed,
                            total_timesteps=timesteps,
                            eval_freq=training.eval_freq,
                            n_eval_episodes=training.n_eval_episodes,
                            log_dir=log_dir,
                        )
                    )

    logger.info(f"Expanded {len(run_configs)} run configs")
    return run_configs
