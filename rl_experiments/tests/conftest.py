"""Shared fixtures for rl_experiments tests."""

import pytest

from rl_experiments.config import FullConfig, TrainingConfig, ExperimentGroup


@pytest.fixture
def minimal_algorithms():
    """Minimal algorithm configs for testing."""
    return {
        "ppo": {
            "policy": "MlpPolicy",
            "learning_rate": 0.0003,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
        },
        "dqn": {
            "policy": "MlpPolicy",
            "learning_rate": 0.0001,
            "buffer_size": 100_000,
            "learning_starts": 1000,
            "batch_size": 64,
            "tau": 1.0,
            "gamma": 0.99,
            "train_freq": 4,
            "target_update_interval": 1000,
            "exploration_fraction": 0.1,
            "exploration_final_eps": 0.05,
        },
        "a2c": {
            "policy": "MlpPolicy",
            "learning_rate": 0.0007,
            "n_steps": 5,
            "gamma": 0.99,
            "gae_lambda": 1.0,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
        },
        "reinforce": {
            "_base": "a2c",
            "policy": "MlpPolicy",
            "learning_rate": 0.001,
            "n_steps": 500,
            "gamma": 0.99,
            "gae_lambda": 1.0,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
        },
    }


@pytest.fixture
def minimal_config(minimal_algorithms):
    """A minimal FullConfig for testing."""
    return FullConfig(
        experiments=[
            ExperimentGroup(
                name="test_group",
                problems=["CartPole-v1"],
                algorithms=["ppo"],
            )
        ],
        training=TrainingConfig(
            total_timesteps=1000,
            n_eval_episodes=2,
            eval_freq=500,
            n_seeds=2,
            log_dir="/tmp/rl_test",
        ),
        algorithms=minimal_algorithms,
    )
