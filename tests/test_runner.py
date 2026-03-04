"""Tests for rl_experiments.runner."""

import os

import pytest

from rl_experiments.config import RunConfig
from rl_experiments.runner import run_experiment


@pytest.fixture
def cartpole_run_config(tmp_path):
    """A minimal RunConfig for PPO + CartPole."""
    return RunConfig(
        experiment_name="test",
        env_id="CartPole-v1",
        algo_name="ppo",
        algo_class="ppo",
        algo_kwargs={
            "policy": "MlpPolicy",
            "learning_rate": 0.0003,
            "n_steps": 128,
            "batch_size": 64,
            "n_epochs": 2,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
        },
        seed=0,
        total_timesteps=512,
        eval_freq=256,
        n_eval_episodes=2,
        log_dir=str(tmp_path / "results"),
    )


class TestRunExperiment:
    @pytest.mark.slow
    def test_training_completes(self, cartpole_run_config):
        result = run_experiment(cartpole_run_config)

        assert result.config == cartpole_run_config
        assert os.path.exists(result.model_path + ".zip")
        assert os.path.exists(
            os.path.join(result.log_dir, "best_model.zip")
        )
        assert os.path.exists(
            os.path.join(result.log_dir, "evaluations.npz")
        )

    @pytest.mark.slow
    def test_result_fields(self, cartpole_run_config):
        result = run_experiment(cartpole_run_config)

        assert result.log_dir == cartpole_run_config.log_dir
        assert result.model_path == os.path.join(
            cartpole_run_config.log_dir, "model"
        )
