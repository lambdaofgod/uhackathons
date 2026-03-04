"""Tests for rl_experiments.tracking."""

import pytest

from rl_experiments.config import RunConfig
from rl_experiments.runner import run_experiment
from rl_experiments.tracking import (
    MlflowEvalCallback,
    log_run_artifacts,
    run_exists,
    setup_tracking,
    start_mlflow_run,
)


@pytest.fixture
def cartpole_run_config(tmp_path):
    return RunConfig(
        experiment_name="test_tracking",
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


class TestSkipIfExists:
    @pytest.mark.slow
    def test_run_exists_false_initially(self, cartpole_run_config, tmp_path):
        setup_tracking(str(tmp_path / "mlruns"))
        assert run_exists(cartpole_run_config) is False

    @pytest.mark.slow
    def test_run_exists_true_after_logging(self, cartpole_run_config, tmp_path):
        setup_tracking(str(tmp_path / "mlruns"))

        with start_mlflow_run(cartpole_run_config):
            result = run_experiment(
                cartpole_run_config, eval_callback_class=MlflowEvalCallback
            )
            log_run_artifacts(result)

        assert run_exists(cartpole_run_config) is True

    @pytest.mark.slow
    def test_duplicate_not_rerun(self, cartpole_run_config, tmp_path):
        """After logging, run_exists returns True so a second run would be skipped."""
        setup_tracking(str(tmp_path / "mlruns"))

        with start_mlflow_run(cartpole_run_config):
            result = run_experiment(
                cartpole_run_config, eval_callback_class=MlflowEvalCallback
            )
            log_run_artifacts(result)

        # Simulate the CLI check
        assert run_exists(cartpole_run_config) is True

        # A different seed should not exist
        other = cartpole_run_config.model_copy(update={"seed": 99})
        assert run_exists(other) is False
