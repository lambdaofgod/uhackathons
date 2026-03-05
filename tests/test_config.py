"""Tests for rl_experiments.config."""

import pytest

from rl_experiments.config import (
    ExperimentGroup,
    FullConfig,
    ProblemConfig,
    TrainingConfig,
    expand_matrix,
    is_compatible,
    resolve_algo_config,
    validate_algo_kwargs,
)


class TestResolveAlgoConfig:
    def test_simple_algo(self, minimal_algorithms):
        algo_class, kwargs = resolve_algo_config("ppo", minimal_algorithms)
        assert algo_class == "ppo"
        assert kwargs["learning_rate"] == 0.0003
        assert "_base" not in kwargs

    def test_base_inheritance(self, minimal_algorithms):
        algo_class, kwargs = resolve_algo_config("reinforce", minimal_algorithms)
        assert algo_class == "a2c"
        # Overridden values from reinforce
        assert kwargs["learning_rate"] == 0.001
        assert kwargs["n_steps"] == 500
        assert kwargs["ent_coef"] == 0.0
        # Inherited from a2c base (not overridden)
        assert "policy" in kwargs
        assert "_base" not in kwargs

    def test_unknown_algo_raises(self, minimal_algorithms):
        with pytest.raises(ValueError, match="Unknown algorithm 'sac'"):
            resolve_algo_config("sac", minimal_algorithms)

    def test_invalid_base_raises(self, minimal_algorithms):
        algos = {**minimal_algorithms, "bad": {"_base": "nonexistent"}}
        with pytest.raises(ValueError, match="_base 'nonexistent'"):
            resolve_algo_config("bad", algos)


class TestValidateAlgoKwargs:
    def test_valid_config_passes(self, minimal_config):
        # Should not raise
        validate_algo_kwargs(minimal_config)

    def test_invalid_kwarg_raises(self, minimal_algorithms):
        algos = {**minimal_algorithms}
        algos["ppo"] = {**algos["ppo"], "bogus_param": 42}
        config = FullConfig(
            experiments=[
                ExperimentGroup(
                    name="test", problems=["CartPole-v1"], algorithms=["ppo"]
                )
            ],
            training=TrainingConfig(
                total_timesteps=1000,
                n_eval_episodes=2,
                eval_freq=500,
                n_seeds=1,
                log_dir="/tmp/test",
            ),
            algorithms=algos,
        )
        with pytest.raises(ValueError, match="invalid kwargs.*bogus_param"):
            validate_algo_kwargs(config)

    def test_reinforce_validates_against_a2c(self, minimal_algorithms):
        config = FullConfig(
            experiments=[
                ExperimentGroup(
                    name="test",
                    problems=["CartPole-v1"],
                    algorithms=["reinforce"],
                )
            ],
            training=TrainingConfig(
                total_timesteps=1000,
                n_eval_episodes=2,
                eval_freq=500,
                n_seeds=1,
                log_dir="/tmp/test",
            ),
            algorithms=minimal_algorithms,
        )
        # Should not raise - reinforce kwargs are valid A2C params
        validate_algo_kwargs(config)


class TestIsCompatible:
    def test_dqn_discrete_compatible(self):
        assert is_compatible("CartPole-v1", "dqn") is True

    def test_dqn_continuous_incompatible(self):
        assert is_compatible("Pendulum-v1", "dqn") is False

    def test_ppo_discrete_compatible(self):
        assert is_compatible("CartPole-v1", "ppo") is True

    def test_ppo_continuous_compatible(self):
        assert is_compatible("Pendulum-v1", "ppo") is True

    def test_a2c_continuous_compatible(self):
        assert is_compatible("Pendulum-v1", "a2c") is True


class TestExpandMatrix:
    def test_basic_expansion(self, minimal_config):
        runs = expand_matrix(minimal_config)
        # 1 env x 1 algo x 2 seeds
        assert len(runs) == 2

    def test_seed_values(self, minimal_config):
        runs = expand_matrix(minimal_config)
        seeds = [r.seed for r in runs]
        assert seeds == [0, 1]

    def test_log_dir_format(self, minimal_config):
        runs = expand_matrix(minimal_config)
        assert runs[0].log_dir == "/tmp/rl_test/test_group/ppo_CartPole-v1_seed0"

    def test_skips_incompatible(self, minimal_algorithms):
        config = FullConfig(
            experiments=[
                ExperimentGroup(
                    name="test",
                    problems=["Pendulum-v1"],
                    algorithms=["dqn"],
                )
            ],
            training=TrainingConfig(
                total_timesteps=1000,
                n_eval_episodes=2,
                eval_freq=500,
                n_seeds=1,
                log_dir="/tmp/test",
            ),
            algorithms=minimal_algorithms,
        )
        runs = expand_matrix(config)
        assert len(runs) == 0

    def test_multi_env_multi_algo(self, minimal_algorithms):
        config = FullConfig(
            experiments=[
                ExperimentGroup(
                    name="test",
                    problems=["CartPole-v1", "MountainCar-v0"],
                    algorithms=["ppo", "dqn"],
                )
            ],
            training=TrainingConfig(
                total_timesteps=1000,
                n_eval_episodes=2,
                eval_freq=500,
                n_seeds=1,
                log_dir="/tmp/test",
            ),
            algorithms=minimal_algorithms,
        )
        runs = expand_matrix(config)
        # 2 envs x 2 algos x 1 seed = 4
        assert len(runs) == 4
        combos = {(r.env_id, r.algo_name) for r in runs}
        assert combos == {
            ("CartPole-v1", "ppo"),
            ("CartPole-v1", "dqn"),
            ("MountainCar-v0", "ppo"),
            ("MountainCar-v0", "dqn"),
        }

    def test_fields_populated(self, minimal_config):
        runs = expand_matrix(minimal_config)
        run = runs[0]
        assert run.experiment_name == "test_group"
        assert run.env_id == "CartPole-v1"
        assert run.algo_name == "ppo"
        assert run.algo_class == "ppo"
        assert run.total_timesteps == 1000
        assert run.eval_freq == 500
        assert run.n_eval_episodes == 2
        assert "policy" in run.algo_kwargs

    def test_per_problem_timesteps_override(self, minimal_algorithms):
        config = FullConfig(
            experiments=[
                ExperimentGroup(
                    name="test",
                    problems=[
                        "CartPole-v1",
                        ProblemConfig(env_id="MountainCar-v0", total_timesteps=500_000),
                    ],
                    algorithms=["ppo"],
                )
            ],
            training=TrainingConfig(
                total_timesteps=1000,
                n_eval_episodes=2,
                eval_freq=500,
                n_seeds=1,
                log_dir="/tmp/test",
            ),
            algorithms=minimal_algorithms,
        )
        runs = expand_matrix(config)
        by_env = {r.env_id: r for r in runs}
        assert by_env["CartPole-v1"].total_timesteps == 1000
        assert by_env["MountainCar-v0"].total_timesteps == 500_000
