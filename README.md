# RL Experiments

A configurable pipeline for running Stable Baselines 3 algorithms across Gymnasium environments, collecting results with MLFlow, and comparing performance.

## Setup

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
# Install dependencies
uv sync

# Verify everything works
uv run pytest -m "not slow"
```

## Quick start

```bash
# 1. Preview the experiment matrix (no training)
uv run python -m rl_experiments --dry-run

# 2. Run a quick smoke test (~1 min, tiny timesteps)
uv run python -m rl_experiments --config experiments_quick.yaml

# 3. Run the full experiment suite (takes a while)
uv run python -m rl_experiments
```

## Viewing results with MLFlow

All runs are logged to a local `./mlruns/` directory. No MLFlow server setup is needed -- just start the UI:

```bash
uv run mlflow ui
```

Then open http://127.0.0.1:5000 in your browser. You will see:

- **Experiments** in the left sidebar (e.g. `discrete_sparse`, `continuous_control`)
- **Runs** listed per experiment, one per `(algorithm, environment, seed)` combo
- **Params** tab: algorithm name, env_id, seed, all hyperparameters
- **Metrics** tab: `eval/mean_reward` and `eval/std_reward`
- **Artifacts** tab: saved model `.zip` and evaluation data

To compare runs, select multiple runs with the checkboxes and click "Compare".

### Re-running experiments

The runner automatically skips runs that already exist in MLFlow (matched by env, algorithm, and seed). To re-run everything from scratch:

```bash
rm -rf mlruns/ results/
uv run python -m rl_experiments --config experiments_quick.yaml
```

## Configuration

Experiments are defined in YAML. See `experiments.yaml` for the full config or `experiments_quick.yaml` for a minimal version.

```yaml
experiments:
  - name: discrete_sparse
    problems:
      - CartPole-v1
    algorithms:
      - ppo
      - dqn

training:
  total_timesteps: 100_000
  n_seeds: 3

algorithms:
  ppo:
    policy: MlpPolicy
    learning_rate: 0.0003
    # ... all SB3 constructor kwargs
```

Key config features:

- **Algorithm inheritance**: `reinforce` uses `_base: a2c` to reuse the A2C class with different hyperparameters
- **Compatibility filtering**: DQN is automatically skipped for continuous action spaces
- **Direct SB3 kwargs**: algorithm config maps directly to Stable Baselines 3 constructor arguments

## Tests

```bash
# Fast tests only (config parsing, validation)
uv run pytest -m "not slow"

# All tests including integration tests (trains small models)
uv run pytest -v
```

## Project structure

```
rl_experiments/
  __init__.py
  __main__.py      # CLI entry point
  config.py        # YAML parsing, matrix expansion
  runner.py        # SB3 training execution
  tracking.py      # MLFlow logging
experiments.yaml        # Full experiment config
experiments_quick.yaml  # Quick validation config (tiny timesteps)
```
