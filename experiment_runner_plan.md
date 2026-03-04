# RL Experiment Runner — Design Plan

A configurable pipeline for running Stable Baselines 3 algorithms across Gymnasium environments, collecting results, and comparing performance.

## Architecture

Three layers: **config parsing → experiment runner → results collection**. Each algorithm config maps directly to SB3 constructor kwargs, so no translation layer is needed.

---

## Config Structure

```yaml
# experiments.yaml

experiments:
  - name: discrete_sparse
    problems:
      - CartPole-v1
      - MountainCar-v0
      - LunarLander-v2
    algorithms:
      - dqn
      - ppo

  - name: continuous_control
    problems:
      - Pendulum-v1
      - LunarLanderContinuous-v2
    algorithms:
      - ppo
      - a2c

training:
  total_timesteps: 100_000
  n_eval_episodes: 10
  eval_freq: 5000        # evaluate every N steps
  n_seeds: 3             # repeat each combo for variance estimation
  log_dir: ./results

algorithms:
  dqn:
    policy: MlpPolicy
    learning_rate: 0.0001
    buffer_size: 100_000
    learning_starts: 1000
    batch_size: 64
    tau: 1.0
    gamma: 0.99
    train_freq: 4
    target_update_interval: 1000
    exploration_fraction: 0.1
    exploration_final_eps: 0.05

  ppo:
    policy: MlpPolicy
    learning_rate: 0.0003
    n_steps: 2048
    batch_size: 64
    n_epochs: 10
    gamma: 0.99
    gae_lambda: 0.95
    clip_range: 0.2
    ent_coef: 0.01        # entropy bonus — mild curiosity-adjacent exploration
    vf_coef: 0.5

  a2c:
    policy: MlpPolicy
    learning_rate: 0.0007
    n_steps: 5
    gamma: 0.99
    gae_lambda: 1.0
    ent_coef: 0.01
    vf_coef: 0.5

  reinforce:
    # SB3 doesn't have vanilla REINFORCE, but A2C with n_steps=rollout_length
    # and gae_lambda=1.0 approximates it (full Monte Carlo returns, no bootstrapping)
    _base: a2c
    policy: MlpPolicy
    learning_rate: 0.001
    n_steps: 500          # full episode rollout
    gamma: 0.99
    gae_lambda: 1.0       # no bootstrapping — pure Monte Carlo
    ent_coef: 0.0
    vf_coef: 0.5
```

### Key design decisions

- **`_base` inheritance**: REINFORCE is not a standalone SB3 class. We implement it as A2C with `gae_lambda=1.0` (no bootstrapping, pure Monte Carlo returns) and long `n_steps` (full episode rollouts). The `_base` field tells the config parser to use the A2C class but override with the given kwargs.
- **Direct kwargs mapping**: Every key under an algorithm name (except `_base` and `policy`) is passed directly to the SB3 constructor. No custom abstraction needed.

---

## Module Plan

### 1. `config.py` — Parse and Validate

**Responsibilities:**
- Load YAML config file
- Resolve `_base` inheritance (merge parent algo config with child overrides)
- Validate algorithm/problem compatibility (DQN + continuous action space → skip with warning)
- Expand the experiment matrix: each `(problem, algorithm, seed)` triple becomes one `RunConfig`

**Core data structures:**

```python
@dataclass
class RunConfig:
    experiment_name: str
    env_id: str
    algo_name: str
    algo_class: str          # actual SB3 class name (e.g. "a2c" for reinforce)
    algo_kwargs: dict        # everything passed to constructor
    seed: int
    total_timesteps: int
    eval_freq: int
    n_eval_episodes: int
    log_dir: str
```

**Compatibility filter logic:**

```python
import gymnasium as gym

def is_compatible(env_id: str, algo_name: str) -> bool:
    env = gym.make(env_id)
    is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    env.close()
    if algo_name == "dqn" and not is_discrete:
        return False
    return True
```

---

### 2. `runner.py` — Execute a Single Run

**Responsibilities:**
- Takes a `RunConfig`, creates the environment wrapped with `Monitor`
- Instantiates the SB3 model
- Attaches an `EvalCallback` for periodic evaluation
- Calls `model.learn()`, catches divergence/NaN errors
- Saves the trained model checkpoint

**Algorithm map:**

```python
from stable_baselines3 import DQN, PPO, A2C

ALGO_MAP = {
    "dqn": DQN,
    "ppo": PPO,
    "a2c": A2C,
    "reinforce": A2C,  # with specific kwargs from config
}
```

**Run function sketch:**

```python
def run_experiment(config: RunConfig) -> RunResult:
    env = Monitor(gym.make(config.env_id))

    AlgoClass = ALGO_MAP[config.algo_name]
    model = AlgoClass(
        env=env,
        seed=config.seed,
        verbose=0,
        **config.algo_kwargs,
    )

    eval_env = Monitor(gym.make(config.env_id))
    eval_callback = EvalCallback(
        eval_env,
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        log_path=config.log_dir,
        best_model_save_path=config.log_dir,
    )

    model.learn(
        total_timesteps=config.total_timesteps,
        callback=eval_callback,
    )
    model.save(os.path.join(config.log_dir, "model"))

    return RunResult(config=config, log_dir=config.log_dir)
```

---

### 3. `tracking.py` — MLFlow Integration

Local file-based MLFlow tracking (`./mlruns/`, no server). Browse with `mlflow ui`.

MLFlow experiment = experiment group name. MLFlow run = one `(algo, env, seed)` combo.

**Post-training logging** (called by runner after training):
- Log params: `env_id`, `algo_name`, `seed`, all algo kwargs
- Run `evaluate_policy()`, log `eval/mean_reward` and `eval/std_reward` as metrics
- Log artifacts: saved model `.zip`, `evaluations.npz`, per-run learning curve plot

**Skip-if-exists**: before launching a run, query MLFlow for runs with matching params. Skip already-completed runs so re-running the script only fills in missing ones.

---

### 4. `__main__.py` — CLI Entry Point

Load config, expand matrix, loop over `RunConfig`s. For each: check MLFlow for existing run, skip or execute, log results. Per-run error handling (log and continue). `--dry-run` flag to print the matrix without training.

---

### 5. `analysis.py` — Cross-Run Comparison

Query MLFlow via `MlflowClient`, group by `(env, algo)`, aggregate across seeds. Produce comparison plots (one per env, shaded confidence bands) and a summary table.

---

## Implementation Order

| Phase | Module | Deliverable | Notes |
|---|---|---|---|
| 1 | `config.py` | YAML -> list of `RunConfig` | Print the expanded matrix to verify combos |
| 2 | `runner.py` | Single end-to-end run | Hardcode PPO + CartPole first, get training working |
| 3 | `tracking.py` | MLFlow integration | Post-training logging + skip-if-exists |
| 4 | `__main__.py` | CLI entry point | Loop over matrix, error handling, --dry-run |
| 5 | `analysis.py` | Cross-run comparison | Query MLFlow, aggregate across seeds, comparison plots |
| 6 | Extras | Parallel runs, curiosity wrappers, Optuna sweeps | Optional extensions |

---

## Phase 7: Live MLFlow Training Monitoring

Currently MLFlow logging is post-training only (final eval metrics + artifacts). This phase adds step-by-step metric logging during training so the MLFlow UI shows learning curves, not just single final values.

**Key change**: The MLFlow run must be open *during* training, not just after. This means restructuring the run lifecycle.

### Approach

1. **`MlflowEvalCallback`** in `tracking.py` -- subclass SB3's `EvalCallback`, override `_on_step` to log eval metrics to MLFlow with `step=self.num_timesteps` after each evaluation.

2. **Restructure run lifecycle** -- split `log_run()` into:
   - `start_mlflow_run(config)` context manager: opens run, logs params
   - `log_run_artifacts(result)`: logs post-training artifacts within the already-open run

3. **Update `__main__.py`** flow:
   ```python
   with start_mlflow_run(config):
       result = run_experiment(config)   # callback logs live metrics
       log_run_artifacts(result)         # post-training artifacts
   ```

4. **Update `runner.py`** -- use `MlflowEvalCallback` instead of plain `EvalCallback`.

### Files to modify

- `tracking.py` -- add callback + context manager, refactor `log_run`
- `runner.py` -- swap callback class
- `__main__.py` -- wrap training in context manager
- `tests/test_tracking.py` -- update for new API
- `README.md` -- document live monitoring in the MLFlow section

---

## Future Extensions
- **Curiosity wrappers**: SB3 doesn't include ICM/RND natively. These can be implemented as Gymnasium reward wrappers that add an intrinsic reward component. This would appear in the config as an `env_wrappers` list under each experiment.
- **Hyperparameter sweeps**: Integrate Optuna by turning algorithm config values into search spaces (e.g. `learning_rate: [0.0001, 0.001]` triggers a sweep). MLFlow parent/child runs can group sweep trials.
- **Custom callbacks**: Episode video recording, gradient norm tracking, or early stopping on plateau.
