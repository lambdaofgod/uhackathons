# RL Experiment Runner - Implementation Tasks

Based on `experiment_runner_plan.md`. Parallelism is out of scope for now.

---

## Phase 0: Environment Setup

- [x] Add dependencies: `stable-baselines3`, `gymnasium`, `pyyaml`, `matplotlib`, `pandas`, `pydantic`, `mlflow`
- [x] Verify imports work
- [x] Create `rl_experiments/` package

## Phase 1: Config (`rl_experiments/config.py`)

- [x] Define Pydantic models (`RunConfig`, `TrainingConfig`, `ExperimentGroup`, `FullConfig`)
- [x] Load and parse YAML config file
- [x] Implement `_base` inheritance resolution
- [x] Implement compatibility filter: skip DQN on continuous envs
- [x] Expand experiment matrix: `(problem x algorithm x seed)` -> list of `RunConfig`
- [x] Write default `experiments.yaml`
- [x] Test: print expanded matrix, verify combos

## Phase 2: Runner (`rl_experiments/runner.py`)

- [x] Define `ALGO_MAP` and `RunResult`
- [x] Implement `run_experiment(config: RunConfig) -> RunResult`
- [x] Smoke test: PPO + CartPole, verify training completes and model saves

## Phase 2.5: Config Validation

- [x] Introspect SB3 class constructor signatures (`inspect.signature`)
- [x] Validate all YAML kwargs against the resolved SB3 class (handle `_base` aliases)
- [x] Fail fast at config-load time with clear errors

## Phase 2.6: Tests (`tests/`)

- [x] Add `pytest` dependency, create `tests/` dir with `conftest.py`
- [x] `test_config.py`: resolve_algo_config, validate_algo_kwargs, is_compatible, expand_matrix
- [x] `test_runner.py`: PPO + CartPole integration test (small timesteps), verify model save and RunResult

## Phase 3: Tracking (`rl_experiments/tracking.py`)

- [x] Post-training logging: params, `evaluate_policy()` metrics, artifacts
- [x] Skip-if-exists: query MLFlow for matching params before launching a run
- [x] Local file-based setup (`./mlruns/`)
- [x] Test skip-if-exists with duplicate run

## Phase 4: CLI Entry Point (`rl_experiments/__main__.py`)

- [x] Load config, expand matrix, loop with per-run error handling
- [x] Progress logging
- [x] `--dry-run` mode
- [x] Test --dry-run outputs matrix without training

## Phase 5: Analysis (`rl_experiments/analysis.py`)

- [ ] Query MLFlow, group by `(env, algo)`, aggregate across seeds
- [ ] Comparison plots: one figure per env, shaded confidence bands
- [ ] Summary table

## Phase 6: Integration & Polish

- [ ] End-to-end test: full config through all phases
