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

- [ ] Define `ALGO_MAP` and `RunResult`
- [ ] Implement `run_experiment(config: RunConfig) -> RunResult`
- [ ] Smoke test: PPO + CartPole, verify training completes and model saves

## Phase 2.5: Config Validation

- [ ] Introspect SB3 class constructor signatures (`inspect.signature`)
- [ ] Validate all YAML kwargs against the resolved SB3 class (handle `_base` aliases)
- [ ] Fail fast at config-load time with clear errors

## Phase 3: Tracking (`rl_experiments/tracking.py`)

- [ ] Post-training logging: params, `evaluate_policy()` metrics, artifacts
- [ ] Skip-if-exists: query MLFlow for matching params before launching a run
- [ ] Local file-based setup (`./mlruns/`)

## Phase 4: CLI Entry Point (`rl_experiments/__main__.py`)

- [ ] Load config, expand matrix, loop with per-run error handling
- [ ] Progress logging
- [ ] `--dry-run` mode

## Phase 5: Analysis (`rl_experiments/analysis.py`)

- [ ] Query MLFlow, group by `(env, algo)`, aggregate across seeds
- [ ] Comparison plots: one figure per env, shaded confidence bands
- [ ] Summary table

## Phase 6: Integration & Polish

- [ ] End-to-end test: full config through all phases
