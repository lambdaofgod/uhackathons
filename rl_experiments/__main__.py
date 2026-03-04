"""CLI entry point for the RL experiment runner."""

import argparse
import logging

from rl_experiments.analysis import print_summary
from rl_experiments.config import expand_matrix, load_config
from rl_experiments.runner import run_experiment
from rl_experiments.tracking import log_run, run_exists, setup_tracking

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="RL Experiment Runner")
    parser.add_argument(
        "--config",
        default="experiments.yaml",
        help="Path to experiments YAML config (default: experiments.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the experiment matrix without training",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Print summary table and save comparison plots from existing runs",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config = load_config(args.config)
    runs = expand_matrix(config)

    if args.analyze:
        setup_tracking()
        print_summary()
        return

    if args.dry_run:
        print(f"Experiment matrix: {len(runs)} runs\n")
        for i, run in enumerate(runs, 1):
            print(
                f"  {i:3d}. {run.experiment_name} | "
                f"{run.algo_name:12s} | {run.env_id:30s} | seed={run.seed}"
            )
        return

    setup_tracking()

    completed = 0
    skipped = 0
    failed = 0

    for i, run in enumerate(runs, 1):
        tag = f"[{i}/{len(runs)}]"

        if run_exists(run):
            logger.info(f"{tag} Skipping (already exists): {run.algo_name}/{run.env_id}/seed{run.seed}")
            skipped += 1
            continue

        try:
            logger.info(f"{tag} Starting: {run.algo_name}/{run.env_id}/seed{run.seed}")
            result = run_experiment(run)
            log_run(result)
            completed += 1
        except Exception:
            logger.exception(f"{tag} Failed: {run.algo_name}/{run.env_id}/seed{run.seed}")
            failed += 1

    logger.info(
        f"Done: {completed} completed, {skipped} skipped, {failed} failed "
        f"(out of {len(runs)} total)"
    )


if __name__ == "__main__":
    main()
