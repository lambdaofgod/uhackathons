"""Cross-run analysis: query MLFlow, aggregate across seeds, plot comparisons."""

import logging

import matplotlib.pyplot as plt
import pandas as pd
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


def fetch_results(tracking_uri: str = "./mlruns") -> pd.DataFrame:
    """Query all completed runs from MLFlow and return a tidy DataFrame.

    Columns: env_id, experiment_group, algo_name, seed, mean_reward, std_reward
    """
    client = MlflowClient(tracking_uri=tracking_uri)

    rows = []
    for exp in client.search_experiments():
        if exp.name == "Default":
            continue
        for run in client.search_runs(experiment_ids=[exp.experiment_id]):
            params = run.data.params
            metrics = run.data.metrics
            if "eval/mean_reward" not in metrics:
                continue
            rows.append({
                "env_id": exp.name,
                "experiment_group": run.data.tags.get("experiment_group", ""),
                "algo_name": params.get("algo_name", ""),
                "seed": int(params.get("seed", 0)),
                "mean_reward": metrics["eval/mean_reward"],
                "std_reward": metrics.get("eval/std_reward", 0.0),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("No completed runs found in MLFlow")
    return df


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results across seeds: mean and std of mean_reward per (env, algo)."""
    if df.empty:
        return df
    grouped = (
        df.groupby(["env_id", "algo_name"])["mean_reward"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "reward_mean", "std": "reward_std", "count": "n_seeds"})
        .reset_index()
        .sort_values(["env_id", "reward_mean"], ascending=[True, False])
    )
    return grouped


def plot_comparison(df: pd.DataFrame, output_dir: str = "./results") -> list[str]:
    """Create one bar chart per environment comparing algorithms.

    Returns list of saved figure paths.
    """
    import os

    os.makedirs(output_dir, exist_ok=True)
    saved = []

    if df.empty:
        logger.warning("No data to plot")
        return saved

    for env_id, env_df in df.groupby("env_id"):
        agg = (
            env_df.groupby("algo_name")["mean_reward"]
            .agg(["mean", "std"])
            .sort_values("mean", ascending=True)
        )

        fig, ax = plt.subplots(figsize=(8, max(3, len(agg) * 0.8)))
        ax.barh(agg.index, agg["mean"], xerr=agg["std"], capsize=4)
        ax.set_xlabel("Mean Reward")
        ax.set_title(str(env_id))
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()

        path = os.path.join(output_dir, f"comparison_{env_id}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)
        logger.info(f"Saved comparison plot: {path}")

    return saved


def print_summary(tracking_uri: str = "./mlruns", output_dir: str = "./results") -> None:
    """Fetch results, print summary table, and save comparison plots."""
    df = fetch_results(tracking_uri)
    if df.empty:
        print("No completed runs found. Run experiments first.")
        return

    table = summary_table(df)
    print("\n=== Results Summary ===\n")
    print(table.to_string(index=False, float_format="%.2f"))
    print()

    plots = plot_comparison(df, output_dir)
    if plots:
        print(f"Saved {len(plots)} comparison plot(s) to {output_dir}/")
