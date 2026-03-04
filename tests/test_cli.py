"""Tests for rl_experiments.__main__ (CLI)."""

import os
import subprocess
import sys


class TestDryRun:
    def test_dry_run_prints_matrix(self):
        """--dry-run should print the experiment matrix without training."""
        result = subprocess.run(
            [sys.executable, "-m", "rl_experiments", "--config", "experiments.yaml", "--dry-run"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        output = result.stdout

        # Should show run count
        assert "runs" in output.lower()

        # Should list algorithms and environments from experiments.yaml
        assert "ppo" in output
        assert "CartPole-v1" in output

    def test_dry_run_no_mlruns_created(self, tmp_path):
        """--dry-run should not create any mlruns directory."""
        subprocess.run(
            [sys.executable, "-m", "rl_experiments", "--config", "experiments.yaml", "--dry-run"],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
        )
        assert not os.path.exists(tmp_path / "mlruns")
