"""Exercise real script entry points so import-path errors cannot hide behind pytest."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def command(script, *args):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )


def test_report_cli(tmp_path):
    output = tmp_path / "STATUS.md"
    result = command("generate_report.py", "--output", str(output))
    assert result.returncode == 0, result.stderr
    assert "No verified experiments available." in output.read_text(encoding="utf-8")


def test_training_preflight_and_unimplemented_execution():
    args = (
        "--config",
        "configs/experiments/baseline.yaml",
        "--environment",
        "configs/environments/local.yaml",
    )
    result = command("train.py", *args, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "experiment_config_hash" in result.stdout
    result = command("train.py", *args)
    assert result.returncode == 2 and "not implemented" in result.stderr


def test_download_cli_fails_closed():
    result = command("download_data.py", "--dry-run")
    assert result.returncode == 2 and "immutable official dataset" in result.stderr


def test_governance_cli():
    result = command("check_governance.py")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "0 hard failures" in result.stdout
