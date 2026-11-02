import copy
import json
from pathlib import Path

import pytest
import yaml

from geoid.tracking.experiments import comparable, validate_run, write_run
from scripts.generate_report import generate, github_activity

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def run_record():
    return dict(
        experiment_id="synthetic",
        run_id="fixture",
        responsible_developer="fixture",
        agent="none",
        git_sha="a" * 40,
        architecture_version="fixture",
        config_hash="b" * 64,
        manifest_hash="c" * 64,
        preprocessing_version="fixture",
        model="test-fixture",
        fusion="fixture",
        dg_method="none",
        seed=1,
        hyperparameters={},
        evaluation_protocol="fixture",
        metrics={},
        status="planned",
    )


def test_tracking_validation_and_comparability(run_record, tmp_path):
    validate_run(run_record)
    assert comparable(run_record, copy.deepcopy(run_record))
    other = {**run_record, "manifest_hash": "d" * 64}
    assert not comparable(run_record, other)
    path = write_run(run_record, tmp_path)
    assert json.loads(path.read_text())["status"] == "planned"
    with pytest.raises(ValueError, match="identity"):
        write_run(other, tmp_path)
    with pytest.raises(ValueError, match="metrics"):
        validate_run({**run_record, "metrics": {"score": float("nan")}})


def test_offline_report_excludes_unverified_runs(tmp_path, run_record):
    write_run(
        {**run_record, "status": "completed", "metrics": {"synthetic_metric": 0.123}}, tmp_path
    )
    report = generate(ROOT, runs_dir=tmp_path)
    assert "No verified experiments available." in report
    assert "0.123" not in report
    assert "offline mode" in report
    assert "ADR-0005" in report
    assert "uncommitted changes" in report
    for section in ("A.", "B.", "C.", "D.", "E.", "F.", "G."):
        assert f"## {section}" in report


def test_report_lists_reviewed_fixture_without_ranking(tmp_path, run_record):
    record = {
        **run_record,
        "status": "completed",
        "metrics": {"synthetic_metric": 0.123},
        "verification": {"reviewer": "test fixture", "evidence": "synthetic test evidence"},
    }
    write_run(record, tmp_path)
    report = generate(ROOT, runs_dir=tmp_path)
    assert "0.123" in report and "without cross-protocol ranking" in report


def test_github_unavailable_is_explicit(monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("network unavailable")

    monkeypatch.setattr("scripts.generate_report.urlopen", unavailable)
    activity, missing = github_activity("owner/repo")
    assert activity == {} and len(missing) == 3


def test_workflow_structure_and_immutable_actions():
    import re

    for path in (ROOT / ".github/workflows").glob("*.yml"):
        workflow = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
        assert workflow["permissions"]["contents"] == "read"
        assert "pull_request_target" not in workflow["on"]
        for job in workflow["jobs"].values():
            for step in job["steps"]:
                if "uses" in step:
                    assert re.fullmatch(r"[\w-]+/[\w-]+@[0-9a-f]{40}", step["uses"])
