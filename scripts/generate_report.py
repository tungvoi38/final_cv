"""Evidence-based Markdown report with explicit offline and integration gaps."""

import argparse
import json
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

from geoid.config import load_yaml, require
from geoid.tracking.experiments import validate_run

if __package__:
    from .check_governance import read_adr
else:
    from check_governance import read_adr

ROOT = Path(__file__).resolve().parents[1]
STATUSES = {
    "Not started",
    "In progress",
    "Implemented",
    "Tested",
    "Integrated",
    "Validated",
    "Blocked",
}


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def github_activity(repository, token=None):
    require(
        bool(re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository)),
        "invalid GitHub repository",
    )
    sources, missing = {}, []
    endpoints = {
        "issues": "issues?state=all&sort=updated&per_page=30",
        "pulls": "pulls?state=closed&sort=updated&direction=desc&per_page=30",
        "checks": "actions/runs?per_page=30",
    }
    for name, endpoint in endpoints.items():
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "geoid-status-report"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            request = Request(
                f"https://api.github.com/repos/{repository}/{endpoint}", headers=headers
            )
            with urlopen(request, timeout=20) as response:
                sources[name] = json.load(response)
        except (OSError, ValueError):
            missing.append(f"GitHub {name} unavailable (permission/network/API error).")
    return sources, missing


def generate(root=ROOT, *, runs_dir=None, online=False):
    root = Path(root)
    progress = load_yaml(root / "reports/progress.yaml")
    require(isinstance(progress.get("components"), list), "progress needs components")
    for item in progress["components"]:
        require(
            isinstance(item, dict)
            and item.get("status") in STATUSES
            and bool(item.get("name"))
            and bool(item.get("evidence")),
            "invalid component status",
        )
    missing = ["MLflow unavailable: no tracking backend is configured."]
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root))
    except (OSError, subprocess.CalledProcessError):
        sha, dirty = "unavailable", None
        missing.append("Git metadata unavailable.")
    activity = {}
    if online:
        activity, gaps = github_activity(
            os.environ.get("GITHUB_REPOSITORY", "tungvoi38/final_cv"), os.environ.get("GH_TOKEN")
        )
        missing.extend(gaps)
    else:
        missing.append("GitHub issues, PRs and CI unavailable: offline mode.")
    stamp = datetime.now(UTC).isoformat()
    lines = [
        "# Research status",
        "",
        "## A. Project Summary",
        "",
        f"- Timestamp: {stamp}",
        f"- Git revision: {sha}",
        f"- Working tree has uncommitted changes: {dirty}",
        f"- Current milestone: {cell(progress.get('milestone', 'unavailable'))}",
        f"- Overall status: {cell(progress.get('overall_status', 'unavailable'))}",
        "",
        "## B. Architecture Progress",
        "",
        "| Component | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    lines.extend(
        f"| {cell(x['name'])} | {cell(x['status'])} | {cell(x['evidence'])} |"
        for x in progress["components"]
    )
    lines += [
        "",
        "Statuses are declared evidence records, not inferred from files or issue closure.",
        "Validated means experimentally validated; no component is promoted automatically.",
        "",
        "## C. Architecture Changes",
        "",
    ]
    adrs = [read_adr(p) for p in sorted((root / "docs/decisions").glob("[0-9]*.md"))]
    for status in ("proposed", "accepted", "rejected", "superseded"):
        records = [f"{a['id']} ({a['title']})" for a in adrs if a["status"] == status]
        lines.append(f"- {status.capitalize()}: {cell('; '.join(records) or 'None recorded')}")
    lines += [
        "- Awaiting Lead approval: "
        + ", ".join(a["id"] for a in adrs if a["status"] == "proposed"),
        "",
        "## D. Current Experimental Results",
        "",
    ]
    records, excluded = [], 0
    directory = Path(runs_dir) if runs_dir else root / "outputs/runs"
    if not directory.is_dir():
        missing.append("Local experiment record directory unavailable.")
    for path in sorted(directory.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            validate_run(record)
            if record["status"] == "completed" and record.get("verification") and record["metrics"]:
                records.append(record)
            else:
                excluded += 1
        except (ValueError, OSError):
            missing.append(f"Invalid experiment record excluded: {path.name}")
    if not records:
        lines.append("No verified experiments available.")
    else:
        lines += [
            "Records have review evidence; this report does not authenticate that evidence.",
            "Runs are listed without cross-protocol ranking.",
            "",
            "| Experiment / run | Model / DG | Metrics | Status | SHA | Dataset hash | Review |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for run in records:
            values = [
                f"{run['experiment_id']} / {run['run_id']}",
                f"{run['model']} / {run['dg_method']}",
                json.dumps(run["metrics"]),
                run["status"],
                run["git_sha"],
                run["manifest_hash"],
                run["verification"]["evidence"],
            ]
            lines.append("| " + " | ".join(cell(v) for v in values) + " |")
    lines += [
        f"\nUnverified/incomplete records excluded: {excluded}.",
        "",
        "## E. Current Blockers",
        "",
    ]
    lines.extend("- " + cell(b) for b in progress.get("blockers", []))
    for issue in activity.get("issues", []):
        if (
            "pull_request" not in issue
            and issue.get("state") == "open"
            and any(label["name"] == "blocked" for label in issue.get("labels", []))
        ):
            lines.append(f"- GitHub blocker #{issue['number']}: {cell(issue['title'])}")
    lines += ["", "## F. Recent Activity", ""]
    merged = [pr for pr in activity.get("pulls", []) if pr.get("merged_at")]
    lines.append("Merged PRs (up to 30 recently updated closed PRs):")
    lines.extend(f"- #{p['number']}: {cell(p['title'])}" for p in merged)
    if not merged:
        lines.append("- None in available snapshot; external absence is not proof of no activity.")
    lines.append("Completed tasks (explicit evidence only):")
    tasks = progress.get("completed_tasks", [])
    for task in tasks:
        require(
            isinstance(task, dict) and task.get("task") and task.get("evidence"),
            "completed task requires evidence",
        )
        lines.append(f"- {cell(task['task'])}: {cell(task['evidence'])}")
    if not tasks:
        lines.append("- No completed task records supplied.")
    failures = [
        run
        for run in activity.get("checks", {}).get("workflow_runs", [])
        if run.get("conclusion") in {"failure", "timed_out", "cancelled"}
    ]
    lines.append("Failed/cancelled checks (up to 30 recent workflow runs):")
    lines.extend(f"- {cell(r['name'])}: {r['conclusion']} ({r['html_url']})" for r in failures)
    if not failures:
        lines.append("- None in available snapshot; CI success is not inferred.")
    lines += [
        "Relevant architecture decisions: see section C.",
        "",
        "## G. Next Milestones",
        "",
        (root / "docs/ROADMAP.md").read_text(encoding="utf-8").split("\n", 1)[1].strip(),
        "",
        "## Data availability",
        "",
    ]
    lines.extend("- " + gap for gap in missing)
    lines.append("- GitHub snapshots, when requested, are bounded to 30 records per endpoint.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/STATUS.md")
    parser.add_argument("--runs-dir", type=Path)
    parser.add_argument("--github", action="store_true", help="Read-only optional GitHub snapshot")
    args = parser.parse_args()
    try:
        report = generate(runs_dir=args.runs_dir, online=args.github)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Report failed: {exc}\n")
    print(f"Wrote report: {args.output.name}")


if __name__ == "__main__":
    main()
