"""Offline governance checks. Success is not human approval."""

import argparse
import json
import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = (
    "Context",
    "Decision",
    "Rationale",
    "Alternatives Considered",
    "Consequences",
    "Testing / Verification",
    "Approval Status",
)
PROTECTED = (
    "notebooks/",
    "docs/COLAB.md",
    "AGENTS.md",
    "docs/ARCHITECTURE.md",
    "docs/CONTRACTS.md",
    "docs/decisions/",
    "configs/",
    "src/geoid/",
    "scripts/",
    ".github/",
    "pyproject.toml",
    "uv.lock",
    "tests/",
    "data/",
    "reports/progress.yaml",
)


def read_adr(path):
    text = path.read_text(encoding="utf-8")
    match = re.match(r"\A---\r?\n(.*?)\r?\n---(?:\r?\n|$)", text, re.S)
    if not match:
        raise ValueError(f"{path.name}: missing YAML front matter")
    meta = yaml.safe_load(match[1])
    if not isinstance(meta, dict):
        raise ValueError(f"{path.name}: metadata must be a mapping")
    for key in ("id", "title", "status", "proposed_by", "related_issue", "related_pr", "impact"):
        if key not in meta:
            raise ValueError(f"{path.name}: missing {key}")
    if not re.fullmatch(r"ADR-\d{4}", str(meta["id"])):
        raise ValueError(f"{path.name}: invalid ADR id")
    if not path.name.startswith(meta["id"][4:] + "-"):
        raise ValueError(f"{path.name}: filename and ADR id disagree")
    if meta["status"] not in {"proposed", "accepted", "rejected", "superseded"}:
        raise ValueError(f"{path.name}: invalid status")
    if not isinstance(meta["title"], str) or not meta["title"].strip():
        raise ValueError(f"{path.name}: empty title")
    proposer = meta["proposed_by"]
    if not isinstance(proposer, dict) or any(
        not isinstance(proposer.get(k), str) or not proposer[k].strip()
        for k in ("developer", "agent", "session_id")
    ):
        raise ValueError(f"{path.name}: incomplete attribution")
    if (
        not isinstance(meta["impact"], list)
        or not meta["impact"]
        or any(not isinstance(x, str) or not x.strip() for x in meta["impact"])
    ):
        raise ValueError(f"{path.name}: impact must be a nonempty string list")
    for section in SECTIONS:
        if not re.search(rf"^## {re.escape(section)}\s*$", text, re.M):
            raise ValueError(f"{path.name}: missing section {section}")
    if meta["status"] in {"accepted", "superseded"}:
        approval = meta.get("approval", {})
        if (
            not isinstance(approval, dict)
            or approval.get("reviewer") != "@tungvoi38"
            or not str(approval.get("evidence", "")).startswith("https://github.com/")
        ):
            raise ValueError(f"{path.name}: accepted history needs Lead review evidence")
    return meta


def protected(path):
    return any(path == p or (p.endswith("/") and path.startswith(p)) for p in PROTECTED)


def check_pr(body, changed, adrs):
    errors, reviews = [], []
    match = re.search(r"```yaml governance\s*\n(.*?)\n```", body, re.S)
    if not match:
        return ["PR: missing yaml governance block"], []
    try:
        meta = yaml.safe_load(match[1])
    except yaml.YAMLError as exc:
        return [f"PR: malformed governance YAML: {exc}"], []
    if not isinstance(meta, dict):
        return ["PR: governance metadata must be a mapping"], []
    required = (
        "responsible_developer",
        "agent_tool",
        "agent_model",
        "agent_session",
        "related_issue",
        "commits",
        "architecture_impact",
        "experiment_impact",
    )
    for key in required:
        value = meta.get(key)
        if not isinstance(value, str) or not value.strip() or "REPLACE_WITH" in value:
            errors.append(f"PR: fill {key}")
    level = meta.get("decision_level")
    if level not in {"L0", "L1", "L2", "L3"}:
        errors.append("PR: decision_level must be L0-L3")
    refs = meta.get("adrs")
    if not isinstance(refs, list) or any(not isinstance(x, str) for x in refs):
        errors.append("PR: adrs must be a string list")
        refs = []
    for ref in refs:
        if ref not in adrs:
            errors.append(f"PR: unknown ADR {ref}")
        elif adrs[ref]["status"] in {"rejected", "superseded"}:
            errors.append(f"PR: {ref} is not a current decision")
    if meta.get("lead_review") not in {"required", "not_required"}:
        errors.append("PR: lead_review must be required or not_required")
    sensitive = [p for p in changed if protected(p)]
    if sensitive or level in {"L2", "L3"}:
        if meta.get("lead_review") != "required":
            errors.append("PR: protected changes require lead_review: required")
        reviews.append(
            "Lead must review impact and current-head changes; metadata is not approval."
        )
    if level in {"L2", "L3"} and not refs:
        errors.append("PR: L2/L3 changes require an ADR reference")
    if level == "L3" and meta.get("experiment_impact", "").lower() in {"none", "unknown"}:
        errors.append("PR: L3 requires a reasoned experiment impact review")
    if sensitive:
        reviews.append("Sensitive paths: " + ", ".join(sensitive))
    return errors, reviews


def changed_files(root, base=None):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root).decode("utf-8").split("\0")

    result = set(git("diff", "--name-only", "-z", "HEAD"))
    result.update(git("ls-files", "--others", "--exclude-standard", "-z"))
    if base:
        # --no-renames includes both old and new paths, including protected deletions.
        result.update(git("diff", "--no-renames", "--name-only", "-z", f"{base}...HEAD"))
    return sorted(p for p in result if p)


def check_repository(root, changed, body=None):
    from geoid.config import validate_configs

    errors, warnings, reviews, adrs = [], [], [], {}
    paths = sorted((root / "docs/decisions").glob("[0-9]*.md"))
    if not paths:
        errors.append("No ADR records found")
    for path in paths:
        try:
            meta = read_adr(path)
            if meta["id"] in adrs:
                raise ValueError(f"Duplicate {meta['id']}")
            adrs[meta["id"]] = meta
        except (ValueError, yaml.YAMLError, OSError) as exc:
            errors.append(str(exc))
    for meta in adrs.values():
        if meta.get("supersedes") and meta["supersedes"] not in adrs:
            errors.append(f"{meta['id']}: unknown supersedes reference")
    try:
        validate_configs(root)
    except (ValueError, yaml.YAMLError, OSError) as exc:
        errors.append(f"Configuration: {exc}")
    if body is None:
        warnings.append("PR metadata unavailable: local/offline checks do not verify reviews.")
        if any(protected(p) for p in changed):
            reviews.append("Protected changes detected; complete PR metadata and Lead review.")
    else:
        pr_errors, pr_reviews = check_pr(body, changed, adrs)
        errors.extend(pr_errors)
        reviews.extend(pr_reviews)
    proposed = [key for key, value in adrs.items() if value["status"] == "proposed"]
    if proposed:
        reviews.append("Research proposals await approval: " + ", ".join(proposed))
    return errors, warnings, reviews


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--base", help="Git base ref; failures are not silently ignored")
    parser.add_argument("--pr-body", type=Path)
    parser.add_argument("--event", type=Path, help="GitHub pull_request event JSON")
    args = parser.parse_args()
    try:
        body = args.pr_body.read_text(encoding="utf-8") if args.pr_body else None
        if args.event:
            event = json.loads(args.event.read_text(encoding="utf-8"))
            body = event["pull_request"].get("body") or ""
        changed = changed_files(args.root, args.base)
        errors, warnings, reviews = check_repository(args.root, changed, body)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"HARD FAILURE: cannot inspect governance context: {exc}")
        return 1
    for label, items in (
        ("HARD FAILURE", errors),
        ("WARNING", warnings),
        ("REQUIRES HUMAN REVIEW", reviews),
    ):
        for item in items:
            print(f"{label}: {item}")
    print(
        f"Governance: {len(errors)} hard failures, {len(warnings)} warnings, "
        f"{len(reviews)} human-review notices. This does not grant approval."
    )
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
