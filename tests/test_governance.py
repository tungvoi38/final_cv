import shutil
from pathlib import Path

import pytest
import yaml

from geoid.config import validate_configs
from scripts.check_governance import check_pr, check_repository, protected, read_adr

ROOT = Path(__file__).resolve().parents[1]


def body(**overrides):
    meta = dict(
        responsible_developer="unknown",
        agent_tool="Codex",
        agent_model="unknown",
        agent_session="not available",
        related_issue="local-task-test",
        commits="HEAD",
        decision_level="L2",
        architecture_impact="contract proposal",
        adrs=["ADR-0001"],
        lead_review="required",
        experiment_impact="No prior experiments; review before M1",
    )
    meta.update(overrides)
    return "```yaml governance\n" + yaml.safe_dump(meta) + "```"


def test_positive_pr():
    errors, reviews = check_pr(body(), ["docs/CONTRACTS.md"], {"ADR-0001": {"status": "proposed"}})
    assert not errors
    assert reviews


@pytest.mark.parametrize(
    "change",
    [
        {"adrs": []},
        {"adrs": ["ADR-9999"]},
        {"lead_review": "approved"},
        {"responsible_developer": ""},
        {"decision_level": "L7"},
        {"commits": "REPLACE_WITH_COMMIT_REFERENCES"},
        {"adrs": "ADR-0001"},
    ],
)
def test_bad_pr_metadata(change):
    errors, _ = check_pr(
        body(**change), ["configs/data/geoid.yaml"], {"ADR-0001": {"status": "proposed"}}
    )
    assert errors


def test_low_level_sensitive_edit_still_requires_review():
    errors, _ = check_pr(
        body(decision_level="L1", adrs=[], lead_review="not_required"),
        ["src/geoid/data/preprocess.py"],
        {},
    )
    assert any("require" in error for error in errors)


@pytest.mark.parametrize(
    "text", ["", "```yaml governance\n[broken\n```", "```yaml governance\n[]\n```"]
)
def test_missing_or_malformed_pr(text):
    assert check_pr(text, [], {})[0]


def test_adr_and_repository_configuration():
    validate_configs(ROOT)
    meta = read_adr(ROOT / "docs/decisions/0001-data-protocol.md")
    assert meta["status"] == "proposed"
    errors, warnings, reviews = check_repository(ROOT, ["configs/data/geoid.yaml"])
    assert not errors and warnings and reviews


@pytest.mark.parametrize(
    "mutation", ["status", "approval", "section", "attribution", "frontmatter"]
)
def test_bad_adr(tmp_path, mutation):
    source = ROOT / "docs/decisions/0001-data-protocol.md"
    text = source.read_text(encoding="utf-8")
    replacements = {
        "status": ("status: proposed", "status: magic"),
        "approval": ("status: proposed", "status: accepted"),
        "section": ("## Rationale", "## Missing"),
        "attribution": ("developer: unknown", "developer: null"),
        "frontmatter": ("---", "bad"),
    }
    old, new = replacements[mutation]
    target = tmp_path / source.name
    target.write_text(text.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(ValueError):
        read_adr(target)


@pytest.mark.parametrize("mutation", ["channels", "train", "missing", "malformed"])
def test_bad_config(tmp_path, mutation):
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    path = tmp_path / "configs/data/geoid.yaml"
    text = path.read_text(encoding="utf-8")
    if mutation == "missing":
        path.unlink()
    else:
        text = {
            "channels": text.replace("sar: 4", "sar: 5"),
            "train": text.replace("fit_split: train", "fit_split: test"),
            "malformed": "x: [",
        }[mutation]
        path.write_text(text, encoding="utf-8")
    with pytest.raises((ValueError, yaml.YAMLError)):
        validate_configs(tmp_path)


def test_protected_paths():
    assert protected(".github/workflows/governance.yml")
    assert protected("src/geoid/evaluation/metrics.py")
    assert protected("docs/CONTRACTS.md")
    assert not protected("reports/STATUS.md")
