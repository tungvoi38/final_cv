# GEOID-Flood Domain Generalization

Domain generalization for multimodal flood segmentation across unseen flood events.
This repository is a lightweight foundation for five developers and their coding
agents. Research models, preprocessing, DG algorithms and training are **not yet
implemented**. All five initial research/runtime ADRs are **proposed**, not approved.

The proposed inputs are pre/post Sentinel-1 VV/VH, pre-event Sentinel-2 and DEM;
the proposed study compares ResNet50-U-Net and TerraMind-B with ClassMix,
MixStyle and CORAL variants. Dataset revision, event isolation and modality
semantics require verification before implementation.

## Start here

- [Architecture and responsibilities](docs/ARCHITECTURE.md)
- [Shared contracts](docs/CONTRACTS.md) and [proposed ADRs](docs/decisions/README.md)
- [Contributing and GitHub setup](docs/CONTRIBUTING.md); [agent instructions](AGENTS.md)
- [Data provenance and acquisition policy](data/README.md)
- [Google Colab GPU setup](docs/COLAB.md) and [notebook](notebooks/colab_train.ipynb)
- [Milestones](docs/ROADMAP.md) and [generated status](reports/STATUS.md)

## Installation and checks

Use Python 3.11 (3.12 supported) and uv. From this repository in PowerShell:

```powershell
uv sync --locked --group dev --no-editable
uv run --locked --no-editable ruff check .
uv run --locked --no-editable ruff format --check .
uv run --locked --no-editable pytest -q
uv run --locked --no-editable python scripts/check_governance.py
uv run --locked --no-editable python scripts/generate_report.py
uv run --locked --no-editable python scripts/train.py --config configs/experiments/baseline.yaml --environment configs/environments/local.yaml --dry-run
```

Non-editable installation avoids the observed Windows Store Python Unicode-path
issue. The lock contains lightweight infrastructure dependencies only. Core tests
need no GPU, real data, weights, cloud credentials or MLflow server. Local CPU
preflight works without PyTorch. Colab uses its installed GPU torch through a
system-site-packages environment; its version is recorded outside the core lock.

## Layout

```text
.github/             ownership, issue/PR templates, CI/governance/report workflows
docs/                architecture, contracts, roadmap, Colab and proposed decisions
configs/             data, models, experiments and environment YAML
src/geoid/           data, models, DG, training, evaluation, tracking boundaries
scripts/             pipeline entry points, governance checks and report generator
tests/               synthetic contracts, safety, runtime and infrastructure checks
notebooks/           minimal Colab setup and CLI invocation
reports/             explicit progress records and generated status
data/                provenance documentation; raw/processed data are ignored
```

## Research and data policy

Preserve official splits and verify event disjointness; do not silently reconcile
an AoI split with an unseen-event claim. Fit normalization only on training data.
No bootstrap command downloads the dataset or trains models. The acquisition CLI
defaults to dry run and stops until an immutable official revision and explicit
file sizes/checksums are configured. Keep data in `data/raw/geoid-flood/`, derived
artifacts in `data/processed/`, and weights/run artifacts in ignored directories.

## Collaboration and reporting

Use scoped feature branches, attributed PRs and L0-L3 decision classification.
L2/L3 requires an ADR and Research Lead review. CODEOWNERS routes initial review
to @tungvoi38; GitHub protections still require administrator setup and verification.
AGENTS.md is canonical across tools; load it explicitly if a tool does not.

The offline report reads Git metadata, ADRs, explicit progress evidence, roadmap
and any reviewed run records. `--github` adds bounded read-only API snapshots;
unavailable sources remain visible. CI publishes reports as artifacts and summaries,
never uncontrolled commits. No experiments or benchmark results are claimed.
