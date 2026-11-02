# Google Colab execution

The [notebook](../notebooks/colab_train.ipynb) is a setup entry point. All runtime
logic is in `src/geoid/training/`; `scripts/train.py` is the shared CLI. Model
construction, data loading and the training loop are not implemented. Today,
`--dry-run` prints a runtime plan; execution without it fails explicitly. No data
download, checkpoint creation or experiment record is triggered by preflight.

## Setup

1. Read AGENTS.md, architecture, contracts and ADR-0005. Obtain approval for
   relevant research decisions before implementing or executing real experiments.
2. Once a reviewed commit is available remotely, open the notebook in Colab.
   This local branch has not been pushed automatically. Choose a Python 3.11/3.12
   runtime and select a GPU accelerator for the Colab environment configuration.
3. Mount Drive interactively. Set COMMIT to the exact reviewed 40-character SHA.
   Checkout refuses dirty worktrees and a mismatched remote. Never put a token in
   the notebook; private repository authentication is separate user setup.
4. The notebook installs pinned uv, creates a virtual environment with access to
   Colab system packages (including GPU PyTorch), and installs the core lock.
   Sync/run explicitly select Colab's interpreter so the local .python-version
   preference cannot replace the GPU environment with another Python installation.
   Do not replace the provider's CUDA stack casually. Torch is not in the core
   lock; a future experiment must record/pin its version and environment snapshot.
5. Set dataset, checkpoint and run directories to your approved Drive locations.
   These must refer to already verified data. No dataset acquisition occurs here.
6. Invoke the shared pipeline with DRY_RUN=True. CUDA absence is a clear failure,
   not silent CPU training. Leave real execution disabled during bootstrap.

Colab resources are variable and sessions expire; mounted Drive can also have
I/O failures. Persist verified checkpoints and run records, and test recovery.
See the [official Colab FAQ](https://research.google.com/colaboratory/faq.html).

## Environment configuration

`configs/environments/local.yaml` chooses auto device and auto precision: CPU uses
fp32, CUDA uses fp16. `colab.yaml` requires CUDA. bf16 is accepted only on a device
that reports support. Research configs, splits, labels and loss choices are shared
and cannot be changed by selecting an environment. AMP choice is part of runtime
provenance and checkpoint identity; changing it creates a new comparison condition.

Paths use `${NAME:-fallback}` syntax, expanded by package code. Relative paths
resolve against the repository, independent of the caller's working directory.
Supported overrides: GEOID_DATA_ROOT, GEOID_CHECKPOINT_ROOT, GEOID_RUN_ROOT.
No machine-specific path is embedded in portable source; Colab defaults live only
in its environment config and notebook setup.

PowerShell local preflight:

```powershell
uv sync --locked --group dev --no-editable
$env:GEOID_DATA_ROOT = 'data/raw/geoid-flood'
uv run --locked --no-editable python scripts/train.py --config configs/experiments/baseline.yaml --environment configs/environments/local.yaml --dry-run
```

Non-editable installation avoids a Windows Store Python/Unicode editable-path
problem observed during bootstrap. After source edits, uv rebuilds the local
package when running/syncing using explicit source-file cache keys in pyproject.toml.
The project targets Python 3.11; 3.12 is supported
for compatible hosted runtimes. Other runtime Python versions require a reviewed
dependency update before use.

## Checkpoint and resume contract

`checkpoint_due` implements configurable periodic saves in completed epochs.
The future trainer must call capture_state/save_checkpoint after a complete epoch
and advance scheduler consistently before capture. Include model, optimizer,
scheduler, AMP scaler, next epoch, global step, sampler state, Python/NumPy/torch
and CUDA RNG state, plus Git/config/manifest/protocol/precision/framework identity.
Files are written through a temporary file then replaced; Drive filesystem
atomicity/durability is not assumed. Keep the last known-good checkpoint separately.

`restore_checkpoint` validates provenance and component compatibility, uses
`weights_only=True`, restores all supplied components and RNG, then returns the
cursor for the trainer. The model/optimizer/scheduler/scaler/sampler must be
constructed compatibly first. Use trusted files; restricted loading is not a
general trust guarantee. See [PyTorch serialization](https://docs.pytorch.org/docs/stable/notes/serialization.html).

Resume via `--resume path/to/checkpoint.pt` (or environment YAML resume). Paths are
never auto-selected by most-recent timestamp. This is full epoch-boundary state
resume, not a weights-only warm start. A disconnect loses work since the last
completed checkpoint. Mid-epoch/persistent-worker/prefetch resume is unsupported;
future loaders must use reproducibly seeded epoch workers and stateful sampling.
Cross-version/device bitwise equivalence is not promised. Actual training resume
is pending integration; bootstrap tests cover utilities only.

The precision helper returns autocast and, for fp16, a GradScaler. The future
trainer must use scaler.scale/backward, scaler.step and scaler.update correctly;
there is no AMP training loop in the notebook. API reference:
[PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html).

## Tracking and governance

JSON run records can persist to the environment runs path via the package writer;
records require the shared experiment metadata and evidence rules. Record both
research config hash and resolved runtime hash, torch/CUDA versions, immutable
checkout SHA, manifest and preprocessing hashes, seed and hyperparameters. Save
an environment snapshot with future runs. Resumed segments retain experiment
identity and record checkpoint provenance; do not fabricate completed runs.
MLflow is a future backend configured by URI/secret environment; selecting its
name does not deploy or implement it. The reporter excludes unverified results.

Notebooks/configs/runtime and checkpoint code are protected paths. ADR-0005 and
Lead review are required for integration; changes to methodology still require
the corresponding L3 ADR and experiment-impact review.
