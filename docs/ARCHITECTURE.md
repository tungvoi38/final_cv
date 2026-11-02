# Research architecture

Status: proposed research design, bootstrap infrastructure only. Research Lead:
@tungvoi38. Contract version: `0.1-proposed`. No research ADR is accepted yet.

```text
Official revision + split metadata -> immutable raw manifest
 -> event/split audit -> split-local sampling -> chips
 -> modality preprocessing -> training-only augmentation
 -> early/intermediate fusion -> segmentation backbone + decoder
 -> CE + Dice (+ optional source-event DG regularizer)
 -> event evaluation -> versioned run record -> status report
```

| Module | Responsibility and dependency direction | Owner role |
| --- | --- | --- |
| data.manifest/download | Provenance, safe acquisition and verification | Preprocessing 1 |
| data.dataset/sampling | Manifest to samples, preserve event and parent-tile splits | Preprocessing 1 |
| data.preprocess/augmentation | Aligned SAR, optical, DEM transforms; ClassMix | Preprocessing 2 |
| models.resnet_unet/fusion | ResNet50-U-Net and fusion | Model 1 |
| models.terramind | TerraMind-B adapters and decoder | Model 2 |
| dg | MixStyle, Deep CORAL, class-conditional CORAL | Model 3 |
| training/evaluation | Orchestration, losses, selection and metrics | Model 3 |
| tracking | Run metadata boundary; proposed MLflow integration | Model 3 |

Data and models must not import the trainer. Models expose logits and alignment
features through [CONTRACTS.md](CONTRACTS.md). The trainer consumes these boundaries
and DG results; evaluation never fits preprocessing or updates models. Tracking
consumes metadata and metrics and must not alter research computations.

ResNet50 with ImageNet initialization and early fusion is the proposed baseline;
intermediate fusion is a separate experiment. TerraMind-B requires an explicit
adapter feasibility study: temporal SAR support and checkpoint compatibility are
unverified. No checkpoint URL or automatic weight download is configured.

Infrastructure uses Python 3.11, NumPy synthetic arrays, PyYAML, pytest, Ruff and
GitHub Actions. PyTorch, raster libraries and MLflow are deliberately deferred to
implementation PRs with dependency and reproducibility review. NumPy contract
checks validate boundary examples; they do not implement research models.

## Scientific blockers

The live dataset card differs from earlier indexed descriptions and the paper
release. Pinning a dataset revision is mandatory. In particular, AoI-disjointness
does not prove event-disjointness; the card's sample includes two AoIs of one
activation in different splits. Preserve official assignments, audit event overlap,
and stop for ADR-0001 review if they conflict with the proposed unseen-event task.
Never silently manufacture an event-disjoint split. See [data provenance](../data/README.md).

## Evolution

Local and Colab use the same CLI and module boundaries. Environment configs select
paths/device/precision; training.runtime resolves them, training.checkpoints handles
full epoch-state persistence, and tracking writes run records. The notebook contains
setup only. These utilities await real-trainer integration; see [COLAB.md](COLAB.md)
and proposed ADR-0005. Runtime support does not approve research algorithms.

Stable interfaces enable five developers to use separate branches/worktrees.
Shared contract changes go through an ADR before dependent implementations merge.
L2/L3 approval is human review, never a boolean written by an agent. Infrastructure
cannot prove scientific correctness from paths or passing tests. See
[contribution and GitHub setup](CONTRIBUTING.md) and [decisions](decisions/README.md).
