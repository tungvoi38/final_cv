# Shared contracts — 0.1-proposed

All research semantics below await Research Lead approval. Validators enforce
this proposal for synthetic integration only. A major semantic change increments
the contract version and links a superseding accepted ADR in the implementation PR.

## Samples and batches

| Field | Unbatched / batched shape | Dtype / meaning |
| --- | --- | --- |
| sar | [4,H,W] / [B,4,H,W] | float32, proposed pre VV,VH then post VV,VH |
| optical | [12,H,W] / [B,12,H,W] | float32; band order must be revision-verified |
| dem | [1,H,W] / [B,1,H,W] | float32 |
| mask | [H,W] / [B,H,W] | int64, 0 background, 1 permanent water, 2 flood, 255 ignore |
| event_id, aoi_id, tile_id | nonempty string / B strings | source identifiers, never inferred from split |

Dimensions must be positive. All imagery is finite after preprocessing. All
modalities must share CRS, transform, extent, resolution and pixel registration;
matching array dimensions alone is insufficient. Metadata sidecars must retain
CRS/transform, parent tile, window, source revision, file checksums, split and
normalization provenance. The loader must check spatial metadata before emitting
samples. Original source encoding is not the model tensor encoding.

Proposed conservative missing-data policy: reject incomplete modalities or
misregistered rasters; ignore invalid pixels in the target and fill their input
values with a documented neutral value after normalization. Cloud policy, SAR
product choice, resampling and neutral fill are unresolved in ADR-0002. No silent
zero-filled missing modality is allowed. Real loading remains blocked on these
decisions. Statistics are fit only on training events and valid training pixels;
record training manifest hash, bands/units, algorithm, count and preprocessing
version. Validation/test use frozen training statistics.

## Manifest and isolation

`data.manifest.validate_manifest` accepts a JSON object with dataset_name,
source, revision, downloaded_at (ISO timestamp with zone) and nonempty records.
Each record has event_id, aoi_id, tile_id, parent_tile_id, split (train/val/test),
and files (modality -> {path, size_bytes, sha256}). Paths are portable relative
paths under the raw root; missing checksums may be null and must be reported as
unverified. Acquisition plans require checksums. At least sar_pre, sar_post,
optical, dem and mask are required. Chip identity adds window coordinates later.
The manifest describes immutable raw provenance. Derived artifacts separately
record raw manifest hash, preprocessing version, config hash and output hashes.

Keep official splits. Reject event, AoI or parent-tile reuse across splits under
the proposed event-domain protocol. If official metadata fails, report and seek
an ADR; do not relabel it. Sampling is split-local. Augmentation pairs only source
training events; chips from a parent tile never cross split boundaries.

## Models and fusion

`model(batch) -> {logits, features: {alignment}}`. Logits are finite float32
`[B,3,H,W]`, before softmax. Proposed adapter output for alignment is finite
float32 `[B,D,h,w]` with positive D,h,w. CNN and Transformer features need explicit
adapters (including token-to-grid restoration), and the same feature definition
within a comparison. Feature masks are nearest-neighbor mapped by an approved
adapter; ignore pixels remain ignored. Early fusion concatenates the proposed
17 channels in the order above; intermediate fusion exposes versioned modality
features. TerraMind's native interfaces and temporal support are not assumed.

## DG and training

`dg_regularizer(features, masks, event_ids) -> {loss, metrics}`. Loss is a finite
scalar float32; metrics maps names to finite scalar values. Masks align to the
feature grid and use the same class/ignore semantics. Pair distinct training
events only. Proposed covariance rule: fewer than two valid vectors per event
(per class for class-CORAL) skips that pair and records the skip count. Average
valid pairs equally; if all pairs skip, return a differentiable zero with an
explicit skipped flag. These are proposed algorithm requirements, not implemented
behavior. No covariance estimation or adaptation on validation/test data.
ClassMix/MixStyle are training-only transforms, not forced into this loss interface.
Trainer combines CE + Dice + configured DG weight; weights, Dice smoothing,
class inclusion and sampling schedule require approval. Ignore pixels contribute
to neither objective; all-ignore batches must be skipped explicitly and counted.

## Evaluation and selection

Proposed prediction is argmax over three logits. Exclude mask==255 from all
confusion counts. For class c, IoU=TP/(TP+FP+FN), F1=2TP/(2TP+FP+FN).
Flood uses class 2. Pixel/global metrics sum counts before division. Event metrics
first sum counts within each event, then divide. Event-macro Flood IoU averages
defined event Flood IoUs equally; worst-event is the minimum of those values,
with event ID and excluded-event count reported. mIoU averages defined IoUs over
classes 0,1,2. Zero denominators yield null, never perfect scores or zero; report
valid denominators and excluded classes/events. All-ignore evaluation yields null.
Evaluation chips must cover valid pixels once; any overlap requires an approved
reconstruction rule. Metrics are not implemented in this bootstrap.

Select checkpoints/hyperparameters on validation event-macro Flood IoU only;
record tie handling before experiments. Lock selection before one final test
evaluation. No test-guided tuning. ADR-0001 must approve these definitions.

## Run records

Execution environment is separate from research configuration. Persist resolved
runtime hash, device/precision, framework/CUDA versions and environment snapshot
alongside research provenance. Checkpoints at completed epochs contain model,
optimizer, scheduler, scaler, sampler, cursor and all RNG state. Resume requires
matching identity and compatible hardware; notebook logic cannot bypass this.
See [runtime contract](COLAB.md). Mid-epoch exact resume is not supported.

`tracking.experiments.validate_run` checks JSON metadata; `comparable` gates
ranking on architecture version, manifest hash, preprocessing version and
evaluation protocol, not merely a matching metric name. Each run records experiment
and run IDs, responsible developer, agent attribution, Git SHA, architecture
version, config and manifest SHA256, preprocessing version, model, fusion, DG,
seed, hyperparameters, evaluation protocol, metrics and status. Completed is not
verified: a human verification record with reviewer and evidence is required to
appear in the report. Different dataset/split/metric protocols must be reported
separately. Local records live under ignored `outputs/runs/` unless reviewed for
publication. MLflow is a later backend behind `log_run`; core checks need no server.
