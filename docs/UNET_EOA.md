# A3 Ensemble of Averages on Kuro Siwo

## Fixed 11-channel baseline with Flip TTA and calibration

The [baseline configuration](../configs/experiments/unet_a3_eoa_flip_tta.json)
pins the original 20-epoch, event-sqrt SMA members for seeds 42, 43 and 44
(epochs 14, 18 and 20, selected by validation pooled mIoU). The eleven inputs
include the original SAR and DEM channels, two SAR delta channels and two
pre-event variability channels. For each seed, the evaluator averages aligned
logits from the original image and three flipped views (horizontal, vertical
and both). It then averages the three SMA models and applies the global
flood-logit bias +0.95 chosen by pooled Flood IoU on validation.
The run records report validation pooled mIoU 68.604% and Flood F1 68.692%.
The corresponding test values are 74.510% and 79.005%.

```bash
cd /home/phun/tung/test/final_cv
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv
gpu=2
gpu_uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
mkdir -p ../artifacts
out=$(mktemp -d "$PWD/../artifacts/a3-eoa-flip-tta-XXXXXX")

CUDA_VISIBLE_DEVICES="$gpu_uuid" .venv/bin/python -u scripts/evaluate_unet_eoa.py \
  calibrated --split validation --out "$out/validation"
```

To reproduce the saved test evaluation explicitly, run the same fixed
configuration with `calibrated --split test --out "$out/test"`. Each split
writes pooled metrics and per-event counts. The evaluator verifies SHA-256 for
the selection and its three checkpoints before inference.
Use `calibrated --no-tta` to reproduce the previous calibrated baseline.

For flood PR and bias diagnostics, use `scripts/evaluate_unet_eoa_diagnostics.py`
with `--validation-dir` pointing to the saved Flip TTA validation comparison
and `--out` pointing to a fresh directory. The script checks the complete
validation bias sweep before a single test forward pass. Use
`test_paper_metrics.csv` for the validation-selected global bias. The
`DIAGNOSTIC_ONLY` tables use test labels to find separate biases in each event
and serve as upper bounds over the fixed grid. The JSON report gives the
population standard deviation of event IoU at zero, at the validation bias
and at the diagnostic oracle. The PR figures and exact curve arrays use
the flood logit minus the maximum nonflood logit.

To retrain the 11-channel architecture, use the training loop below with
`--epochs 20 --sar-delta --sar-pre-variability` for all three seeds. The
calibrated configuration is specific to the archived checkpoints listed
above. New checkpoints require their own validation selection and calibration.

## Historical uncalibrated comparison

The [frozen configuration](../configs/experiments/unet_a3_eoa.json) uses the
original A3 1×1 event-sqrt setup. All three seeds use 4,000 train patches, 15
epochs, batch size 2, AdamW, CE+Dice and the original learning rates. The
reference ERM run is recorded in the configuration.

## Train seeds 42, 43 and 44

Run on the server, inside `tmux` if you need the job to run after disconnecting.
The output directory and every run ID are new.

```bash
bash <<'BASH'
set -euo pipefail
cd /home/phun/tung/test/final_cv

gpu=2
nvidia-smi -i "$gpu"
gpu_uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
export CUDA_VISIBLE_DEVICES="$gpu_uuid"
export HF_HOME=/DATA/phun/kuro_siwo/.hf HF_HUB_OFFLINE=1

raw=/DATA/phun/kuro_siwo/raw/geobench_kuro_siwo.tortilla
suffix=$(.venv/bin/python -c 'from uuid import uuid4; print(uuid4().hex[:8])')
out="/DATA/phun/kuro_siwo/eoa/a3-eoa-$(date +%Y%m%d-%H%M%S)-$suffix"
test -r "$raw"
mkdir -p /DATA/phun/kuro_siwo/eoa
mkdir "$out"
cp configs/experiments/unet_a3_eoa.json "$out/protocol.json"
.venv/bin/python -m pip freeze > "$out/requirements.txt"
printf 'EoA output: %s\n' "$out"

for seed in 42 43 44; do
  pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)
  if [ -n "$pids" ]; then
    echo "GPU $gpu đang bận. Dừng trước seed $seed."
    exit 1
  fi
  run_id="a3-eoa-seed$seed"
  log="$out/$run_id.log"
  printf 'Theo dõi: tail -f %q\n' "$log"

  .venv/bin/python -u scripts/train_unet.py \
    --repo . --raw "$raw" --out "$out" --run-id "$run_id" \
    --epochs 15 --batch-size 2 --seed "$seed" --workers 2 \
    --optimizer adamw --loss ce_dice \
    --encoder-lr 3e-5 --decoder-lr 1e-4 --weight-decay 0.01 \
    --head a3 --sampling event_sqrt --averaging sma --selection pooled_miou \
    2>&1 | tee "$log"
done
printf 'Set this in the next terminal: out=%q\n' "$out"
BASH
```

If interrupted after a completed epoch, rerun the same trainer command with
`--resume` and the same run ID, output path, arguments and source files. A
partial epoch is restarted from the latest complete snapshot. The optimizer,
scheduler, GradScaler, Python/NumPy/PyTorch RNG and loader/sampler generators
are restored.

## Freeze selection and compare on validation

Set `out` to the exact path printed above. Select a GPU that has no compute
processes. Training results for all three seeds must be complete.

```bash
cd /home/phun/tung/test/final_cv
out=/DATA/phun/kuro_siwo/eoa/REPLACE_WITH_PRINTED_DIRECTORY
gpu=2
nvidia-smi -i "$gpu"
gpu_uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)

CUDA_VISIBLE_DEVICES="$gpu_uuid" \
.venv/bin/python -u scripts/evaluate_unet_eoa.py validation \
  --protocol "$out/protocol.json" \
  --runs "$out/a3-eoa-seed42" "$out/a3-eoa-seed43" "$out/a3-eoa-seed44" \
  --out "$out/comparison"
```

The evaluator verifies the frozen training recipe, split manifest and
validation rankings, then writes `selection.json`. It saves scores for the
original ERM model, each matched online and SMA model, the online ensemble and
EoA. `validation_comparison.csv` includes the mean of the single-seed results
as well as the ensemble results. Each ensemble receives the same batch in
each member and averages its aligned logits in float32 before argmax.

## Evaluate the frozen selection on test

After reading the validation results, explicitly run the test command.
Reuse the same `out` and `gpu_uuid` variables.

```bash
CUDA_VISIBLE_DEVICES="$gpu_uuid" \
.venv/bin/python -u scripts/evaluate_unet_eoa.py test --out "$out/comparison"
```

The test command requires completed validation output, checks the selection
and checkpoint digests and writes to a fresh `comparison/test` directory.
Its `test_metrics.csv`, `test_metrics.json` and `test_comparison.csv` report
pooled mIoU, Flood IoU/F1, event-macro Flood IoU, Flood Precision and Recall.
These comparisons use the same mask/argmax protocol and zero logit bias. The
legacy evaluator continues to provide bias or epoch diagnostics separately.

## Checkpoint semantics

Online weights are trained by the original ERM update. In the official
recurrence, step 99 is the last direct copy. The update at successful
optimizer step 100 averages states 99 and 100 with count 1. Every following
successful step contributes a new state to the arithmetic mean. The
implementation uses in-place interpolation with the equivalent coefficient.
Floating BatchNorm running buffers are averaged and the integer batch count
is retained from the initial copy. Inference uses eval mode.
[Salesforce MovingAvg](https://github.com/salesforce/ensemble-of-averages/blob/main/domainbed/algorithms.py#L109-L160)

Each completed epoch stores both weight sets, the averaging step/count and
all continuation state. In SMA runs, `model` is the SMA weight set and
`online_model` is the online weight set. `best.pt` and `best_online.pt`
point to the validation-selected snapshots for their respective weight sets.
The original ERM runs and checkpoints keep their existing semantics.
