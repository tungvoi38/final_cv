# Data management and provenance

Use `data/raw/geoid-flood/` for immutable source files and `data/processed/` for
derived products. Both are ignored; scripts create directories only when needed.
Track small reviewed manifests/configs, never rasters, archives or checkpoints.
No GEOID-Flood files were downloaded for bootstrap.

## Verified distribution, unresolved revision

Sources inspected 2026-11-02: [official code repository](https://github.com/links-ads/geoid-flood)
and [official dataset card](https://huggingface.co/datasets/links-ads/geoid-flood).
The card describes a revised release, 12 optical bands, pre/post VV/VH SAR, DEM,
labels 0/1/2 with 255 ignored, and AoI split assignments. Its sample places two
AoIs of the same activation in train and test: event isolation requires an audit.
Older indexed descriptions differ. No immutable revision is selected here.

The card describes SAR as linear backscatter, optical reflectance scaling,
no-data/cloud masks, and training statistics. Inspect actual raster tags and
revision metadata before adopting any conversion, band order or statistics.
Choose GRD versus RTC explicitly. Do not mix releases.

The card labels compilation/splits/labels CC BY 4.0 with attribution and Copernicus
notices; DEM has a separate WorldDEM-30 license. Review those linked terms before
redistribution. Authentication requirements have not been tested.

## Safe interface

Default command is an offline dry run and currently exits with a clear blocker:

```powershell
uv run --locked --no-editable python scripts/download_data.py
uv run --locked --no-editable python scripts/download_data.py --mode sample --dry-run
uv run --locked --no-editable python scripts/download_data.py --verify-only
uv run --locked --no-editable python scripts/inspect_data.py data/manifests/reviewed.json --raw-root data/raw/geoid-flood
```

Before acquisition, a reviewed config must pin the same 40-character dataset
commit in `revision` and `acquisition.revision`. Populate `acquisition.files` with
explicit path, official revision-pinned HTTPS URL, size_bytes, verified sha256 and
sample boolean. `sample_max_bytes` caps sample selection (proposed 100 MiB).
These are acquisition records; they are distinct from the per-tile raw manifest.
Do not fill URLs/checksums from guesswork. Empty or floating sources fail closed.

Later authorized download: add `--execute`; full selection additionally requires
`--mode full --confirm-full`. `--overwrite` is explicit. Disk space, size and
SHA256 are checked. Downloads use temporary files and never extract archives.
The official downloader documents resume support; this minimal client restarts
partial files and refuses existing files unless overwrite is explicit. For large
acquisition, review a pinned official downloader and its extraction policy first.

## Manifests and derived artifacts

See [manifest contract](../docs/CONTRACTS.md). Store dataset/source/revision/time,
event/AoI/tile/parent IDs, split and modality file sizes/hashes. Never rewrite raw
provenance to describe a transformed file. Derived provenance references the raw
manifest hash, preprocessing version, config hash and output hashes separately.
Unknown checksums are null and are reported unverified, never silently trusted.

Official splits remain immutable. Reject cross-split event/AoI/parent reuse under
the proposed unseen-event protocol and escalate conflicts through ADR-0001.
No-data handling, cloud treatment and normalization await ADR-0002. Reject missing
modalities/misalignment; input fillers and ignored targets need an approved policy.
