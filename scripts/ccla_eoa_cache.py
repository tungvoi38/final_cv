"""Combine frozen single-seed FP32 Flip TTA cache with seed-43/44 inference."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.utils.data import DataLoader

from adapt_unet_event_bias import digest, flood_margin, write_json
from adapt_unet_local_context import margin_features
from ccla_crossfit_cache import (
    PACKED_BYTES, SPECS, PhysicalPatches, append, cache_path, collate, flipped_single,
    lookup, read_manifest, values,
)
from evaluate_unet_eoa import load_model, pre_variability_inputs, seed_everything
from evaluate_unet_eoa_flip_tta import FLIPS

SEEDS = (42,43,44)


def ensemble_mean(a,b,c):
    return ((a+b)+c)/np.float32(3)


def loader(archive, rows):
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise ValueError("Export requires one visible CUDA GPU")
    seed_everything(42)
    return DataLoader(PhysicalPatches(archive,rows,features_only=True),batch_size=2,num_workers=2,
                      shuffle=False,collate_fn=collate,pin_memory=True)


def models(members, seeds):
    if tuple(m["seed"] for m in members)!=tuple(seeds):
        raise ValueError("Unexpected ensemble member order")
    result = []
    for m in members:
        if m["sha256"]!=digest(Path(m["checkpoint"])):
            raise ValueError(f"Changed checkpoint for seed {m['seed']}")
        result.append(load_model(m,torch.device("cuda")))
    return result


def write_packet(staging,identity,rows,counts,files):
    write_json(staging/"manifest.json",identity | dict(
        seed=42,flip_axes=[list(a) for a in FLIPS],patches=rows,event_counts=counts,files=files,
        format="input-valid rows in original cache order; masks packed little-endian"))


def check_mask(meta, valid, old, masks):
    if any(meta[key]!=old[key] for key in ("event_id","patch_id","input_pixels","aoi_id","crs","affine")):
        raise ValueError(f"Cached patch metadata changed: {old['patch_id']}")
    expected = masks[old["mask_index"],0]
    if not np.array_equal(np.packbits(valid.reshape(-1),bitorder="little"),expected):
        raise ValueError(f"Seed-42 input-valid mask changed: {old['patch_id']}")


@torch.inference_mode()
def export_ensemble(archive,rows,path,pilot,members,identity):
    if path.exists():
        read_manifest(path,identity)
        print(f"Skip completed ensemble cache {path}",flush=True)
        return
    old_path,old = pilot
    if tuple(m["seed"] for m in members)!=SEEDS or members[0]["sha256"]!=old["checkpoint_sha256"]:
        raise ValueError("The cached seed-42 logits must come from the frozen member")
    if rows["tortilla:id"].astype(str).tolist()!=[p["patch_id"] for p in old["patches"]]:
        raise ValueError("Frozen patch traversal differs from seed 42")
    index, _ = lookup([pilot])
    cached = {e:values(index,e,"logits") for e in index}
    masks = {e:np.memmap(cache_path(old_path,e,"masks"),dtype="u1",mode="r").reshape(-1,2,PACKED_BYTES)
             for e in index}
    stream = loader(archive,rows)
    loaded = models(members[1:],SEEDS[1:])
    with TemporaryDirectory(prefix="eoa-cache-",dir=path.parent) as temp:
        staging = Path(temp)/"packet"
        staging.mkdir()
        files = {}
        for e in index:
            for field in ("labels","sar","extra","masks"):
                if field!="masks":
                    values(index,e,field)
                source = cache_path(old_path,e,field)
                cache_path(staging,e,field).hardlink_to(source)
                files[source.name] = old["files"][source.name]
        patch_rows = []
        for step,(batch,items) in enumerate(stream,1):
            x,_ = pre_variability_inputs({k:t.cuda(non_blocking=True) for k,t in batch.items()})
            with torch.autocast("cuda"):
                z43,z44 = [flipped_single(m,x).permute(0,2,3,1).cpu().numpy() for m in loaded]
            for (meta,valid,*_),a,b in zip(items,z43,z44,strict=True):
                p = old["patches"][len(patch_rows)]
                e = p["event_id"]
                check_mask(meta,valid,p,masks[e])
                sl = slice(p["start"],p["stop"])
                z = ensemble_mean(cached[e][sl],a[valid],b[valid])
                if z.shape!=(p["input_pixels"],3) or not np.isfinite(z).all():
                    raise ValueError(f"Invalid combined logits: {e}/{p['patch_id']}")
                append(cache_path(staging,e,"logits"),z,"<f4")
                patch_rows.append(p | margin_features(flood_margin(z)))
            if step%100==0 or step==len(stream):
                print(f"{identity['role']}: {step}/{len(stream)} batches",flush=True)
        if len(patch_rows)!=len(old["patches"]):
            raise ValueError("Incomplete combined cache")
        for e,n in old["event_counts"].items():
            file = cache_path(staging,e,"logits")
            if file.stat().st_size!=n*12:
                raise ValueError(f"Incomplete event logits: {e}")
            files[file.name] = digest(file)
        write_packet(staging,identity,patch_rows,old["event_counts"],files)
        staging.rename(path)


@torch.inference_mode()
def export_paired_diagnostic(archive,rows,path,members,identity):
    if path.exists():
        data = [read_manifest(path/name,identity | dict(member_seeds=seeds))
                for name,seeds in (("pilot",[42]),("ensemble",list(SEEDS)))]
        return data
    stream = loader(archive,rows)
    loaded = models(members,SEEDS)
    paired = ([],[])
    counts = {}
    for step,(batch,items) in enumerate(stream,1):
        x,_ = pre_variability_inputs({k:t.cuda(non_blocking=True) for k,t in batch.items()})
        with torch.autocast("cuda"):
            logits = [flipped_single(m,x).permute(0,2,3,1).cpu().numpy() for m in loaded]
        for (meta,valid,*_),a,b,c in zip(items,*logits,strict=True):
            e = meta["event_id"]
            start = counts.get(e,0)
            stop = start+int(valid.sum())
            counts[e] = stop
            for output,z in zip(paired,(a[valid],ensemble_mean(a[valid],b[valid],c[valid])),strict=True):
                if not np.isfinite(z).all():
                    raise ValueError("Nonfinite diagnostic logits")
                output.append(meta | dict(start=start,stop=stop) | margin_features(flood_margin(z)))
        if step%200==0 or step==len(stream):
            print(f"Full-train paired diagnostic: {step}/{len(stream)} batches",flush=True)
    with TemporaryDirectory(prefix="diagnostic-",dir=path.parent) as temp:
        staging = Path(temp)/"packet"
        staging.mkdir()
        for name,seeds,items in zip(("pilot","ensemble"),([42],list(SEEDS)),paired,strict=True):
            dest = staging/name
            dest.mkdir()
            if [p["patch_id"] for p in items]!=rows["tortilla:id"].astype(str).tolist():
                raise ValueError("Incomplete paired diagnostic patch order")
            write_packet(dest,identity | dict(member_seeds=seeds),items,counts,{})
        staging.rename(path)
    return [read_manifest(path/name,identity | dict(member_seeds=seeds))
                                 for name,seeds in (("pilot",[42]),("ensemble",list(SEEDS)))]
