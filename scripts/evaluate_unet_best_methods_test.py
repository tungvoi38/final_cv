#!/usr/bin/env python3
"""Historical test benchmark with label-free predictions locked before oracle scoring."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import multiprocessing as mp
from itertools import combinations
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from rasterio.warp import transform
from scipy.special import expit
import torch
from torch.utils.data import DataLoader

from adapt_unet_event_bias import IGNORE, digest, flood_margin, load_json, oracle_from_ranks, scores, write_json
from adapt_unet_expected_iou import sar_features
from adapt_unet_graph_cut import graph_cut, patch_inputs, tail_bootstrap
from adapt_unet_graph_cut import WORKER, FIELDS, maxflow_module
from adapt_unet_iw_iou import OVERLAP, draw, feature_rows, fit_domain, log_ratio, weighted_source_curve
from adapt_unet_local_context import margin_features
from audit_ccla_scalar_bias import F1, lock_oof, stats
from audit_unet_bias_failures import checked, read_csv, save_csv
from bootstrap_unet_events import ITERATIONS, SEED, interval
from ccla_crossfit_analysis import (
    EVENT, GLOBAL, GLOBAL_BIAS, GRID, IW_METHODS, IW_MARGIN, IW_SAR, MARGIN, NOBIAS,
    PATCH, REF, apply_predictor, arrays, flat_patches,
)
from ccla_crossfit_cache import PACKED_BYTES, SPECS, append, cache_path, collate, flipped_single, lookup, values
from ccla_crossfit_protocol import REPO, select_rows, split_rows
from ccla_eoa_cache import SEEDS, ensemble_mean, models
from diagnose_sar_scattering import slope_degrees
from diagnose_unet_spatial_oracle import apply_groups, average_precision, group, predicted_confusion, shifted_rank
from evaluate_unet_eoa import pre_variability_inputs, seed_everything
from evaluate_unet_eoa_diagnostics import rank_event, rank_pooled
from evaluate_unet_eoa_flip_tta import FLIPS
from kuro_dataset import RawArchive, normalize_physical_inputs, preprocess, read_metadata, read_sample
from run_ccla_crossfit_eoa import full_member
from unet_conservative_expected_iou import best_index

SCOPE = "historical-test exploratory evaluation"
GRAPH = ("graph_uniform_ccla","graph_sar_guided_ccla")
ROBUST = "robust_ccla_margin"
CONTINUOUS = (NOBIAS,GLOBAL,F1,REF,IW_MARGIN,IW_SAR,MARGIN,ROBUST)
LOCKED = (*CONTINUOUS,*GRAPH)
METHODS = (*LOCKED,EVENT,PATCH)
RAW_FIELDS = ("logits","sar","extra")
PAPER = "https://proceedings.neurips.cc/paper_files/paper/2024/file/43612b0662cb6a4986edf859fd6ebafe-Paper-Datasets_and_Benchmarks_Track.pdf"
REPO_FILES = ("evaluate_unet_best_methods_test.py","kuro_dataset.py","evaluate_unet_eoa.py",
              "evaluate_unet_eoa_flip_tta.py","evaluate_unet_eoa_diagnostics.py","ccla_crossfit_analysis.py",
              "ccla_crossfit_cache.py","ccla_eoa_cache.py","adapt_unet_local_context.py",
              "adapt_unet_graph_cut.py","adapt_unet_iw_iou.py","adapt_unet_event_bias.py",
              "audit_ccla_scalar_bias.py","diagnose_unet_spatial_oracle.py","diagnose_sar_scattering.py",
              "train_unet.py","bootstrap_unet_events.py","run_ccla_crossfit_eoa.py","ccla_crossfit_protocol.py")
PRE_MARGIN_ONLY_FIX = "0d4f5d48e956dbcb8fed263b03f5f23998f8b081ef135bfb8b82bc0a061b780d"


def code_identity():
    return {f:digest(REPO/"scripts"/f) for f in REPO_FILES}


def compatible_code(saved, current):
    script = Path(__file__).name
    if saved.get(script)==PRE_MARGIN_ONLY_FIX:
        saved = saved | {script:current[script]}
    return saved==current


def footprint(frame):
    affine = np.stack(frame["stac:geotransform"])
    if (not frame["stac:crs"].eq("EPSG:3857").all()
            or not np.allclose(affine[:,[1,2,4,5]],[10,0,0,-10],rtol=0,atol=1e-6)
            or any(tuple(a)!=(224,224) for a in frame["stac:raster_shape"])):
        raise ValueError("Unexpected official grid or patch shape")
    role = frame["tortilla:data_split"].to_numpy()
    e = frame.actid.astype(str).to_numpy()
    ids = frame["tortilla:id"].astype(str).to_numpy()
    if len(set(ids))!=len(ids):
        raise ValueError("Repeated official sample identifier")
    x = affine[:,[0,3]]
    test = np.flatnonzero(role=="test")
    pairs = []
    for i in test:
        diff = np.abs(x-x[i])
        near = np.flatnonzero((diff<2240-1e-6).all(axis=1))
        for j in near:
            if role[j]=="test" and i>=j:
                continue
            pairs.append(dict(test_patch=ids[i],test_event=e[i],other_patch=ids[j],
                              other_event=e[j],other_split=role[j],
                              intersection_m2=float(np.prod(2240-diff[j]))))
    result = {r:sum(p["other_split"]==r for p in pairs) for r in ("train","validation","test")}
    return pairs,result


def sources(run, robust, graph):
    config = load_json(run/"frozen_config.json")
    cp = load_json(run/"fit/predictor.json")
    r_cfg = load_json(robust/"analysis_config.json")
    r_lock = load_json(robust/"predictor_lock.json")
    g_cfg = load_json(graph/"frozen_config.json")
    g_lock = load_json(graph/"graph_cut_lock.json")
    config_hash = digest(run/"frozen_config.json")
    if (cp["frozen_config_sha256"]!=config_hash or cp["run_id"]!=config["run_id"]
            or config["seeds"]!=list(SEEDS) or config["flip_axes"]!=[list(x) for x in FLIPS]
            or tuple(m["seed"] for m in config["full_train"])!=SEEDS
            or any(m["epoch"]!=20 or m["weights_key"]!="model" or m["input_channels"]!=11 for m in config["full_train"])):
        raise ValueError("CCLA and epoch-20 ensemble provenance mismatch")
    if (Path(r_cfg["run"]).resolve()!=run or r_cfg["split"]!="validation"
            or r_cfg["validation_manifest_sha256"]!=config["validation_manifest_sha256"]
            or digest(robust/"predictor_lock.json")!=r_cfg["predictor_lock_sha256"]
            or digest(robust/"gamma_lock.json")!=r_lock["gamma_lock_sha256"]
            or load_json(robust/"gamma_lock.json")["gamma"]!=r_lock["gamma"]):
        raise ValueError("Robust CCLA is not the locked OOF predictor for this run")
    checked(r_lock["gamma_zero"]["coefficients"],cp["models"][MARGIN]["coefficients"],"Robust CCLA gamma-zero ancestry",1e-8)
    if (Path(g_cfg["source"]).resolve()!=run or g_cfg["source_run_sha256"]!=config_hash
            or digest(graph/"meta_cv_summary.csv")!=g_lock["metrics_sha256"]
            or digest(graph/"meta_parameters.json")!=g_lock["meta_parameters_sha256"]):
        raise ValueError("Graph Cut settings are not from the locked OOF study")
    choices = [g_lock["selected"][m] for m in GRAPH]
    if any(c["key"]!=m or c["init"]!=MARGIN for c,m in zip(choices,GRAPH,strict=True)):
        raise ValueError("Graph Cut configuration must use CCLA initialization")
    return config,cp,r_lock,choices,load_json(graph/"meta_parameters.json")["all_oof_sar_scale"]


def source_paths(run,robust,graph):
    return [run/"frozen_config.json",run/"fit/predictor.json",run/"fit/oof_patch_curves.npz",
            robust/"analysis_config.json",robust/"predictor_lock.json",robust/"gamma_lock.json",
            graph/"frozen_config.json",graph/"graph_cut_lock.json",graph/"meta_parameters.json",graph/"meta_cv_summary.csv"]


def plan(args):
    out,run,robust,graph = args.out,args.run,args.robust_results,args.graph_results
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    config,_,r,choices,scale = sources(run,robust,graph)
    if [full_member(m) for m in config["full_train"]]!=config["full_train"]:
        raise ValueError("Full-train epoch-20 checkpoints differ from the frozen selection")
    frame = read_metadata(Path(config["raw"]))
    test = split_rows(frame,"test")
    events = sorted(set(test.actid.astype(str)),key=int)
    history = args.historical/"test_flood_pr_summary.csv"
    protocol_path = args.historical/"protocol.json"
    prior = load_json(protocol_path)
    prior_events = {row["scope"] for row in read_csv(history) if row["scope"]!="pooled"}
    other_events = set(frame.loc[frame["tortilla:data_split"]!="test","actid"].astype(str))
    if (len(test)!=2000 or len(events)!=10 or prior_events!=set(events) or set(events)&other_events):
        raise ValueError("Official test patches or historical event IDs differ")
    pairs, counts = footprint(frame)
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/"test_manifest.json",dict(split="test",events=events,patch_ids=test["tortilla:id"].astype(str).tolist(),
        event_patch_counts={e:int(test.actid.astype(str).eq(e).sum()) for e in events},
        patches=2000,coverage="all official test patches, including identified within-test footprint overlap"))
    write_json(out/"footprint_audit.json",dict(projection="EPSG:3857",overlap_pair_counts=counts,pairs=pairs,
        cross_role_policy="stop if any test footprint intersects train or validation",
        same_role_policy="retain official patch observations, no mosaic deduplication",
        historical_events_verified=True,historical_exact_patch_ids="unverified; historical summary is event-level",
        historical_members_match=all(a["sha256"]==b["sha256"] for a,b in zip(prior["checkpoint_members"],config["full_train"],strict=True)),
        parent_scene_independence="unverified"))
    if counts["train"] or counts["validation"]:
        raise ValueError(f"Test footprint intersects development data. See {out/'footprint_audit.json'}")
    # Freeze the audited artifact identities before any test imagery or annotation is opened.
    write_json(out/"frozen_config.json",dict(scope=SCOPE,source=str(run),robust_results=str(robust),graph_results=str(graph),
        raw=config["raw"],members=config["full_train"],flip_axes=[list(x) for x in FLIPS],
        oof_scope="4000 training patches across 27 events",methods=list(METHODS),robust_gamma=r["gamma"],
        graph_cut=choices,sar_scale=scale,manifest_sha256=digest(out/"test_manifest.json"),
        footprint_audit_sha256=digest(out/"footprint_audit.json"),
        source_sha256={str(f):digest(f) for f in source_paths(run,robust,graph)},
        historical_sha256={str(f):digest(f) for f in (history,protocol_path)},
        code_sha256=code_identity()))
    lock_oof(run,out/"scalar_inputs")
    print(f"Frozen {SCOPE}: 2,000 patches from 10 events; footprint pairs={counts}",flush=True)


def test_rows(frame,out,config):
    p = out/"test_manifest.json"
    if digest(p)!=config["manifest_sha256"]:
        raise ValueError("Changed test scope")
    ids = load_json(p)["patch_ids"]
    return select_rows(split_rows(frame,"test"),ids)


class TestPatches:
    """Physical model inputs only. Annotation TIFFs are skipped explicitly."""

    def __init__(self,archive,rows):
        self.archive,self.rows = archive,rows
        affine = np.stack(rows["stac:geotransform"])[:,[0,3]]+[1120,-1120]
        _,lat = transform("EPSG:3857","EPSG:4326",*affine.T)
        self.ground = 10*np.cos(np.radians(lat))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,i):
        self.archive.check_unchanged()
        row = self.rows.iloc[i]
        sample = preprocess(*read_sample(self.archive.path,row,include_labels=False),include_labels=False)
        valid = sample["input_valid"]
        sar,dem = normalize_physical_inputs(sample["sar_db"],sample["dem_m"],valid)
        slope = slope_degrees(sample["dem_m"][0],valid,self.ground[i])[valid]
        cov = sar_features(sample["sar_db"],valid)
        extra = np.column_stack((sample["sar_db"][2,1][valid],slope))
        meta = dict(patch_id=str(row["tortilla:id"]),event_id=str(row.actid),input_pixels=int(valid.sum()),
                    crs=str(row["stac:crs"]),affine=[float(v) for v in row["stac:geotransform"]],aoi_id=str(row.aoiid))
        return torch.from_numpy(sar),torch.from_numpy(dem),meta,valid,None,cov,extra


@torch.inference_mode()
def cache_test(out,config):
    path = out/"test_unlabeled"
    if path.exists():
        read_test_cache(out,config)
        print("Reuse locked unlabeled test logits",flush=True)
        return
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise ValueError("Cache export requires exactly one CUDA GPU")
    archive = RawArchive(config["raw"])
    rows = test_rows(archive.frame,out,config)
    seed_everything(42)
    loader = DataLoader(TestPatches(archive,rows),batch_size=2,num_workers=2,
                        shuffle=False,pin_memory=True,collate_fn=collate)
    ensemble = models(config["members"],SEEDS)
    offsets,mask_index,patches = {},{},[]
    with TemporaryDirectory(prefix="unlabeled-",dir=out) as temp:
        staging = Path(temp)/"packet"
        staging.mkdir()
        for step,(batch,items) in enumerate(loader,1):
            x,_ = pre_variability_inputs({k:v.cuda(non_blocking=True) for k,v in batch.items()})
            with torch.autocast("cuda"):
                a,b,c = [flipped_single(m,x).permute(0,2,3,1).cpu().numpy() for m in ensemble]
            for (m,valid,annotation,sar,extra),z0,z1,z2 in zip(items,a,b,c,strict=True):
                if annotation is not None:
                    raise ValueError("Unexpected test annotation before the lock")
                e = m["event_id"]
                z = ensemble_mean(z0[valid],z1[valid],z2[valid])
                if not np.isfinite(z).all():
                    raise ValueError(f"Invalid logits for {m['patch_id']}")
                start = offsets.get(e,0)
                stop = start+len(z)
                number = mask_index.get(e,0)
                patches.append(m|dict(start=start,stop=stop,mask_index=number)|margin_features(flood_margin(z)))
                offsets[e],mask_index[e] = stop,number+1
                for field,data in (("logits",z),("sar",sar),("extra",extra)):
                    append(cache_path(staging,e,field),data,SPECS[field][0])
                append(cache_path(staging,e,"masks"),np.packbits(valid.ravel(),bitorder="little"),"u1")
            if step%100==0 or step==len(loader):
                print(f"Unlabeled test cache: {step}/{len(loader)} batches",flush=True)
        if [p["patch_id"] for p in patches]!=rows["tortilla:id"].astype(str).tolist():
            raise ValueError("Inference patch order differs from frozen scope")
        write_json(staging/"manifest.json",dict(split="test",labels_present=False,
            manifest_sha256=config["manifest_sha256"],members=config["members"],flip_axes=config["flip_axes"],
            event_counts=offsets,patches=patches,files={f.name:digest(f) for f in staging.glob("*.bin")}))
        staging.rename(path)
    read_test_cache(out,config)


def read_test_cache(out,cfg):
    path = out/"test_unlabeled"
    m = load_json(path/"manifest.json")
    scope = load_json(out/"test_manifest.json")
    if (m["split"]!="test" or m["labels_present"] is not False or m["manifest_sha256"]!=cfg["manifest_sha256"]
            or m["members"]!=cfg["members"] or m["flip_axes"]!=cfg["flip_axes"]
            or [p["patch_id"] for p in m["patches"]]!=scope["patch_ids"]
            or set(m["event_counts"])!=set(scope["events"])):
        raise ValueError("Unlabeled cache provenance or test membership mismatch")
    allowed = {cache_path(path,e,k).name for e in scope["events"] for k in (*RAW_FIELDS,"masks")}
    if allowed!=set(m["files"]) or allowed!={p.name for p in path.glob("*.bin")}:
        raise ValueError("Unexpected file in the unlabeled cache")
    offsets,counts = dict.fromkeys(scope["events"],0),dict.fromkeys(scope["events"],0)
    for p in m["patches"]:
        e = p["event_id"]
        if p["start"]!=offsets[e] or p["mask_index"]!=counts[e] or p["stop"]-p["start"]!=p["input_pixels"]:
            raise ValueError(f"Unaligned test patch {p['patch_id']}")
        offsets[e] = p["stop"]
        counts[e] += 1
    if counts!=scope["event_patch_counts"] or offsets!=m["event_counts"] or not all(offsets.values()):
        raise ValueError("Incomplete event membership or cache coverage")
    for e,n in counts.items():
        f = cache_path(path,e,"masks")
        if f.stat().st_size!=n*PACKED_BYTES or digest(f)!=m["files"][f.name]:
            raise ValueError(f"Input-valid mask changed for event {e}")
    return path,m


def check_oof(run,predictor):
    packs = []
    seen = set()
    for f,hash_ in predictor["oof_manifest_sha256"].items():
        folder = run/f"fold-{f}"
        path = folder/"oof"
        m = load_json(path/"manifest.json")
        split = load_json(folder/"split_manifest.json")
        if digest(path/"manifest.json")!=hash_ or m["split"]!="train" or m["member_seeds"]!=list(SEEDS):
            raise ValueError("Changed OOF training cache")
        ids = [p["patch_id"] for p in m["patches"]]
        events = set(map(int,m["event_counts"]))
        if (ids!=split["holdout_patch_ids"] or events!=set(split["holdout_events"])
                or events&set(split["source_events"]) or seen&set(ids)):
            raise ValueError("Duplicate or in-sample OOF patch")
        seen.update(ids)
        packs.append((path,m))
    index,_ = lookup(packs)
    if len(seen)!=4000 or len(index)!=27:
        raise ValueError("Incomplete partition of 4000 OOF patches")
    return packs


def init_worker(files):
    WORKER.clear()
    for e,(path,n) in files.items():
        root = Path(path)
        WORKER[e] = {k:np.memmap(cache_path(root,e,k),dtype=SPECS[k][0],mode="r",shape=(n,SPECS[k][1]))
                     for k in RAW_FIELDS}
        WORKER[e]["masks"] = np.memmap(cache_path(root,e,"masks"),dtype="u1",mode="r").reshape(-1,1,PACKED_BYTES)


def predict_event(task):
    path,e,patches,settings,graphs,scale = task
    if path.exists():
        return e
    sizes = np.array([(p["input_pixels"]+7)//8 for p in patches],dtype=np.int64)
    ranges = np.r_[0,sizes.cumsum()]
    bits = np.empty((len(LOCKED),int(ranges[-1])),dtype=np.uint8)
    for i,p in enumerate(patches):
        _,_,valid,z,margin,links = patch_inputs(p,scale,graphs)
        original = margin[valid]
        p_id = p["patch_id"]
        scalar,local = settings["scalar_biases"],settings["patch_biases"]
        all_masks = [original+(scalar[m] if m in scalar else local[m][p_id])>0 for m in CONTINUOUS]
        for g in graphs:
            d = margin+local[MARGIN][p_id]
            all_masks.append(graph_cut(d,valid,g["strength"],links.get((g["family"],g["sigma"])))[valid])
        if len(z)!=p["input_pixels"]:
            raise ValueError("Test patch changed during prediction")
        for m,mask in enumerate(all_masks):
            bits[m,ranges[i]:ranges[i+1]] = np.packbits(mask,bitorder="little")
    tmp = path.with_suffix(".partial")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle,methods=LOCKED,patch_ids=[p["patch_id"] for p in patches],ranges=ranges,predictions=bits)
    os.replace(tmp,path)
    return e


def lock_test_biases(oof, target, predictor, robust_model, f1_bias):
    src_index, src_patches = lookup(oof)
    test_index, test_patches = lookup([target])
    source = sorted(src_index,key=int)
    rows = flat_patches(src_patches)
    sx,ys,ss,se = arrays(src_index,("logits","labels","sar","extra"))
    tx,ts,te = arrays(test_index,RAW_FIELDS)  # The target path has no label accessor.

    def source_features(event, indices):
        return feature_rows(sx,ss,se,event,indices)

    locks = {}
    for event in sorted(test_index,key=int):
        target_x = feature_rows(tx,ts,te,event,draw(len(tx[event]),event,target=event))
        source_sizes,source_x = {},[]
        for e in source:
            eligible = np.flatnonzero(ys[e]!=IGNORE)
            if not len(eligible):
                raise ValueError(f"No scored OOF source pixels in {e}")
            source_sizes[e] = len(eligible)
            source_x.append(source_features(e,draw(eligible,e,target=event)))
        scalar = {NOBIAS:0.,GLOBAL:GLOBAL_BIAS,REF:predictor["reference_bias"],F1:f1_bias}
        diagnostics = {}
        for method,width in IW_METHODS:
            model = fit_domain(source_x,target_x,width)
            p = expit(log_ratio(target_x,model))
            counts,diag = weighted_source_curve(source,ys,source_sizes,source_features,model,GRID)
            scalar[method] = float(GRID[best_index(counts[-1],GRID)])
            diagnostics[method] = diag | model | dict(
                target_overlap_fraction=float(np.mean((p>=OVERLAP[0])&(p<=OVERLAP[1]))))
            print(f"IW {method} event {event}: b={scalar[method]:+.2f}",flush=True)
        patches = test_patches[event]
        local = {method:{p["patch_id"]:float(b) for p,b in zip(patches,apply_predictor(rows,patches,model),strict=True)}
                 for method,model in ((MARGIN,predictor["models"][MARGIN]),(ROBUST,robust_model))}
        locks[event] = dict(sources=source,scalar_biases=scalar,patch_biases=local,iw_diagnostics=diagnostics)
        print(f"Locked bias for test event {event} from unlabeled inputs",flush=True)
    return locks


def predict(out,cfg,workers):
    lock_file = out/"test_predictions_lock.json"
    if lock_file.exists():
        prediction_lock(out,cfg)
        print("Reuse frozen non-oracle test predictions",flush=True)
        return
    cache,m = read_test_cache(out,cfg)
    run,robust = map(Path,(cfg["source"],cfg["robust_results"]))
    predictor = load_json(run/"fit/predictor.json")
    inputs = check_oof(run,predictor)
    index,patches = lookup([(cache,m)])
    # Open the known label-free target fields and verify payload digests before processing.
    for e in index:
        for k in RAW_FIELDS:
            values(index,e,k)
    scalar_lock = load_json(out/"scalar_inputs/scalar_bias_lock.json")
    bias = scalar_lock["chosen"][F1]["bias"]
    checked([bias,predictor["reference_bias"]],[1.50,3.70],"OOF fixed scalars")
    b_path = out/"test_bias_lock.json"
    provenance = dict(input_cache_sha256=digest(cache/"manifest.json"),
                      frozen_config_sha256=digest(out/"frozen_config.json"))
    if b_path.exists():
        prior = load_json(b_path)
        if any(prior[k]!=v for k,v in provenance.items()):
            raise ValueError("Saved test biases do not match current cache")
        locks = prior["events"]
    else:
        r_model = load_json(robust/"predictor_lock.json")["robust"]
        locks = lock_test_biases(inputs,(cache,m),predictor,r_model,bias)
        write_json(b_path,provenance|dict(test_labels_used=False,scalar_source="frozen training OOF only",
            iw_source="27 OOF train events with labels, per-event unlabeled test covariates",events=locks))
    pred = out/"predictions"
    pred.mkdir(exist_ok=True)
    tasks = [(pred/f"{e}.npz",e,ps,locks[e],cfg["graph_cut"],cfg["sar_scale"]) for e,ps in patches.items()]
    files = {e:(str(d[0]),d[2]) for e,d in index.items()}
    pending = [task for task in tasks if not task[0].exists()]
    if workers==1:
        init_worker(files)
        for task in pending:
            print(f"Predicted event {predict_event(task)}",flush=True)
    elif pending:
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context("fork"),
                                 initializer=init_worker,initargs=(files,)) as pool:
            for e in pool.map(predict_event,pending):
                print(f"Predicted event {e}",flush=True)
    for path,e,ps,*_ in tasks:
        read_predictions(path,ps)
    write_json(lock_file,provenance | dict(bias_lock_sha256=digest(b_path),methods=list(LOCKED),
        test_labels_used=False,prediction_sha256={e:digest(path) for path,e,*_ in tasks}))
    print(f"Frozen predictions for {len(tasks)} historical test events",flush=True)


def read_predictions(path,patches):
    with np.load(path,allow_pickle=False) as a:
        ids,methods,span,bits = a["patch_ids"].tolist(),a["methods"].tolist(),a["ranges"],a["predictions"]
    expected = np.r_[0,np.cumsum([(p["input_pixels"]+7)//8 for p in patches])]
    if (ids!=[p["patch_id"] for p in patches] or methods!=list(LOCKED)
            or not np.array_equal(span,expected) or bits.shape!=(len(LOCKED),int(expected[-1]))
            or bits.dtype!=np.uint8):
        raise ValueError(f"Malformed locked prediction packet {path}")
    return span,bits


def prediction_lock(out,cfg):
    val = load_json(out/"test_predictions_lock.json")
    path,m = read_test_cache(out,cfg)
    if (val["frozen_config_sha256"]!=digest(out/"frozen_config.json")
            or val["input_cache_sha256"]!=digest(path/"manifest.json")
            or val["bias_lock_sha256"]!=digest(out/"test_bias_lock.json")
            or val["test_labels_used"] is not False or val["methods"]!=list(LOCKED)
            or set(val["prediction_sha256"])!=set(m["event_counts"])):
        raise ValueError("Non-oracle predictions are missing or were not locked correctly")
    for e,h in val["prediction_sha256"].items():
        if digest(out/f"predictions/{e}.npz")!=h:
            raise ValueError(f"Locked test predictions changed for event {e}")
    return path,m


def labels_for_event(archive,rows,patches,mask_file,n):
    packed = np.memmap(mask_file,dtype="u1",mode="r").reshape(-1,PACKED_BYTES)
    y = np.empty(n,dtype=np.uint8)
    for (_,row),p in zip(rows.iterrows(),patches,strict=True):
        archive.check_unchanged()
        physical = preprocess(*read_sample(archive.path,row,include_labels=True),include_labels=True)
        valid = physical["input_valid"]
        if (str(row.actid)!=p["event_id"] or str(row["tortilla:id"])!=p["patch_id"]
                or not np.array_equal(np.packbits(valid.ravel(),bitorder="little"),packed[p["mask_index"]])):
            raise ValueError(f"Ground truth and locked input geometry differ at {p['patch_id']}")
        labels = physical["mask"][valid]
        y[p["start"]:p["stop"]] = labels
    return y


def score_event(e,z,y,patches,predictions,settings):
    keep = y!=IGNORE
    y_score = y[keep]
    if not len(y_score) or not np.isin(y_score,[0,1,2]).all():
        raise ValueError(f"Unexpected scored classes for {e}")
    logits = z[keep]
    margin = flood_margin(logits)
    nonflood = logits[:,:2].argmax(1)
    del logits
    flood = y_score==2
    raw = rank_event([(margin,flood)])
    event_bias,event_iou = oracle_from_ranks(raw) if flood.any() else (0.,0.)
    scalar,local = settings["scalar_biases"],settings["patch_biases"]
    rank = {m:rank_pooled([shifted_rank(raw,b)]) for m,b in (scalar|{EVENT:event_bias}).items() if m in METHODS}
    cms = {m:np.zeros((3,3),dtype=np.int64) for m in LOCKED}
    starts,bits = predictions
    offset = 0
    groups,shifted = [],{m:[] for m in (MARGIN,ROBUST)}
    for i,p in enumerate(patches):
        sl = slice(p["start"],p["stop"])
        subset = keep[sl]
        n = int(subset.sum())
        view = slice(offset,offset+n)
        group_ = group(p["patch_id"],view,margin,flood,p["input_pixels"])
        groups.append(group_)
        for m in shifted:
            if group_.rank is not None:
                shifted[m].append(shifted_rank(group_.rank,local[m][p["patch_id"]]))
        target = y_score[view].astype(np.int64)*3
        for j,m in enumerate(LOCKED):
            pred = np.unpackbits(bits[j,starts[i]:starts[i+1]],count=p["input_pixels"],bitorder="little").astype(bool)[subset]
            if m in CONTINUOUS:
                b = scalar[m] if m in scalar else local[m][p["patch_id"]]
                np.testing.assert_array_equal(pred,margin[view].astype(np.float64)>-b,
                                              err_msg=f"Locked margin prediction {e}/{m}/{p['patch_id']}")
            cls = np.where(pred,2,nonflood[view])
            cms[m] += np.bincount(target+cls,minlength=9).reshape(3,3)
        offset += n
    if offset!=len(y_score):
        raise ValueError("Patch labels do not cover scored support")
    for m,r in shifted.items():
        rank[m] = rank_pooled(r)
    cms[EVENT] = predicted_confusion(margin,nonflood,y_score,event_bias)
    checked(scores(cms[EVENT])["flood_iou"],event_iou,f"{e} exact event oracle")
    cms[PATCH],rank[PATCH],patch_oracles = apply_groups(e,PATCH,groups,margin,nonflood,y_score)
    truth = np.bincount(y_score,minlength=3)
    for cm in cms.values():
        np.testing.assert_array_equal(cm.sum(1),truth)
    return cms,rank,dict(event_id=e,bias=event_bias,flood_iou=event_iou),patch_oracles


def report(out,cms,ranks,mask_counts,event_oracles,patch_oracles):
    events = list(cms)
    n = len(events)
    one = np.ones((1,n),dtype=int)
    draws = np.random.default_rng(SEED).integers(n,size=(ITERATIONS,n))
    weights = (draws[...,None]==np.arange(n)).sum(axis=1)
    methods = {}
    for m in METHODS:
        matrices = np.stack([cms[e][m] for e in events])
        iou = np.array([scores(c)["flood_iou"] for c in matrices])
        point,boot = stats(matrices,one),stats(matrices,weights)
        point["lower_tail_iou"] = tail_bootstrap(iou,one)
        boot["lower_tail_iou"] = tail_bootstrap(iou,weights)
        methods[m] = (point,boot,iou)
    summary,event_rows,ci = [],[],[]
    for m in METHODS:
        point,_,ious = methods[m]
        diagnostic = m in (EVENT,PATCH)
        have_ap = m not in GRAPH
        ap = average_precision(rank_pooled([ranks[e][m] for e in events])) if have_ap else None
        summary.append(dict(method=m,diagnostic_only=diagnostic,events=n,patches=sum(v["patches"] for v in mask_counts.values()),
            **{k:float(v[0]) for k,v in point.items()},pooled_flood_ap=ap,
            worst_event_id=events[int(np.argmin(ious))]))
        for e in events:
            cm = cms[e][m]
            event_rows.append(dict(event_id=e,method=m,diagnostic_only=diagnostic,**mask_counts[e],**scores(cm),
                flood_ap=average_precision(ranks[e][m]) if have_ap else None,
                tp=int(cm[2,2]),fp=int(cm[:2,2].sum()),fn=int(cm[2,:2].sum())))
    for ref,m in combinations(METHODS,2):
        a,b = methods[m][0],methods[ref][0]
        for key,v in methods[m][1].items():
            lo,hi,k = interval(100*(v-methods[ref][1][key]))
            ci.append(dict(method=m,reference=ref,oracle_involved=bool({m,ref}&{EVENT,PATCH}),metric=key,
                delta_pp=100*(float(a[key][0])-float(b[key][0])),ci_low_pp=lo,ci_high_pp=hi,iterations=k))
    bench = [dict(method=r["method"],pooled_miou=r["pooled_miou"],pooled_flood_f1=r["pooled_flood_f1"],
        blackbench_miou=.762,blackbench_flood_f1=.8012,miou_gap_pp=100*(r["pooled_miou"]-.762),
        flood_f1_gap_pp=100*(r["pooled_flood_f1"]-.8012),comparison="descriptive only; unmatched patch-level protocol",
        source=PAPER) for r in summary if not r["diagnostic_only"]]
    with TemporaryDirectory(dir=out) as temp:
        staging = Path(temp)/"packet"
        staging.mkdir()
        for name,rows in (("summary",summary),("event_metrics",event_rows),("paired_bootstrap",ci),
                          ("blackbench_reference",bench),("test_coverage",[dict(event_id=e,**v) for e,v in mask_counts.items()])):
            save_csv(staging/f"{name}.csv",rows)
        diagnostics = staging/"oracle_DIAGNOSTIC_ONLY"
        diagnostics.mkdir()
        save_csv(diagnostics/"event_biases.csv",event_oracles)
        save_csv(diagnostics/"patch_biases.csv",patch_oracles)
        np.savez_compressed(staging/"event_confusions.npz",event_ids=events,methods=METHODS,
            confusion=np.stack([[cms[e][m] for m in METHODS] for e in events]))
        write_json(staging/"evaluation_config.json",dict(
            scope=SCOPE,split="test",events=n,patches=2000,
            test_predictions_lock_sha256=digest(out/"test_predictions_lock.json"),
            fingerprint_sha256=digest(out/"test_manifest.json"),footprint_audit_sha256=digest(out/"footprint_audit.json"),
            oracle="annotation-derived upper bounds under one bias per event or patch; diagnostics only",
            ap="exact grouped-tie pooled AP from continuous adjusted margins, point estimates; Graph Cut masks have no AP",
            bootstrap="10000 paired event-level percentile replicates, seed 42; metric from resampled confusion counts",
            source_selection="source OOF fits and frozen prior settings only; IW adapts to unlabeled test inputs",
            blackbench=dict(source=PAPER,table=2,model="BlackBench U-Net ResNet50, three GRD acquisitions, no DEM/slope",
                miou=.762,flood_f1=.8012,patch_scope="GEO-Bench-2 official 2000 test patches; exact BlackBench patch "
                "and scored pixel correspondence unverified; protocol and modalities differ; no BlackBench confidence interval")))
        staging.rename(out/"results")


def score_test(out,cfg):
    # No label-reader entry point before the necessary manifests and predictions have been verified.
    path,manifest = prediction_lock(out,cfg)
    if (out/"results/summary.csv").exists():
        print("Reuse completed historical-test benchmark",flush=True)
        return
    index,patches = lookup([(path,manifest)])
    locks = load_json(out/"test_bias_lock.json")["events"]
    archive = RawArchive(cfg["raw"])
    test = test_rows(archive.frame,out,cfg)
    cms,ranks,support,event_oracle,patch_oracle = {},{},{},[],[]
    for e in sorted(patches,key=int):
        ps = patches[e]
        selected = select_rows(test,[p["patch_id"] for p in ps])
        z = values(index,e,"logits")
        packed = read_predictions(out/f"predictions/{e}.npz",ps)
        y = labels_for_event(archive,selected,ps,cache_path(path,e,"masks"),len(z))
        c,r,ev,po = score_event(e,z,y,ps,packed,locks[e])
        cms[e],ranks[e] = c,r
        event_oracle.append(ev)
        patch_oracle.extend(po)
        support[e] = dict(patches=len(ps),input_valid_pixels=len(y),scored_pixels=int(np.count_nonzero(y!=IGNORE)),
                          ignored_input_valid_pixels=int(np.count_nonzero(y==IGNORE)),
                          flood_pixels=int(np.count_nonzero(y==2)))
        print(f"Scored locked predictions and diagnostic oracles for event {e}",flush=True)
    report(out,cms,ranks,support,event_oracle,patch_oracle)
    print(f"Completed {SCOPE}: {out/'results'}",flush=True)


class BenchmarkTests(unittest.TestCase):
    def test_lock_accepts_target_with_only_margin_features(self):
        z = np.arange(24,dtype=np.float32).reshape(8,3)
        y = np.array([0,2,1,2,0,1,2,0],dtype=np.uint8)
        sar,extra = z/10,z[:,:2]/10
        source_p = dict(patch_id="a",event_id="1",input_pixels=8,margin_mean=0.)
        target_p = dict(patch_id="b",event_id="205",input_pixels=8,margin_mean=1.)
        model = dict(feature_names=["intercept","margin_mean"],coefficients=[.5,0.],
                     source_feature_scaler=[dict(feature="margin_mean",median=0.,mean=0.,std_used=1.)])
        predictor = dict(reference_bias=3.7,models={MARGIN:model,
                         "ccla_margin_sar":dict(feature_names=["pre_mean_vv_db_mean"])})
        def arrays_for(index, keys):
            e, = index
            if e=="205":
                self.assertEqual(keys,RAW_FIELDS)
            data = dict(logits=z,labels=y,sar=sar,extra=extra)
            return [{e:data[key]} for key in keys]
        module = __import__(__name__)
        with patch.object(module,"lookup",side_effect=[({"1":None},{"1":[source_p]}),
                                                        ({"205":None},{"205":[target_p]})]), \
                patch.object(module,"arrays",side_effect=arrays_for):
            result = lock_test_biases(object(),object(),predictor,model,1.5)
        self.assertEqual(result["205"]["patch_biases"],{MARGIN:{"b":.5},ROBUST:{"b":.5}})
        self.assertEqual(set(result["205"]["scalar_biases"]),set(CONTINUOUS)-{MARGIN,ROBUST})

    def test_locked_patch_prediction_replay_and_oracles(self):
        z = np.array([[0,1,.5],[0,0,1],[2,0,0],[0,1,3],[0,0,-1]],dtype=np.float32)
        y = np.array([1,2,255,2,0],dtype=np.uint8)
        ps = [dict(patch_id="a",start=0,stop=3,input_pixels=3),dict(patch_id="b",start=3,stop=5,input_pixels=2)]
        settings = dict(scalar_biases={m:.25 for m in CONTINUOUS if m not in (MARGIN,ROBUST)},
                        patch_biases={m:dict(a=0.,b=.5) for m in (MARGIN,ROBUST)})
        margin = flood_margin(z)
        bits = []
        for method in LOCKED:
            scalar = settings["scalar_biases"]
            local = settings["patch_biases"]
            for p in ps:
                b = scalar.get(method) if method in scalar else local[MARGIN if method in GRAPH else method][p["patch_id"]]
                bits.append(np.packbits(margin[p["start"]:p["stop"]].astype(float)>-b,bitorder="little")[0])
        data = np.asarray(bits,dtype=np.uint8).reshape(len(LOCKED),2)
        cms,ranks,_e,patch = score_event("205",z,y,ps,(np.array([0,1,2]),data),settings)
        self.assertEqual(set(cms),set(METHODS))
        self.assertTrue(all(mat.sum()==4 for mat in cms.values()))
        self.assertEqual(len(patch),2)
        self.assertGreaterEqual(scores(cms[PATCH])["flood_iou"],scores(cms[EVENT])["flood_iou"])
        self.assertTrue(all(m not in ranks for m in GRAPH))
        data[0,0] ^= 1
        with self.assertRaises(AssertionError):
            score_event("205",z,y,ps,(np.array([0,1,2]),data),settings)

    def test_score_cannot_open_raw_archive_without_lock(self):
        with TemporaryDirectory() as d:
            with self.assertRaisesRegex(FileNotFoundError,"test_predictions_lock.json"):
                score_test(Path(d),{})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("plan","cache","predict","score","self-test"))
    p.add_argument("--out",type=Path)
    p.add_argument("--run",type=Path)
    p.add_argument("--robust-results",type=Path)
    p.add_argument("--graph-results",type=Path)
    p.add_argument("--historical",type=Path,default=REPO.parent/"artifacts/a3-eoa-flood-diagnostics-YtaAJg/results")
    p.add_argument("--workers",type=int,default=6)
    a = p.parse_args()
    if a.stage=="self-test":
        tests = unittest.defaultTestLoader.loadTestsFromTestCase(BenchmarkTests)
        raise SystemExit(0 if unittest.TextTestRunner(verbosity=2).run(tests).wasSuccessful() else 1)
    if a.out is None or a.workers<1:
        p.error("Provide --out and a positive --workers")
    a.out = a.out.resolve()
    if a.stage=="plan":
        for name in ("run","robust_results","graph_results","historical"):
            if getattr(a,name) is None:
                p.error(f"plan needs --{name.replace('_','-')}")
            setattr(a,name,getattr(a,name).resolve())
        plan(a)
        return
    config = load_json(a.out/"frozen_config.json")
    if (not compatible_code(config["code_sha256"],code_identity()) or config["manifest_sha256"]!=digest(a.out/"test_manifest.json")
            or config["footprint_audit_sha256"]!=digest(a.out/"footprint_audit.json")
            or any(digest(Path(f))!=h for f,h in config["source_sha256"].items())):
        raise ValueError("Frozen benchmark inputs or code changed")
    with (a.out/".test_benchmark.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.stage=="cache":
            cache_test(a.out,config)
        elif a.stage=="predict":
            if maxflow_module().__version__!=load_json(Path(config["graph_results"])/"frozen_config.json")["pymaxflow"]:
                raise ValueError("Installed PyMaxflow version does not match the frozen Graph Cut run")
            predict(a.out,config,a.workers)
        else:
            score_test(a.out,config)


if __name__=="__main__":
    main()
