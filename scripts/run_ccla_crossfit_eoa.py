#!/usr/bin/env python3
"""Extend a completed seed-42 CCLA pilot to three fixed-epoch SMA members."""
import argparse
import fcntl
from io import TextIOWrapper
from pathlib import Path
import subprocess
import sys
from time import time_ns

import torch

from adapt_unet_event_bias import digest, load_json, write_json
from adapt_unet_local_context import FEATURE_VERSION, RIDGE, SIGMA
from audit_unet_bias_failures import read_csv, save_csv
from ccla_crossfit_analysis import fit
from ccla_crossfit_cache import lookup, read_manifest, values
from ccla_crossfit_protocol import FOLDS, REPO, SETTINGS, select_rows, split_rows
from ccla_eoa_cache import SEEDS, export_ensemble, export_paired_diagnostic
from ccla_eoa_reports import evaluate_ensemble
from evaluate_unet_eoa_flip_tta import FLIPS
from kuro_dataset import RawArchive
from run_ccla_crossfit import (
    code_hashes, find_oof, fold_dir, locked_predictor, mismatch_rows, predictor_input_identity,
    stream_training, validation_packet,
)

NEW_SEEDS = SEEDS[1:]


def hashes():
    return code_hashes() | {f:digest(REPO/"scripts"/f) for f in
        ("run_ccla_crossfit_eoa.py","ccla_eoa_cache.py","ccla_eoa_reports.py")}


def full_member(row):
    path = Path(row["checkpoint"])
    if digest(path)!=row["sha256"]:
        raise ValueError("Full-train snapshot differs from the audited epoch-20 checkpoint")
    seed = int(row["seed"])
    state = torch.load(path,weights_only=True,mmap=True,map_location="cpu")
    cfg = state["config"]
    keys = ("epochs","batch_size","encoder_lr","decoder_lr","weight_decay","optimizer","loss","head","sampling")
    if (state["epoch"]!=20 or cfg["seed"]!=seed or cfg["train_samples"]!=4000
            or cfg["train_draws_per_epoch"]!=4000 or cfg["train_split"]!="train"
            or cfg["input_channels"]!=11 or cfg["averaging"]!="sma" or state["sma"]["count"]<1
            or any(cfg[k]!=SETTINGS[k] for k in keys)):
        raise ValueError(f"Unexpected full-train snapshot for seed {seed}")
    return dict(checkpoint=str(path),sha256=row["sha256"],epoch=20,seed=seed,
                input_channels=11,head="a3",weights_key="model")


def pilot_artifacts(pilot):
    config = load_json(pilot/"frozen_config.json")
    if config["run_id"]!=pilot.name or config["seed"]!=42:
        raise ValueError("Expected an existing seed-42 CCLA pilot")
    packs = find_oof(pilot,config)
    _, predictor_hash = locked_predictor(pilot,config,packs)
    path, identity = validation_packet(pilot,config,predictor_hash)
    val = read_manifest(path,identity)
    scope = pilot/"validation_manifest.json"
    ids = load_json(scope)["patch_ids"]
    if (digest(scope)!=config["validation_manifest_sha256"] or len(ids)!=959
            or len(set(ids))!=959 or [p["patch_id"] for p in val["patches"]]!=ids
            or len(val["event_counts"])!=6):
        raise ValueError("Expected verified pilot cache for the 959 footprint-clean patches")
    return config,packs,(path,val)


def plan(args):
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(args.out)
    pilot,packs,val = pilot_artifacts(args.pilot)
    if (any(pilot["config"][key]!=value for key,value in SETTINGS.items())
            or pilot["ccla"]!=dict(features=FEATURE_VERSION,ridge=RIDGE,gaussian_sigma=SIGMA)
            or pilot["flip_axes"]!=[list(a) for a in FLIPS]):
        raise ValueError("Pilot training, CCLA parameters or Flip TTA changed")
    for path,m in [*packs,val]:
        index,_ = lookup([(path,m)])
        for event in index:
            values(index,event,"logits")  # Validate FP32 anchors now, without running seed 42.
    rows = sorted(read_csv(args.audit/"full_train_epoch20_checkpoints.csv"),key=lambda r:int(r["seed"]))
    if tuple(int(r["seed"]) for r in rows)!=SEEDS:
        raise ValueError("Expected the audit of all three full-train epoch-20 checkpoints")
    full = [full_member(row) for row in rows]
    if full[0]!=pilot["full_train"]:
        raise ValueError("Full-train seed 42 differs from the actual pilot checkpoint")
    summary = read_csv(args.pilot/"results/summary.csv")
    if not summary or not all("event_flood_iou_std" in r for r in summary) or not any(
            r["method"]=="event_oracle" for r in summary):
        raise ValueError("Expected pilot validation IoU std and Event Oracle on the 959 patches")
    args.out.mkdir(parents=True,exist_ok=True)
    for number in FOLDS:
        dst = fold_dir(args.out,number)/"split_manifest.json"
        dst.parent.mkdir(exist_ok=False)
        dst.write_bytes((fold_dir(args.pilot,number)/dst.name).read_bytes())
    (args.out/"validation_manifest.json").write_bytes((args.pilot/"validation_manifest.json").read_bytes())
    save_csv(args.out/"pilot_959_summary.csv",summary)
    write_json(args.out/"frozen_config.json",dict(
        run_id=args.out.name,pilot=str(args.pilot),pilot_config_sha256=digest(args.pilot/"frozen_config.json"),
        pilot_validation_summary_sha256=digest(args.pilot/"results/summary.csv"),
        pilot_oof_cache_sha256={str(m["fold"]):digest(path/"manifest.json") for path,m in packs},
        pilot_validation_cache_sha256=digest(val[0]/"manifest.json"),
        raw=pilot["raw"],folds=pilot["folds"],validation_manifest_sha256=pilot["validation_manifest_sha256"],
        training=pilot["config"],ccla=pilot["ccla"],full_train=full,seeds=list(SEEDS),
        ensemble="mean FP32 logits after per-member Flip TTA, seeds 42/43/44 in order; SMA fixed epoch 20",
        flip_axes=[list(a) for a in FLIPS],code_sha256=hashes()))
    print(f"Frozen three-seed plan: {args.out}",flush=True)


def identity(root):
    return dict(run_id=root.name,frozen_config_sha256=digest(root/"frozen_config.json"),member_seeds=list(SEEDS))


def seat(root,fold,seed):
    return fold_dir(root,fold)/f"seed-{seed}"


def trained_member(root,config,fold,seed):
    cache = seat(root,fold,seed)
    run = cache/f"{root.name}-fold-{fold}-seed{seed}"
    summary, settings = (load_json(run/name) for name in ("summary.json","config.json"))
    if (settings["seed"]!=seed or settings["crossfit_fold"]!=fold or settings["val_samples"]!=0
            or settings["crossfit_split_sha256"]!=config["folds"][str(fold)]
            or settings["train_draws_per_epoch"]!=4000 or settings["input_channels"]!=11
            or summary["checkpoint_epoch"]!=20 or summary["completed_epochs"]!=20
            or summary["selection"]!="fixed_epoch" or summary["sma_count"]<1):
        raise ValueError(f"Unexpected fold/seed checkpoint: {fold}/{seed}")
    path = run/"fixed_sma.pt"
    member = dict(checkpoint=str(path),sha256=digest(path),epoch=20,seed=seed,
                  input_channels=11,head="a3",weights_key="model")
    selection = cache/"selection.json"
    if selection.exists():
        if load_json(selection)!=member:
            raise ValueError("Previously frozen member changed")
    else:
        ck = torch.load(path,weights_only=True,mmap=True,map_location="cpu")
        if ck["config"]!=settings or ck["epoch"]!=20 or ck["sma"]["count"]!=summary["sma_count"]:
            raise ValueError("Fixed checkpoint disagrees with completion record")
        write_json(selection,member)
    return member


def train(root,config,fold,seed):
    base = seat(root,fold,seed)
    base.mkdir(exist_ok=True)
    run = base/f"{root.name}-fold-{fold}-seed{seed}"
    with (base/".train.lock").open("a") as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (run/"summary.json").is_file():
            trained_member(root,config,fold,seed)
            print(f"Skip completed fold {fold}, seed {seed}",flush=True)
            return
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
            raise ValueError("Training requires exactly one visible CUDA GPU")
        resume = run.exists() and any((run/"checkpoints").glob("epoch_*.pt"))
        if run.exists() and not resume:
            run.rename(run.with_name(f"{run.name}.interrupted-{time_ns()}"))
        cmd = [sys.executable,"-u",str(REPO/"scripts/train_unet.py"),"--repo",str(REPO),
               "--raw",config["raw"],"--out",str(base),"--run-id",run.name,
               "--crossfit-fold",str(fold_dir(root,fold)/"split_manifest.json")]
        for key,value in (SETTINGS | dict(seed=seed)).items():
            cmd += [f"--{key.replace('_','-')}"]+([] if value is True else [str(value)])
        if resume:
            cmd.append("--resume")
        with (base/"train.log").open("a",encoding="utf-8",buffering=1) as log:
            log.write(subprocess.list2cmdline(cmd)+"\n")
            with subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT) as proc:
                with TextIOWrapper(proc.stdout,encoding="utf-8",errors="replace",newline="") as stream:
                    stream_training(stream,log,sys.stdout)
                if proc.wait():
                    raise subprocess.CalledProcessError(proc.returncode,cmd)
        trained_member(root,config,fold,seed)


def fold_inputs(root,config,fold,pilot):
    dst = fold_dir(root,fold)
    plan = load_json(dst/"split_manifest.json")
    original = load_json(fold_dir(Path(config["pilot"]),fold)/"selection.json")
    members = [original,*[trained_member(root,config,fold,s) for s in NEW_SEEDS]]
    ancestry = identity(root) | dict(
        role="oof_ensemble",split="train",fold=fold,features_only=False,
        split_sha256=config["folds"][str(fold)],checkpoint_sha256=[m["sha256"] for m in members],
        seed42_cache_sha256=digest(pilot[0]/"manifest.json"))
    return dst/"oof",plan,members,ancestry


def all_oof(root,config,pilot):
    packs, seen_ids, seen_events = [],set(),set()
    for fold,anchor in zip(FOLDS,pilot,strict=True):
        path, plan, _, ancestry = fold_inputs(root,config,fold,anchor)
        m = read_manifest(path,ancestry)
        ids = [p["patch_id"] for p in m["patches"]]
        events = set(map(int,m["event_counts"]))
        if (ids!=plan["holdout_patch_ids"] or events!=set(plan["holdout_events"])
                or events&set(plan["source_events"]) or seen_ids.intersection(ids) or events&seen_events):
            raise ValueError(f"OOF event/split leakage in fold {fold}")
        seen_ids.update(ids)
        seen_events.update(events)
        packs.append((path,m))
    if len(seen_ids)!=4000 or len(seen_events)!=27:
        raise ValueError("Expected all 4000 unique OOF patches from 27 events")
    return packs


def check_anchors(config,packs,val):
    named = {str(m["fold"]):digest(path/"manifest.json") for path,m in packs}
    if named!=config["pilot_oof_cache_sha256"] or digest(val[0]/"manifest.json")!=config["pilot_validation_cache_sha256"]:
        raise ValueError("Seed-42 cache manifests changed after planning")


def mismatch(root,config,pilot,ensemble,pilot_val,val):
    out = root/"mismatch"
    if (out/"diagnostic_config.json").is_file():
        print(f"Skip completed diagnostic {out}",flush=True)
        return
    old_summary = Path(config["pilot"])/"results/summary.csv"
    if digest(old_summary)!=config["pilot_validation_summary_sha256"]:
        raise ValueError("Pilot metrics changed after the ensemble was planned")
    archive = RawArchive(config["raw"])
    ids = split_rows(archive.frame,"train")
    ancestral = identity(root) | dict(role="paired_full_train_diagnostic",split="train",
        features_only=True,checkpoint_sha256=[r["sha256"] for r in config["full_train"]])
    baseline,full = export_paired_diagnostic(archive,ids,root/"mismatch_features",config["full_train"],ancestral)
    out.mkdir(exist_ok=False)
    for name,oof,insample,validation in (("pilot_seed42",pilot,baseline,pilot_val),
                                          ("three_seed_eoa",ensemble,full,val)):
        dest = out/name
        dest.mkdir()
        paired,summary,shift = mismatch_rows(oof,insample,validation[1])
        for key,data in (("paired_patch_features",paired),("paired_summary",summary),("oof_vs_validation",shift)):
            save_csv(dest/f"{key}.csv",data)
    write_json(out/"diagnostic_config.json",ancestral | dict(
        labels_used=False,predictor_updated=False,paired_train_patches=4000,evaluation_patches=959,
        paired="same patch; OOF 18-event model vs full-train 27-event model",
        caveat="in-sample familiarity and training set size both differ"))
    print(f"Saved pilot and ensemble mismatch diagnostics: {out}",flush=True)


def dispatch(args,config):
    root, cmd = args.out,args.command
    pilot_root = Path(config["pilot"])
    if digest(pilot_root/"frozen_config.json")!=config["pilot_config_sha256"]:
        raise ValueError("Pilot plan changed")
    for number in FOLDS:
        if digest(fold_dir(root,number)/"split_manifest.json")!=config["folds"][str(number)]:
            raise ValueError("Cross-fit fold manifest changed")
    val_split = root/"validation_manifest.json"
    if digest(val_split)!=config["validation_manifest_sha256"]:
        raise ValueError("959-patch validation manifest changed")

    if cmd=="train":
        train(root,config,args.fold,args.seed)
        return
    _,pilot,pilot_val = pilot_artifacts(pilot_root)
    check_anchors(config,pilot,pilot_val)
    if cmd=="export-oof":
        path,plan,models,ancestry = fold_inputs(root,config,args.fold,pilot[args.fold-1])
        archive = RawArchive(config["raw"])
        held = select_rows(split_rows(archive.frame,"train"),plan["holdout_patch_ids"])
        export_ensemble(archive,held,path,pilot[args.fold-1],models,ancestry)
        return

    packs = all_oof(root,config,pilot)
    if cmd=="fit":
        if (root/"fit/predictor.json").is_file():
            locked_predictor(root,config,packs)
            print("Skip completed three-seed predictor",flush=True)
        else:
            fit(packs,root/"fit",predictor_input_identity(root,config,packs))
        return
    model,predictor_sha = locked_predictor(root,config,packs)
    ancestry = identity(root) | dict(
        role="validation_ensemble",split="validation",features_only=False,
        scope_sha256=config["validation_manifest_sha256"],predictor_sha256=predictor_sha,
        checkpoint_sha256=[m["sha256"] for m in config["full_train"]],
        seed42_cache_sha256=config["pilot_validation_cache_sha256"])
    ids = load_json(val_split)["patch_ids"]
    path = root/"validation"
    if cmd=="export-validation":
        archive = RawArchive(config["raw"])
        rows = select_rows(split_rows(archive.frame,"validation"),ids)
        export_ensemble(archive,rows,path,pilot_val,config["full_train"],ancestry)
        return
    cache = read_manifest(path,ancestry)
    if len(cache["event_counts"])!=6 or len(ids)!=959 or [p["patch_id"] for p in cache["patches"]]!=ids:
        raise ValueError("Validation cache differs from the original 959-patch scope")
    if cmd=="evaluate":
        evaluation_identity = predictor_input_identity(root,config,packs) | ancestry | dict(
            validation_cache_sha256=digest(path/"manifest.json"))
        evaluate_ensemble(packs,(path,cache),model,root/"results",evaluation_identity)
    else:
        mismatch(root,config,pilot,packs,pilot_val,(path,cache))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command",required=True)
    for name in ("plan","train","export-oof","fit","export-validation","evaluate","mismatch"):
        p = sub.add_parser(name)
        p.add_argument("--out",type=Path,required=True)
        if name=="plan":
            p.add_argument("--pilot",type=Path,required=True)
            p.add_argument("--audit",type=Path,required=True)
        if name in ("train","export-oof"):
            p.add_argument("--fold",type=int,choices=FOLDS,required=True)
        if name=="train":
            p.add_argument("--seed",type=int,choices=NEW_SEEDS,required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.command=="plan":
        args.pilot = args.pilot.resolve()
        args.audit = args.audit.resolve()
        plan(args)
        return
    config = load_json(args.out/"frozen_config.json")
    if config["code_sha256"]!=hashes() or config["run_id"]!=args.out.name:
        raise ValueError("Frozen ensemble code or run directory changed")
    if args.command=="train":
        # Per-seed locks allow different GPU jobs in the same fold.
        dispatch(args,config)
        return
    folder = fold_dir(args.out,args.fold) if args.command=="export-oof" else args.out
    with (folder/".eoa.lock").open("a") as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        dispatch(args,config)


if __name__=="__main__":
    main()
