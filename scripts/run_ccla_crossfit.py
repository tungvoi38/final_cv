#!/usr/bin/env python3
"""Plan, train, export, fit and evaluate the fixed seed-42 cross-fitted CCLA pilot."""
import argparse
import fcntl
from io import TextIOWrapper
from pathlib import Path
import re
import subprocess
import sys
from time import time_ns

import numpy as np
import torch
from scipy.stats import ks_2samp, wasserstein_distance

from adapt_unet_event_bias import digest, load_json, write_json
from adapt_unet_local_context import FEATURE_VERSION, RIDGE, SIGMA
from audit_unet_bias_failures import read_csv, save_csv
from ccla_crossfit_analysis import evaluate, fit
from ccla_crossfit_cache import export, read_manifest
from ccla_crossfit_protocol import DRAWS, FOLDS, REPO, SETTINGS, select_rows, split_rows, verify_plan
from evaluate_unet_eoa_flip_tta import FLIPS
from kuro_dataset import RawArchive, read_metadata
from unet_sma import SMA_POLICY, SMA_START_STEP

FILES = (
    "run_ccla_crossfit.py","ccla_crossfit_cache.py","ccla_crossfit_protocol.py","ccla_crossfit_analysis.py",
    "train_unet.py","unet_sma.py","kuro_dataset.py","adapt_unet_local_context.py","adapt_unet_iw_iou.py",
    "adapt_unet_event_bias.py","diagnose_unet_spatial_oracle.py","diagnose_sar_scattering.py",
    "evaluate_unet_eoa.py","evaluate_unet_eoa_flip_tta.py","evaluate_unet_eoa_diagnostics.py",
    "unet_conservative_expected_iou.py","adapt_unet_expected_iou.py","bootstrap_unet_events.py","loeo_splits.py",
)
PRE_PROGRESS_FIX_RUNNER = "27f7688b0028cebbcf857c911891d20883081ed8cbd7d0440e77ab34fe392a61"
EPOCH_PROGRESS = re.compile(r"^Epoch \d+/\d+:.*%\|")


def code_hashes():
    return {name:digest(REPO/"scripts"/name) for name in FILES}


def compatible_code_hashes(saved, current):
    if saved == current:
        return True
    if saved.get("run_ccla_crossfit.py") != PRE_PROGRESS_FIX_RUNNER:
        return False
    # Existing plans can use the terminal-only runner fix. All computational hashes stay pinned.
    return saved | {"run_ccla_crossfit.py":current["run_ccla_crossfit.py"]} == current


def stream_training(records, log, terminal):
    interactive = terminal.isatty()
    pending = None

    def finish_progress():
        if pending is not None:
            log.write(pending + "\n")
            terminal.write("\n" if interactive else pending + "\n")
            terminal.flush()

    for record in records:
        line = record.rstrip("\r\n")
        if EPOCH_PROGRESS.match(line):
            pending = line
            if interactive:
                terminal.write("\r" + line + "\033[K")
                terminal.flush()
            if record.endswith("\n"):
                finish_progress()
                pending = None
        elif line or record.endswith("\n"):
            finish_progress()
            pending = None
            log.write(line + "\n")
            terminal.write(line + "\n")
            terminal.flush()
    finish_progress()


def fold_dir(root, number):
    return root/f"fold-{number}"


def train_dir(root, number):
    folder = fold_dir(root,number)
    return folder/f"{root.name}-{folder.name}-seed42"


def full_member(record):
    path = Path(record["checkpoint"])
    if digest(path)!=record["sha256"]:
        raise ValueError("Full-train epoch-20 snapshot differs from audited checkpoint")
    ck = torch.load(path,weights_only=True,mmap=True,map_location="cpu")
    config = ck["config"]
    if (ck["epoch"]!=20 or config["seed"]!=42 or config["train_samples"]!=4000
            or config["train_draws_per_epoch"]!=4000 or config["train_split"]!="train"
            or config["input_channels"]!=11 or config["averaging"]!="sma" or ck["sma"]["count"]<1
            or any(config[k]!=SETTINGS[k] for k in (
                "epochs","batch_size","encoder_lr","decoder_lr","weight_decay","optimizer","loss","head","sampling"))):
        raise ValueError("Expected historical seed-42, 4000-patch fixed epoch-20 SMA")
    return dict(checkpoint=str(path),sha256=record["sha256"],epoch=20,seed=42,
                input_channels=11,head="a3",weights_key="model")


def plan(args):
    root = args.out
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(root)
    # No imagery or labels are decoded during planning.
    frame = read_metadata(args.raw)
    files = {name:args.audit/name for name in ("proposed_train_crossfit_manifest.json",
          "proposed_validation_scope.json","full_train_epoch20_checkpoints.csv")}
    folds, val_ids = verify_plan(frame,load_json(files["proposed_train_crossfit_manifest.json"]),
                                       load_json(files["proposed_validation_scope.json"]))
    record, = (r for r in read_csv(files["full_train_epoch20_checkpoints.csv"]) if int(r["seed"])==42)
    member = full_member(record)
    root.mkdir(parents=True,exist_ok=True)
    records = {}
    for f in folds:
        path = fold_dir(root,f["fold"])/"split_manifest.json"
        path.parent.mkdir(exist_ok=False)
        write_json(path,f)
        records[str(f["fold"])] = digest(path)
    write_json(root/"validation_manifest.json",dict(split="validation",patch_ids=val_ids,
                                                       footprint_clean=True,patch_count=959))
    write_json(root/"frozen_config.json",dict(run_id=root.name,raw=str(args.raw),seed=42,
        mode="single seed 42 SMA + Flip TTA",
        config=SETTINGS | dict(checkpoint_rule="fixed_epoch_20_no_validation",train_draws=DRAWS,
                               sma_start_step=SMA_START_STEP,sma_policy=SMA_POLICY),
        ccla=dict(features=FEATURE_VERSION,ridge=RIDGE,gaussian_sigma=SIGMA),
        flip_axes=[list(v) for v in FLIPS],code_sha256=code_hashes(),
        input_artifacts_sha256={name:digest(path) for name,path in files.items()},
        folds=records,validation_manifest_sha256=digest(root/"validation_manifest.json"),
        full_train=member))
    print(f"Frozen single-seed plan: {root}",flush=True)


def train(root, config, number):
    run = train_dir(root,number)
    split = fold_dir(root,number)/"split_manifest.json"
    if (run/"summary.json").is_file():
        frozen_fold(root,config,number)
        print(f"Skip completed training of fold {number}",flush=True)
        return
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise ValueError("Train with exactly one visible CUDA GPU")
    resume = run.exists() and any((run/"checkpoints").glob("epoch_*.pt"))
    if run.exists() and not resume:
        # No optimizer state exists to resume before the first epoch boundary.
        run.rename(run.with_name(f"{run.name}.interrupted-{time_ns()}"))
    command = [sys.executable,"-u",str(REPO/"scripts/train_unet.py"),"--repo",str(REPO),
               "--raw",config["raw"],"--out",str(run.parent),"--run-id",run.name,"--crossfit-fold",str(split)]
    for key,value in SETTINGS.items():
        command += [f"--{key.replace('_','-')}"] + ([] if value is True else [str(value)])
    if resume:
        command.append("--resume")
    with (run.parent/"train.log").open("a",encoding="utf-8",buffering=1) as log:
        log.write(subprocess.list2cmdline(command)+"\n")
        with subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT) as proc:
            with TextIOWrapper(proc.stdout,encoding="utf-8",errors="replace",newline="") as stream:
                stream_training(stream,log,sys.stdout)
            if proc.wait():
                raise subprocess.CalledProcessError(proc.returncode,command)
    frozen_fold(root,config,number)


def frozen_fold(root, config, number):
    run = train_dir(root,number)
    summary, settings = (load_json(run/p) for p in ("summary.json","config.json"))
    if (summary["selection"]!="fixed_epoch" or summary["checkpoint_epoch"]!=20 or summary["completed_epochs"]!=20
            or settings["crossfit_split_sha256"]!=config["folds"][str(number)] or settings["seed"]!=42
            or settings["val_samples"]!=0 or settings["train_draws_per_epoch"]!=4000):
        raise ValueError(f"Fold {number} is not a completed locked fixed-epoch run")
    path = run/"fixed_sma.pt"
    value = dict(checkpoint=str(path),sha256=digest(path),epoch=20,seed=42,
                 input_channels=11,head="a3",weights_key="model")
    selection = fold_dir(root,number)/"selection.json"
    if selection.exists() and load_json(selection)!=value:
        raise ValueError("Frozen fold checkpoint changed")
    if not selection.exists():
        checkpoint = torch.load(path,weights_only=True,mmap=True,map_location="cpu")
        if (checkpoint["epoch"]!=20 or checkpoint["config"]!=settings
                or checkpoint["sma"]["count"]!=summary["sma_count"] or checkpoint["sma"]["count"]<1
                or checkpoint["model"]["encoder.conv1.weight"].shape[1]!=11):
            raise ValueError("Final SMA weights do not match the completed fold")
        write_json(selection,value)
    return value


def identities(root, config):
    return dict(run_id=config["run_id"],frozen_config_sha256=digest(root/"frozen_config.json"))


def fold_packet(root, config, number):
    folder = fold_dir(root,number)
    split_path = folder/"split_manifest.json"
    split_hash = digest(split_path)
    if split_hash!=config["folds"][str(number)]:
        raise ValueError("Fold split manifest changed after planning")
    split = load_json(split_path)
    member = frozen_fold(root,config,number)
    identity = identities(root,config) | dict(
        role="oof",split="train",fold=number,split_sha256=split_hash,
        checkpoint_sha256=member["sha256"],features_only=False)
    return folder/"oof", split, member, identity


def find_oof(root, config):
    packs = []
    all_ids, all_events = set(),set()
    for number in FOLDS:
        path,split,_,identity = fold_packet(root,config,number)
        oof = read_manifest(path,identity)
        ids = [p["patch_id"] for p in oof["patches"]]
        events = set(map(int,oof["event_counts"]))
        if (ids!=split["holdout_patch_ids"] or events!=set(split["holdout_events"])
                or events&set(split["source_events"]) or all_ids.intersection(ids) or all_events&events):
            raise ValueError(f"OOF leakage or invalid coverage in fold {number}")
        all_ids.update(ids)
        all_events.update(events)
        packs.append((path,oof))
    if (len(all_ids),len(all_events))!=(4000,27):
        raise ValueError("Incomplete pooled OOF coverage")
    return packs


def predictor_input_identity(root, config, packs):
    return identities(root,config) | dict(oof_manifest_sha256={
        str(m["fold"]):digest(path/"manifest.json") for path,m in packs})


def locked_predictor(root, config, packs):
    path = root/"fit/predictor.json"
    model = load_json(path)
    identity = predictor_input_identity(root,config,packs)
    if (any(model[k]!=v for k,v in identity.items()) or digest(root/"fit/oof_patch_curves.npz")!=model["curve_sha256"]):
        raise ValueError("Predictor or OOF cache changed after fitting")
    return model,digest(path)


def validation_packet(root, config, predictor_hash):
    identity = identities(root,config) | dict(role="validation_evaluation",split="validation",
        scope_sha256=config["validation_manifest_sha256"],predictor_sha256=predictor_hash,
        checkpoint_sha256=config["full_train"]["sha256"],features_only=False)
    return root/"validation",identity


def mismatch_rows(packs, in_sample, validation):
    oof = {row["patch_id"]:(row,m["fold"]) for _,m in packs for row in m["patches"]}
    full = {row["patch_id"]:row for row in in_sample["patches"]}
    if set(oof)!=set(full):
        raise ValueError("Paired mismatch requires identical 4000 training patch IDs")
    columns = [k for k in next(iter(oof.values()))[0] if k.startswith(("margin_","prevalence_"))]
    paired,summary = [],[]
    for key,(a,fold) in oof.items():
        b = full[key]
        if a["event_id"]!=b["event_id"] or a["input_pixels"]!=b["input_pixels"]:
            raise ValueError("Paired prediction geometry differs")
        for column in columns:
            x,y = a[column],b[column]
            paired.append(dict(patch_id=key,event_id=a["event_id"],fold=fold,feature=column,
                               oof=x,full_train_on_same_patch=y,delta=None if x is None or y is None else y-x))
    for scope in ("all",*FOLDS,*[f"event_{e}" for e in sorted({p[0]["event_id"] for p in oof.values()},key=int)]):
        for col in columns:
            use = [r for r in paired if r["feature"]==col and r["delta"] is not None and
                  (scope=="all" or r["fold"]==scope or scope==f"event_{r['event_id']}")]
            delta = [r["delta"] for r in use]
            a,b = [r["oof"] for r in use],[r["full_train_on_same_patch"] for r in use]
            if use:
                summary.append(dict(scope=scope,feature=col,n=len(use),oof_median=float(np.median(a)),
                    full_train_median=float(np.median(b)),median_paired_delta=float(np.median(delta)),
                    mean_absolute_delta=float(np.mean(np.abs(delta))),ks=float(ks_2samp(a,b).statistic),
                    wasserstein=float(wasserstein_distance(a,b))))
    unpaired = []
    for col in columns:
        a = [r[col] for r,_ in oof.values() if r[col] is not None]
        for event in sorted(validation["event_counts"],key=int):
            b = [r[col] for r in validation["patches"] if r["event_id"]==event and r[col] is not None]
            if a and b:
                unpaired.append(dict(event_id=event,feature=col,n_oof=len(a),n_validation=len(b),
                     oof_median=float(np.median(a)),validation_median=float(np.median(b)),
                     ks=float(ks_2samp(a,b,method="asymp").statistic),wasserstein=float(wasserstein_distance(a,b))))
    return paired,summary,unpaired


def mismatch(root, config, packs, validation, identity):
    path = root/"mismatch_features"
    archive = RawArchive(config["raw"])
    source = split_rows(archive.frame,"train")
    inputs = identities(root,config) | dict(role="paired_full_train_diagnostic",split="train",
                         checkpoint_sha256=config["full_train"]["sha256"],features_only=True)
    export(archive,source,path,config["full_train"],inputs,features_only=True)
    paired,summary,unpaired = mismatch_rows(packs,read_manifest(path,inputs),validation)
    out = root/"mismatch"
    out.mkdir(exist_ok=False)
    for name,rows in (("paired_patch_features",paired),("paired_summary",summary),("oof_vs_validation",unpaired)):
        save_csv(out/f"{name}.csv",rows)
    write_json(out/"diagnostic_config.json",identity | dict(
        paired="same 4000 input patches: OOF model vs full-train model; in-sample familiarity and training-set size both differ",
        unpaired="OOF Train vs full-train model on footprint-clean 959 validation patches",
        labels_used=False,predictor_updated=False))


def dispatch(args, config):
    root, command = args.out,args.command
    if command in ("train","export-oof"):
        number = args.fold
        if digest(fold_dir(root,number)/"split_manifest.json")!=config["folds"][str(number)]:
            raise ValueError("Fold manifest changed")
        if command=="train":
            train(root,config,number)
        else:
            path,split,member,identity = fold_packet(root,config,number)
            archive = RawArchive(config["raw"])
            held = select_rows(split_rows(archive.frame,"train"),split["holdout_patch_ids"])
            export(archive,held,path,member,identity)
        return

    packs = find_oof(root,config)
    base = predictor_input_identity(root,config,packs)
    if command=="fit":
        if (root/"fit/predictor.json").is_file():
            locked_predictor(root,config,packs)
            print("Skip completed predictor",flush=True)
        else:
            fit(packs,root/"fit",base)
        return
    predictor,predictor_hash = locked_predictor(root,config,packs)
    path,val_identity = validation_packet(root,config,predictor_hash)
    val_path = root/"validation_manifest.json"
    if digest(val_path)!=config["validation_manifest_sha256"]:
        raise ValueError("Validation patch manifest changed")
    ids = load_json(val_path)["patch_ids"]
    if command=="export-validation":
        archive = RawArchive(config["raw"])
        rows = select_rows(split_rows(archive.frame,"validation"),ids)
        export(archive,rows,path,config["full_train"],val_identity)
        return
    validation = read_manifest(path,val_identity)
    if [p["patch_id"] for p in validation["patches"]]!=ids or len(ids)!=959:
        raise ValueError("Validation cache differs from locked footprint-clean patch scope")
    identity = base | dict(predictor_sha256=predictor_hash,validation_cache_sha256=digest(path/"manifest.json"))
    if command=="evaluate":
        evaluate(packs,(path,validation),predictor,root/"results",identity)
    elif command=="mismatch":
        mismatch(root,config,packs,validation,identity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command",required=True)
    for name in ("plan","train","export-oof","fit","export-validation","evaluate","mismatch"):
        p = sub.add_parser(name)
        p.add_argument("--out",type=Path,required=True)
        if name=="plan":
            p.add_argument("--audit",type=Path,required=True)
            p.add_argument("--raw",type=Path,required=True)
        if name in ("train","export-oof"):
            p.add_argument("--fold",type=int,choices=FOLDS,required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.command=="plan":
        args.raw = args.raw.resolve()
        plan(args)
        return
    config = load_json(args.out/"frozen_config.json")
    if not compatible_code_hashes(config["code_sha256"],code_hashes()) or config["run_id"]!=args.out.name:
        raise ValueError("Pilot code or run directory changed after planning")
    where = fold_dir(args.out,args.fold) if hasattr(args,"fold") else args.out
    with (where/".pilot.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        dispatch(args,config)


if __name__ == "__main__":
    main()
