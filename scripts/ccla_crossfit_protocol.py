"""Frozen train-only splits for the seed-42 CCLA pilot."""
from pathlib import Path

import numpy as np

from kuro_dataset import RAW_SHA256, REVISION
from loeo_splits import DRAWS, SETTINGS as LOEO_SETTINGS, footprint_neighbors

SETTINGS = LOEO_SETTINGS | {"seed":42}
FOLDS = (1, 2, 3)
GLOBAL_BIAS = .95
REPO = Path(__file__).resolve().parents[1]


def split_rows(frame, split):
    return frame.loc[frame["tortilla:data_split"] == split].reset_index(drop=True)


def select_rows(rows, ids):
    wanted = list(ids)
    if len(wanted)!=len(set(wanted)):
        raise ValueError("Duplicate patch ID in split manifest")
    indexed = rows.set_index(rows["tortilla:id"].astype(str),drop=False)
    return indexed.loc[wanted].reset_index(drop=True)


def verify_training_fold(frame, fold, neighbors=None):
    train = split_rows(frame,"train")
    if (fold["dataset_sha256"],fold["dataset_revision"],fold["train_draws_per_epoch"]) != (RAW_SHA256,REVISION,DRAWS):
        raise ValueError("Unexpected cross-fit dataset or draw budget")
    a,b,c = [fold[name] for name in ("source_patch_ids","holdout_patch_ids","excluded_overlap_patch_ids")]
    source,target,purged = map(set,(a,b,c))
    all_ids = train["tortilla:id"].astype(str).to_numpy()
    if (fold["fold"] not in FOLDS or len(train)!=4000 or len(set(train.actid))!=27
            or source|target|purged != set(all_ids) or source&target or source&purged or target&purged
            or any(len(ids)!=len(set(ids)) for ids in (a,b,c))):
        raise ValueError("Incomplete or overlapping cross-fit patch roles")
    src = select_rows(train,a)
    held = select_rows(train,b)
    src_ids, held_ids = set(src.actid.astype(int)),set(held.actid.astype(int))
    if (len(src_ids)!=18 or len(held_ids)!=9 or src_ids&held_ids
            or src_ids!=set(fold["source_events"]) or held_ids!=set(fold["holdout_events"])):
        raise ValueError("Manifest is not an 18/9 event-disjoint fold")
    if set(b)!=set(train.loc[train.actid.astype(int).isin(held_ids),"tortilla:id"].astype(str)):
        raise ValueError("Held-out events must be covered in full")
    nearby = footprint_neighbors(train) if neighbors is None else neighbors
    targets = np.flatnonzero(np.isin(all_ids,b))
    overlap = {str(all_ids[j]) for i in targets for j in nearby[i]}-target
    if purged!=overlap:
        raise ValueError("Source exclusions disagree with the held-out footprint audit")
    return src


def verify_plan(frame, plan, scope):
    if (plan["dataset_sha256"],plan["dataset_revision"],plan["split"]) != (RAW_SHA256,REVISION,"official train only"):
        raise ValueError("Expected the audited official-training plan")
    if [f["fold"] for f in plan["folds"]]!=list(FOLDS):
        raise ValueError("Expected exactly three ordered folds")
    train = split_rows(frame,"train")
    validation = split_rows(frame,"validation")
    nearby = footprint_neighbors(train)
    records = []
    all_oof = []
    for row in plan["folds"]:
        fold = row | dict(dataset_sha256=RAW_SHA256,dataset_revision=REVISION,train_draws_per_epoch=DRAWS)
        verify_training_fold(frame,fold,nearby)
        all_oof.extend(fold["holdout_patch_ids"])
        records.append(fold)
    if len(all_oof)!=4000 or len(set(all_oof))!=4000:
        raise ValueError("OOF folds must cover all 4000 train patches exactly once")
    dev = frame[frame["tortilla:data_split"].isin(("train","validation"))].reset_index(drop=True)
    roles = dev["tortilla:data_split"].to_numpy()
    ids = dev["tortilla:id"].astype(str).to_numpy()
    overlapping = {ids[i] for i,near in enumerate(footprint_neighbors(dev))
                   if roles[i]=="validation" and np.any(roles[near]=="train")}
    selected = scope["retain_full_train"]["primary_validation_patch_ids"]
    if (len(validation)!=1000 or len(overlapping)!=41 or len(selected)!=959 or len(set(selected))!=959
            or set(selected)!=set(validation["tortilla:id"].astype(str))-overlapping
            or len(set(select_rows(validation,selected).actid))!=6):
        raise ValueError("Validation scope is not the audited 959-patch footprint-clean subset")
    return records,selected
