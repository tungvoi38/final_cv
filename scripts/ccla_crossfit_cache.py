"""Fixed-checkpoint Flip TTA cache and exact patch-alignment validation for CCLA."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from rasterio.warp import transform
import torch
from torch.utils.data import DataLoader, Dataset

from adapt_unet_event_bias import IGNORE, digest, flood_margin, load_json, write_json
from adapt_unet_expected_iou import sar_features
from adapt_unet_local_context import margin_features, radiometry_features
from diagnose_sar_scattering import slope_degrees
from evaluate_unet_eoa import load_model, pre_variability_inputs, seed_everything
from evaluate_unet_eoa_flip_tta import FLIPS
from kuro_dataset import normalize_physical_inputs, preprocess, read_sample

SPECS = {"logits":("<f4",3), "labels":("u1",1), "sar":("<f4",3), "extra":("<f4",2)}
PACKED_BYTES = 224*224//8


class PhysicalPatches(Dataset):
    def __init__(self, archive, rows, features_only):
        self.archive, self.rows, self.features_only = archive, rows, features_only
        affine = np.stack(rows["stac:geotransform"])
        centers = affine[:, [0,3]]+[1120,-1120]
        _, lat = transform("EPSG:3857","EPSG:4326",*centers.T)
        self.ground_m = 10*np.cos(np.radians(lat))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        self.archive.check_unchanged()
        row = self.rows.iloc[i]
        labeled = not self.features_only
        physical = preprocess(*read_sample(self.archive.path,row,include_labels=labeled),include_labels=labeled)
        mask = physical["input_valid"]
        sar,dem = normalize_physical_inputs(physical["sar_db"],physical["dem_m"],mask)
        meta = dict(event_id=str(row.actid), patch_id=str(row["tortilla:id"]), input_pixels=int(mask.sum()),
                    aoi_id=str(row.aoiid),crs=str(row["stac:crs"]),
                    affine=[float(v) for v in row["stac:geotransform"]])
        if labeled:
            slope = slope_degrees(physical["dem_m"][0],mask,self.ground_m[i])[mask]
            meta.update(radiometry_features(physical,slope))
            cov = sar_features(physical["sar_db"],mask)
            extra = np.column_stack((physical["sar_db"][2,1][mask],slope))
            label = physical["mask"]
        else:
            cov = extra = label = None
        return torch.from_numpy(sar),torch.from_numpy(dem),meta,mask,label,cov,extra


def collate(batch):
    sar,dem,*columns = zip(*batch,strict=True)
    return dict(sar=torch.stack(sar),dem=torch.stack(dem)), list(zip(*columns,strict=True))


def flipped_single(model, inputs):
    logits = model(inputs).float()
    for axes in FLIPS[1:]:
        logits = logits+torch.flip(model(torch.flip(inputs,axes)).float(),axes)
    return logits/len(FLIPS)


def cache_path(root,event,field):
    return root/f"{event}.{field}.bin"


def append(path,array,dtype):
    with path.open("ab") as handle:
        np.asarray(array,dtype=dtype).tofile(handle)


@torch.inference_mode()
def export(archive, rows, path, member, identity, *, features_only=False):
    if path.exists():
        read_manifest(path, identity)
        print(f"Skip completed cache {path}",flush=True)
        return
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise ValueError("Export requires exactly one visible CUDA GPU")
    if digest(Path(member["checkpoint"]))!=member["sha256"]:
        raise ValueError("Checkpoint changed before inference")
    seed_everything(42)
    model = load_model(member,torch.device("cuda"))
    data = PhysicalPatches(archive,rows,features_only)
    loader = DataLoader(data,batch_size=2,num_workers=2,shuffle=False,
                        pin_memory=True,collate_fn=collate)
    offsets, patch_rows = {}, []
    with TemporaryDirectory(prefix="cache-",dir=path.parent) as temporary:
        staging = Path(temporary)/"packet"
        staging.mkdir()
        for step,(batch,items) in enumerate(loader,1):
            x,_ = pre_variability_inputs({key:value.cuda(non_blocking=True) for key,value in batch.items()})
            with torch.autocast("cuda"):
                output = flipped_single(model,x)
            z = output.permute(0,2,3,1).cpu().numpy()
            if z.shape!=(len(items),224,224,3) or not np.isfinite(z).all():
                raise ValueError("Unexpected cached logits")
            for logits,(meta,valid,labels,sar,extra) in zip(z,items,strict=True):
                event = meta["event_id"]
                values = logits[valid]
                start = offsets.get(event,0)
                stop = start+len(values)
                meta.update(margin_features(flood_margin(values)))
                meta.update(start=start,stop=stop)
                offsets[event] = stop
                if not features_only:
                    if not np.isin(labels,[0,1,2,IGNORE]).all() or np.any((labels!=IGNORE)&~valid):
                        raise ValueError("Invalid cache label or input mask")
                    meta["mask_index"] = (cache_path(staging,event,"masks").stat().st_size//(2*PACKED_BYTES)
                                          if cache_path(staging,event,"masks").exists() else 0)
                    for key,array in (("logits",values),("labels",labels[valid]),("sar",sar),("extra",extra)):
                        append(cache_path(staging,event,key),array,SPECS[key][0])
                    packed = np.packbits(np.stack((valid,labels!=IGNORE)).reshape(2,-1),axis=1,bitorder="little")
                    append(cache_path(staging,event,"masks"),packed,"u1")
                patch_rows.append(meta)
            if step%100==0 or step==len(loader):
                print(f"{identity['role']}: {step}/{len(loader)} batches",flush=True)
        actual = [r["patch_id"] for r in patch_rows]
        if actual!=rows["tortilla:id"].astype(str).tolist():
            raise ValueError("Cache patch order differs from the manifest")
        files = {p.name:digest(p) for p in staging.glob("*.bin")}
        write_json(staging/"manifest.json",identity | dict(event_counts=offsets,patches=patch_rows,files=files,
                   features_only=features_only,flip_axes=[list(v) for v in FLIPS],seed=42,
                   format="input-valid rows in raster order; masks packed little-endian in input-valid,scored order"))
        staging.rename(path)


def read_manifest(path, identity):
    m = load_json(path/"manifest.json")
    if (any(m[k]!=v for k,v in identity.items()) or m["seed"]!=42
            or m["flip_axes"]!=[list(v) for v in FLIPS]):
        raise ValueError(f"Cache provenance differs: {path}")
    offsets, patch_count, seen = {}, {}, set()
    for p in m["patches"]:
        e = p["event_id"]
        if (p["patch_id"] in seen or p["start"]!=offsets.get(e,0)
                or p["stop"]-p["start"]!=p["input_pixels"]):
            raise ValueError(f"Cache order or range is invalid: {path}")
        if not m["features_only"] and p["mask_index"]!=patch_count.get(e,0):
            raise ValueError(f"Cache mask order is invalid: {path}")
        patch_count[e] = patch_count.get(e,0)+1
        offsets[e] = p["stop"]
        seen.add(p["patch_id"])
    if offsets!=m["event_counts"] or any(n<=0 for n in offsets.values()):
        raise ValueError(f"Incomplete pixel coverage: {path}")
    if not m["features_only"]:
        names = {cache_path(path,e,k).name for e in offsets for k in (*SPECS,"masks")}
        if names!=set(m["files"]):
            raise ValueError(f"Incomplete cache files: {path}")
        for e,n in offsets.items():
            mask = cache_path(path,e,"masks")
            if (mask.stat().st_size!=patch_count[e]*2*PACKED_BYTES or digest(mask)!=m["files"][mask.name]):
                raise ValueError(f"Corrupted packed masks: {e}")
    return m


def lookup(manifests):
    x,patches = {},{}
    for path,m in manifests:
        for e,n in m["event_counts"].items():
            if e in x:
                raise ValueError(f"Event {e} occurs in more than one OOF fold")
            x[e] = (path,m,n)
            patches[e] = []
        for p in m["patches"]:
            patches[p["event_id"]].append(p)
    return x, patches


def values(lookup,e,key):
    path,m,n = lookup[e]
    dtype,width = SPECS[key]
    file = cache_path(path,e,key)
    if (file.stat().st_size!=n*width*np.dtype(dtype).itemsize or digest(file)!=m["files"][file.name]):
        raise ValueError(f"Changed cache file: {file}")
    return np.memmap(file,mode="r",dtype=dtype,shape=(n,width) if width>1 else (n,))
