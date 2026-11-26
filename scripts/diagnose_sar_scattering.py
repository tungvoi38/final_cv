#!/usr/bin/env python3
"""Exploratory flood SAR radiometry and morphology from the pinned raster archive."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform, transform_bounds
from scipy.ndimage import binary_erosion, uniform_filter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from kuro_dataset import KuroDataset, RAW_SHA256, REVISION

FOCUS = ("411", "445")
SLOPE_EDGES = [0, 2, 5, 10, 20, 90]
RADIOMETRY = ["post_vv_db", "post_vh_db", "delta_vv_db", "delta_vh_db", "post_vh_minus_vv_db"]
TEXTURE = [f"post_{band}_variance_{size}x{size}{suffix}_db2"
           for size in (3, 5) for band in ("vv", "vh") for suffix in ("", "_flood_only")]
FEATURES = RADIOMETRY + TEXTURE


def save_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def texture(image, valid, size):
    """Population variance in dB², retained only on fully observed windows."""
    x = np.where(valid, image, 0).astype(np.float64)
    variance = np.maximum(uniform_filter(x*x, size, mode="constant")
                          - uniform_filter(x, size, mode="constant")**2, 0)
    full = binary_erosion(valid, structure=np.ones((size, size), dtype=bool))
    return np.where(full, variance, np.nan)


def features(sar, valid, flood):
    post = sar[2]
    delta = post - (sar[0] + sar[1])/2
    radiometry = [post[0], post[1], delta[0], delta[1], post[1]-post[0]]
    values = []
    for size in (3, 5):
        interior = binary_erosion(flood, structure=np.ones((size, size), dtype=bool))
        for band in range(2):
            var = texture(post[band], valid, size)
            values.extend((var, np.where(interior, var, np.nan)))
    return np.stack([image[flood] for image in (*radiometry, *values)], axis=-1).astype(np.float32)


def slope_degrees(dem, valid, ground_m):
    clean = valid & (dem >= -500) & (dem <= 9000)
    support = binary_erosion(clean)  # center and 4 immediate neighbors
    out = np.full(dem.shape, np.nan, dtype=np.float32)
    dx = (dem[1:-1, 2:] - dem[1:-1, :-2]) / (2*ground_m)
    dy = (dem[2:, 1:-1] - dem[:-2, 1:-1]) / (2*ground_m)
    out[1:-1, 1:-1] = np.degrees(np.arctan(np.hypot(dx, dy)))
    out[~support] = np.nan
    return out


def to_common_grid(sample, affine, longitude, latitude):
    """Average SAR in linear power on a 12 m UTM grid before converting to dB."""
    zone = min(60, int((longitude + 180)//6)+1)
    crs = f"EPSG:{(32600 if latitude >= 0 else 32700)+zone}"
    left, top = affine[0], affine[3]
    bbox = transform_bounds("EPSG:3857", crs, left, top-2240, left+2240, top)
    cell = 12.
    x = np.floor(bbox[0]/cell)*cell
    y = np.ceil(bbox[3]/cell)*cell
    width, height = int(np.ceil((bbox[2]-x)/cell)), int(np.ceil((y-bbox[1])/cell))
    old, new = from_origin(left, top, 10, 10), from_origin(x, y, cell, cell)
    valid = sample["valid_mask"]
    power = np.where(valid[None], 10**(sample["sar_db"].reshape(6, 224, 224)/10), 0)
    dem = sample["dem_m"][0]
    dem_ok = valid & (dem >= -500) & (dem <= 9000)
    stack = np.concatenate((power, valid[None], (sample["mask"] == 2)[None],
                            np.where(dem_ok, dem, 0)[None], dem_ok[None])).astype(np.float32)
    output = np.zeros((len(stack), height, width), dtype=np.float32)
    reproject(stack, output, src_crs="EPSG:3857", dst_crs=crs, src_transform=old,
              dst_transform=new, resampling=Resampling.average)
    # Require a full source neighborhood at the target center. The extra native
    # pixel of buffer excludes partially clipped destination footprints.
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    gx, gy = transform(crs, "EPSG:3857", (x+(cols.ravel()+.5)*cell).tolist(),
                       (y-(rows.ravel()+.5)*cell).tolist())
    gx, gy = np.reshape(gx, (height, width)), np.reshape(gy, (height, width))
    buffer_m = 20
    inside = ((gx >= left+buffer_m) & (gx <= left+2240-buffer_m)
              & (gy >= top-2240+buffer_m) & (gy <= top-buffer_m))
    ok = (output[6] >= .99999) & inside & (output[:6] > 0).all(axis=0)
    flood = ok & (output[7] >= .99999)
    with np.errstate(divide="ignore"):
        sar = (10*np.log10(np.where(ok[None], output[:6], 1))).reshape(3, 2, height, width)
    slopes = slope_degrees(output[8], ok & (output[9] >= .99999), cell)
    return features(sar, ok, flood), slopes[flood]


def overlap_rectangles(corners, events):
    exclusions = defaultdict(list)
    for i, (x, top) in enumerate(corners):
        matches = np.flatnonzero((events[i+1:] == events[i])
                                 & (np.abs(corners[i+1:]-[x, top]) < 2239.999999).all(axis=1))+i+1
        for j in matches:
            x2, t2 = corners[j]
            bounds = (max(x,x2), min(x,x2)+2240, max(top,t2)-2240, min(top,t2))
            for idx, (px, py) in ((i, (x, top)), (j, (x2, t2))):
                l,r,b,t = bounds
                exclusions[idx].append(tuple(round(p/10) for p in (py-t,py-b,l-px,r-px)))
    return exclusions


def component_sizes(points, connectivity):
    # Zero padding prevents encoded row wrap when searching neighbor columns.
    x, y = points[:, :2].astype(np.int64).T
    x, y = x-x.min()+1, y-y.min()+1
    stride = x.max()+2
    code = y*stride+x
    order = np.argsort(code)
    code = code[order]
    if np.any(np.diff(code) == 0):
        raise ValueError("Duplicate morphology coordinates after overlap removal")
    shifts = [1, stride] + ([stride-1, stride+1] if connectivity == 8 else [])
    source, dest = [], []
    for shift in shifts:
        target = code + shift
        idx = np.searchsorted(code, target)
        ii = np.flatnonzero(idx < len(code))
        ii = ii[code[idx[ii]] == target[ii]]
        source.append(ii)
        dest.append(idx[ii])
    u, v = np.concatenate(source), np.concatenate(dest)
    graph = coo_matrix((np.ones(len(u), dtype=bool), (u,v)), shape=(len(code), len(code))).tocsr()
    n, labels = connected_components(graph, directed=False)
    return (np.bincount(labels, minlength=n), np.bincount(labels, weights=points[order, 2], minlength=n),
            np.bincount(labels, weights=points[order, 3], minlength=n) > 0)


def stat_rows(data, fields):
    for col, feature in enumerate(FEATURES):
        x = data[:, col]
        x = x[np.isfinite(x)]
        q = np.quantile(x, [.05,.25,.5,.75,.95]) if len(x) else [None]*5
        yield dict(**fields, feature=feature, n_pixels=len(x), q05=q[0], q25=q[1], median=q[2], q75=q[3],
                   q95=q[4], iqr=q[3]-q[1] if len(x) else None)


def morphology(points):
    records, event_parts = [], defaultdict(list)
    for (event, aoi), chunks in sorted(points.items()):
        array = np.concatenate(chunks)
        if not len(array):
            continue
        for connectivity in (4,8):
            size, area, censored = component_sizes(array, connectivity)
            event_parts[event, connectivity].append((size, area, censored))
            records.extend(dict(event_id=event, aoi_id=aoi, connectivity=connectivity,
                                component_id=i, pixels=int(s), approx_ground_area_m2=float(a),
                                touches_patch_edge_or_invalid=bool(b))
                           for i, (s,a,b) in enumerate(zip(size,area,censored,strict=True),1))
    summary = []
    for (event, connectivity), pieces in sorted(event_parts.items(), key=lambda r:(int(r[0][0]),r[0][1])):
        size, area, censored = [np.concatenate(values) for values in zip(*pieces, strict=True)]
        summary.append(dict(event_id=event, connectivity=connectivity, pixels=int(size.sum()),
                            n_components=len(size), approx_flood_area_m2=float(area.sum()),
                            median_component_pixels=float(np.median(size)), median_component_area_m2=float(np.median(area)),
                            largest_component_area_m2=float(area.max()), largest_area_fraction=float(area.max()/area.sum()),
                            components_per_flood_km2=float(len(size)*1e6/area.sum()),
                            singleton_component_fraction=float(np.mean(size==1)),
                            conservative_censored_component_fraction=float(censored.mean()),
                            conservative_censored_area_fraction=float(area[censored].sum()/area.sum())))
    return records, summary


def figures(out, all_data, stats, images):
    events = sorted(all_data, key=int)
    fig, axs = plt.subplots(2,3,figsize=(14,8),layout="constrained")
    for ax, name in zip(axs.flat, RADIOMETRY, strict=False):
        rows = [next(r for r in stats if r["event_id"]==event and r["feature"]==name) for event in events]
        boxes = [dict(label=r["event_id"], med=r["median"], q1=r["q25"], q3=r["q75"],
                      whislo=r["q05"], whishi=r["q95"], fliers=[]) for r in rows]
        artists = ax.bxp(boxes, showfliers=False, patch_artist=True)
        for patch, event in zip(artists["boxes"],events,strict=True):
            patch.set_facecolor({"411":"#e3a154","445":"#6faecb"}.get(event,"#dedede"))
        ax.set_title(name.replace("_"," "))
        ax.tick_params(axis="x",rotation=55)
    axs.flat[-1].axis("off")
    fig.suptitle("Exploratory flood pixels • boxes Q25–Q75, whiskers Q05–Q95")
    fig.savefig(out/"event_boxplots.png",dpi=170)
    plt.close(fig)

    show = RADIOMETRY + [f"post_{band}_variance_{size}x{size}_db2" for size in (3,5) for band in ("vv","vh")]
    fig, axs = plt.subplots(3,3,figsize=(13,9),layout="constrained")
    for ax, feature in zip(axs.flat, show, strict=True):
        j=FEATURES.index(feature)
        values=[all_data[event][:,j] for event in FOCUS]
        values=[x[np.isfinite(x)] for x in values]
        bins=np.linspace(min(x.min() for x in values),max(x.max() for x in values),41)
        for event,x,color in zip(FOCUS,values,("#a75718","#1e678a"),strict=True):
            ax.hist(x,bins=bins,weights=np.full(len(x),1/len(x)),histtype="step",label=event,color=color)
        ax.set(title=feature.replace("_"," "),ylabel="Fraction per bin")
    axs.flat[0].legend()
    fig.suptitle("Exploratory histograms • flood pixels in events 411 and 445")
    fig.savefig(out/"focused_histograms.png",dpi=170)
    plt.close(fig)

    cmap=matplotlib.colors.ListedColormap(["#252525","#4386a7","#ec992f","#bebebe"])
    fig, axs = plt.subplots(2,3,figsize=(10,7),layout="constrained")
    for i,event in enumerate(FOCUS):
        n,sample=images[event]
        valid=sample["valid_mask"]
        for j,(band,limits) in enumerate((("VV",(-30,0)),("VH",(-37,-7)))):
            image=np.where(valid,sample["sar_db"][2,j],np.nan)
            axs[i,j].imshow(image,cmap="gray",vmin=limits[0],vmax=limits[1])
            axs[i,j].set_title(f"{event} post {band} (dB)")
        mask=sample["mask"]
        axs[i,2].imshow(np.where(mask==255,3,mask),cmap=cmap,vmin=0,vmax=3)
        axs[i,2].set_title(f"Mask • patch {sample['sample_id']} • {n} flood pixels")
        for ax in axs[i]:
            ax.axis("off")
    fig.suptitle("Illustrative max-flood patches • orange flood, blue permanent water, gray invalid")
    fig.savefig(out/"sar_mask_examples.png",dpi=175)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    data=KuroDataset(args.raw,"test")
    rows=data.rows
    affine=np.stack(rows["stac:geotransform"])
    if not (rows["stac:crs"].eq("EPSG:3857").all() and np.all(affine[:,[1,2,4,5]]==[10,0,0,-10])):
        raise ValueError("Unexpected native grid")
    corners=affine[:,[0,3]]
    events=rows.actid.astype(str).to_numpy()
    excluded=overlap_rectangles(corners,events)
    xy=corners+[1120,-1120]
    lon,lat=transform("EPSG:3857","EPSG:4326",xy[:,0],xy[:,1])
    blocks=np.floor(xy/5000).astype(int)
    grid={}
    all_pixels, block_pixels, native_slopes, grid_pixels, topo = [defaultdict(list) for _ in range(5)]
    audit, images={},{}
    for i,event in enumerate(events):
        sample=data.physical(i)
        if sample["event_id"] != event:
            raise ValueError("Unexpected event ID")
        valid=sample["valid_mask"]
        flood=(sample["mask"]==2)
        ground_m=10*np.cos(np.radians(lat[i]))
        r=audit.setdefault(event,dict(event_id=event,patches=0,valid_pixels=0,flood_pixels=0,
                                      flood_patches=0,morphology_excluded_flood_observations=0,
                                      overlap_patches=0,ground_m_min=ground_m,ground_m_max=ground_m))
        r["patches"]+=1
        r["valid_pixels"]+=int(valid.sum())
        r["flood_pixels"]+=int(flood.sum())
        r["ground_m_min"]=min(r["ground_m_min"],ground_m)
        r["ground_m_max"]=max(r["ground_m_max"],ground_m)
        if i in excluded:
            r["overlap_patches"]+=1
        if not flood.any():
            continue
        r["flood_patches"]+=1
        value=features(sample["sar_db"],valid,flood)
        all_pixels[event].append(value)
        block_pixels[event,*blocks[i]].append(value)
        slopes=slope_degrees(sample["dem_m"][0],sample["input_valid"],ground_m)[flood]
        native_slopes[event].append(slopes)
        if event in FOCUS:
            common=to_common_grid(sample,affine[i],lon[i],lat[i])
            grid_pixels[event].append(common)
            if event not in images or int(flood.sum())>images[event][0]:
                images[event]=(int(flood.sum()),sample)

        topology=flood.copy()
        topo_valid=valid.copy()
        for y0,y1,x0,x1 in excluded.get(i,[]):
            topology[y0:y1,x0:x1]=False
            topo_valid[y0:y1,x0:x1]=False
        r["morphology_excluded_flood_observations"]+=int(flood.sum()-topology.sum())
        rr,cc=np.nonzero(topology)
        aoi=sample["aoi_id"]
        origin=grid.setdefault(aoi,corners[i])
        offset=(corners[i]-origin)/10
        if not np.allclose(offset,np.rint(offset),atol=1e-6,rtol=0):
            raise ValueError(f"Patch grids do not align in {aoi}")
        xx,yy=np.rint(offset).astype(np.int64)
        border=~binary_erosion(topo_valid)
        topo[event,aoi].append(np.column_stack((cc+xx,rr-yy,np.full(len(rr),ground_m**2),border[rr,cc])))
        if (i+1)%250==0:
            print(f"Read {i+1}/{len(rows)} patches",flush=True)

    events_order=sorted(audit,key=int)
    if len(events_order)!=10 or sum(r["patches"] for r in audit.values())!=2000:
        raise ValueError("Unexpected test split coverage")
    all_data={e:np.concatenate(all_pixels[e]) for e in events_order}
    stats=[r for e in events_order for r in stat_rows(all_data[e],dict(event_id=e))]
    block_stats=[r for (e,x,y),parts in sorted(block_pixels.items())
                 for r in stat_rows(np.concatenate(parts),dict(event_id=e,block_x=int(x),block_y=int(y),patches=len(parts)))]
    strata=[]
    for e in events_order:
        resolutions=[("native",all_data[e],np.concatenate(native_slopes[e]))]
        if e in grid_pixels:
            resolutions.append(("12m_utm",np.concatenate([a for a,_ in grid_pixels[e]]),
                                np.concatenate([b for _,b in grid_pixels[e]])))
        for resolution, pixels, slopes in resolutions:
            for label, include in [("all",np.ones(len(slopes),dtype=bool)),
                                   ("unknown",~np.isfinite(slopes)),
                                   *[(f"{lo}-{hi}",(slopes>=lo)&(slopes<hi))
                                     for lo,hi in zip(SLOPE_EDGES,SLOPE_EDGES[1:])]]:
                strata.extend(stat_rows(pixels[include],dict(event_id=e,resolution=resolution,
                                                            slope_degrees=label,flood_pixels=int(include.sum()))))
    components, morphology_rows=morphology(topo)
    for e in events_order:
        r=audit[e]
        r["flood_blocks_5km_projected"]=sum(k[0]==e for k in block_pixels)
        assert len(all_data[e])==r["flood_pixels"]
        for m in morphology_rows:
            if m["event_id"]==e:
                assert m["pixels"]+r["morphology_excluded_flood_observations"]==r["flood_pixels"]
    save_csv(args.out/"event_audit.csv",[audit[e] for e in events_order])
    save_csv(args.out/"event_sar_texture.csv",stats)
    save_csv(args.out/"slope_and_resolution.csv",strata)
    save_csv(args.out/"spatial_blocks.csv",block_stats)
    save_csv(args.out/"flood_components.csv",components)
    save_csv(args.out/"event_morphology.csv",morphology_rows)
    figures(args.out,all_data,stats,images)
    manifest=dict(scope="historical test exploratory; no model or new test inference",raw=str(args.raw.resolve()),
                  dataset_sha256=RAW_SHA256,revision=REVISION,ratio_db="VH_dB − VV_dB = 10 log10(VH_power/VV_power)",
                  delta="post dB minus mean of two pre dB values",texture="population variance in dB²; complete valid windows",
                  slope="centered DEM differences with Web Mercator cos(latitude) correction for native cells",
                  resolution="native projected grid 10m; sensitivity at 12m UTM for 411/445, area-average linear power, homogeneous flood cells",
                  morphology="merge across observed aligned tile seams within AOI; 4 and 8 neighbors; remove overlapping patch footprint from both copies only for topology",
                  censoring="conservative: any patch edge or invalid-neighbor contact; even a component crossing an observed seam may be flagged",
                  units="area approximate using projected cell area weighted by cos(latitude)^2",
                  uncertainty="descriptive 5 km projected blocks, no pixel-level p-value; each focal event represents one AOI")
    (args.out/"config.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(f"Saved {len(events_order)} exploratory event summaries: {args.out}",flush=True)


if __name__=="__main__":
    main()
