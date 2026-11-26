"""Label-free Delta-VV histogram and guarded Otsu estimation on validation events."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from kuro_dataset import KuroDataset, RawArchive

BIN_DB = .25
EDGES = np.arange(-400., 400. + BIN_DB, BIN_DB)
RULES = {
    "delta": "post VV dB minus mean of two pre-event VV dB values",
    "bin_width_db": BIN_DB,
    "histogram_bounds_db": [float(EDGES[0]), float(EDGES[-1])],
    "water_posterior_min": .8,
    "stable_water_abs_delta_max_db": 1.,
    "smooth_sigma_db": .5,
    "min_peak_prominence_fraction": .03,
    "min_peak_separation_db": 1.,
    "max_valley_to_smaller_peak": .85,
    "min_class_count": 32,
    "min_flood_fraction": .0001,
    "max_flood_fraction": .5,
    "fallback_bias": 0.,
}


def predicted_stable_water(delta, logits):
    """Conservative proxy using frozen, raw predictions and little temporal change."""
    import scipy.special

    stable = np.flatnonzero((np.abs(delta) <= RULES["stable_water_abs_delta_max_db"])
                            & (logits.argmax(axis=1) == 1))
    excluded = np.zeros(len(delta), dtype=bool)
    if len(stable):
        scores = logits[stable].astype(np.float64)
        water_probability = np.exp(scores[:, 1] - scipy.special.logsumexp(scores, axis=1))
        excluded[stable] = water_probability >= RULES["water_posterior_min"]
    return excluded


def delta_histograms(raw_path, inputs):
    """Stream only validation SAR with the same per-event insertion order as the cache."""
    dataset = KuroDataset(RawArchive(raw_path), "validation")
    if set(dataset.rows.actid.astype(str)) != set(inputs):
        raise ValueError("SAR archive and validation logit cache have different event IDs")
    offsets = dict.fromkeys(inputs, 0)
    counts = {event: np.zeros((2, len(EDGES) - 1), dtype=np.int64) for event in inputs}
    for i in range(len(dataset)):
        sample = dataset.physical(i)
        event = sample["event_id"]
        vv = sample["sar_db"][:, 0, sample["input_valid"]]
        delta = vv[2] - (vv[0] + vv[1]) * .5
        start, stop = offsets[event], offsets[event] + len(delta)
        if stop > len(inputs[event]):
            raise ValueError(f"Delta-VV exceeds the logit cache for {event}")
        logits = inputs[event][start:stop]
        offsets[event] = stop
        if not np.isfinite(delta).all() or (delta < EDGES[0]).any() or (delta > EDGES[-1]).any():
            raise ValueError(f"Nonfinite or out-of-range Delta-VV in {event}")
        keep = ~predicted_stable_water(delta, logits)
        counts[event][0] += np.histogram(delta, bins=EDGES)[0]
        counts[event][1] += np.histogram(delta[keep], bins=EDGES)[0]
        if (i + 1) % 250 == 0 or i + 1 == len(dataset):
            print(f"validation Delta-VV: {i+1}/{len(dataset)} patches", flush=True)
    for event, (_, eligible) in counts.items():
        if offsets[event] != len(inputs[event]) or counts[event][0].sum() != len(inputs[event]):
            raise ValueError(f"Validation logits and raw input-valid counts differ for {event}")
        if eligible.sum() > offsets[event]:
            raise ValueError(f"Invalid histogram counts for {event}")
    return counts


def otsu(counts_all, counts_eligible):
    total, eligible = int(counts_all.sum()), int(counts_eligible.sum())
    diagnostics = dict(status="ok", threshold_db=None, candidate_prevalence=None,
                       input_valid_pixels=total, eligible_pixels=eligible,
                       excluded_stable_water_pixels=total-eligible,
                       low_delta_pixels=None, between_class_variance_fraction=None,
                       mode_left_db=None, mode_right_db=None, valley_ratio=None)

    def reject(reason):
        diagnostics["status"] = f"fallback_{reason}"
        return None, diagnostics

    if not total or eligible < 2 * RULES["min_class_count"] or np.count_nonzero(counts_eligible) < 2:
        return reject("insufficient_histogram_support")

    centers = (EDGES[:-1] + EDGES[1:]) / 2
    mass = counts_eligible / eligible
    cumulative = np.cumsum(mass)[:-1]
    moment = np.cumsum(mass * centers)
    denominator = cumulative * (1-cumulative)
    score = np.divide((moment[-1]*cumulative - moment[:-1])**2, denominator,
                      out=np.zeros_like(denominator), where=denominator > 1e-15)
    split = int(np.argmax(score))
    threshold = float(EDGES[split + 1])
    low = int(counts_eligible[:split+1].sum())
    candidate = low / total  # denominator is all input-valid pixels, including water
    variance = float(np.sum(mass*(centers-moment[-1])**2))
    diagnostics.update(threshold_db=threshold, candidate_prevalence=candidate, low_delta_pixels=low,
                       between_class_variance_fraction=float(score[split]/variance) if variance else None)
    if threshold >= 0:
        return reject("nonnegative_threshold")
    if (min(low, eligible-low) < RULES["min_class_count"]
            or not RULES["min_flood_fraction"] <= candidate <= RULES["max_flood_fraction"]):
        return reject("degenerate_prevalence")

    smoothed = gaussian_filter1d(mass, RULES["smooth_sigma_db"]/BIN_DB)
    peaks = find_peaks(smoothed, prominence=RULES["min_peak_prominence_fraction"]*smoothed.max())[0]
    left = peaks[peaks <= split]
    right = peaks[peaks > split]
    if not len(left) or not len(right):
        return reject("unimodal_or_no_distinct_valley")
    lo, hi = (int(group[np.argmax(smoothed[group])]) for group in (left, right))
    valley = float(smoothed[lo:hi+1].min() / min(smoothed[lo], smoothed[hi]))
    diagnostics.update(mode_left_db=float(centers[lo]), mode_right_db=float(centers[hi]), valley_ratio=valley)
    if ((hi-lo)*BIN_DB < RULES["min_peak_separation_db"]
            or valley > RULES["max_valley_to_smaller_peak"]):
        return reject("unimodal_or_no_distinct_valley")
    return candidate, diagnostics


def histogram_rows(histograms):
    for event in sorted(histograms, key=int):
        all_counts, eligible = histograms[event]
        for i in range(len(EDGES)-1):
            yield dict(event_id=event, bin_left_db=float(EDGES[i]), bin_right_db=float(EDGES[i+1]),
                       all_input_valid=int(all_counts[i]), otsu_eligible=int(eligible[i]),
                       excluded_stable_water=int(all_counts[i]-eligible[i]))


def plot_histograms(path, histograms, decisions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    events = sorted(histograms, key=int)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True, layout="constrained")
    limits = []
    for ax, event in zip(axes.flat, events, strict=True):
        all_counts, eligible = histograms[event]
        for counts, name, color in ((all_counts, "Input-valid", "#8897a3"),
                                    (eligible, "Otsu-eligible", "#255987")):
            ax.stairs(counts / all_counts.sum(), EDGES, label=name, color=color)
        cumulative = all_counts.cumsum() / all_counts.sum()
        limits.extend(EDGES[np.searchsorted(cumulative, [.005, .995])])
        row = decisions[event]["methods"]["otsu"]
        threshold = row["diagnostics"]["threshold_db"]
        if threshold is not None:
            ax.axvline(threshold, color="#af5724", linestyle="--", label=f"Otsu {threshold:+.2f} dB")
            limits.append(threshold)
        accepted = row["diagnostics"]["status"] == "ok"
        ax.set_title(f"{event} · {'Otsu applied' if accepted else 'Fallback b=0'}")
        ax.grid(axis="y", alpha=.2)
        ax.legend(fontsize=7)
    axes[0, 0].set_xlim(min(limits)-1, max(limits)+1)
    fig.supxlabel("Post VV − mean pre VV (dB)")
    fig.supylabel("Fraction of input-valid pixels per bin")
    fig.suptitle("Exploratory validation · Delta-VV histograms")
    fig.savefig(path, dpi=175)
    plt.close(fig)
