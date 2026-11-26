"""Salesforce ERM_SMA recurrence and epoch-boundary training snapshots."""

import copy
import csv
import json
import math
import os
import random
from collections import defaultdict

import numpy as np
import torch

SMA_START_STEP = 100
SMA_POLICY = "salesforce_step_100_includes_step_99"


class SimpleMovingAverage:
    def __init__(self, online):
        self.model = copy.deepcopy(online).requires_grad_(False).eval()
        self.steps = 0
        self.count = 0
        self._groups = defaultdict(lambda: ([], []))
        online_state = online.state_dict()
        for name, target in self.model.state_dict().items():
            if name.rsplit(".", 1)[-1] == "num_batches_tracked":
                continue
            targets, sources = self._groups[target.device, target.dtype]
            targets.append(target)
            sources.append(online_state[name])

    @torch.no_grad()
    def update(self):
        """Call once, after a successful optimizer step."""
        self.steps += 1
        if self.steps >= SMA_START_STEP:
            self.count += 1
        for targets, sources in self._groups.values():
            if self.count and targets[0].is_floating_point():
                # The pre-threshold copy contributes one sample to this mean.
                torch._foreach_lerp_(targets, sources, 1 / (self.count + 1))
            else:
                torch._foreach_copy_(targets, sources)

    def state_dict(self):
        return {"model": self.model.state_dict(), "steps": self.steps, "count": self.count}

    def load_state_dict(self, state):
        steps, count = state["steps"], state["count"]
        if steps < 0 or count != max(0, steps - SMA_START_STEP + 1):
            raise ValueError("Invalid SMA step or averaging count")
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.steps, self.count = steps, count


def capture_state(optimizer, scheduler, scaler, generators):
    numpy_state = np.random.get_state()
    return {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": {
            "python": random.getstate(),
            "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "generators": {name: generator.get_state() for name, generator in generators.items()},
        },
    }


def restore_state(state, optimizer, scheduler, scaler, generators):
    rng = state["rng"]
    if set(rng["generators"]) != set(generators):
        raise ValueError("Resume requires the original data generator layout")
    if len(rng["cuda"]) != torch.cuda.device_count():
        raise ValueError("Resume requires the original number of visible CUDA devices")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    random.setstate(rng["python"])
    name, keys, *rest = rng["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), *rest))
    torch.set_rng_state(rng["torch"])
    if rng["cuda"]:
        torch.cuda.set_rng_state_all(rng["cuda"])
    for name, generator in generators.items():
        generator.set_state(rng["generators"][name])


def save_epoch(path, checkpoint):
    temporary = path.with_suffix(".partial")
    temporary.unlink(missing_ok=True)
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def save_csv(path, rows):
    temporary = path.with_suffix(".partial")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def link_best(run, epoch, name):
    path = run / name
    temporary = path.with_suffix(".partial")
    temporary.unlink(missing_ok=True)
    source = run / "checkpoints" / f"epoch_{epoch:03}.pt"
    if path.is_file() and path.samefile(source):
        return
    os.link(source, temporary)
    os.replace(temporary, path)


def selected_epoch(rows, *, require_sma=False):
    eligible = [
        row
        for row in rows
        if math.isfinite(float(row["pooled_miou"]))
        and (not require_sma or int(row["sma_count"]) > 0)
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda row: (float(row["pooled_miou"]), -int(row["epoch"])))


def publish_epoch(run, journal):
    for filename, rows in journal.items():
        save_csv(run / filename, rows)
    online = selected_epoch(journal["online_history.csv"])
    averaged = selected_epoch(journal["history.csv"], require_sma=True)
    if online is not None:
        link_best(run, online["epoch"], "best_online.pt")
    if averaged is not None:
        link_best(run, averaged["epoch"], "best.pt")
    return online, averaged


def prune_epoch_checkpoints(run, keep=()):
    """Keep only epochs needed to resume and republish the current selections."""
    names = {f"epoch_{epoch:03}.pt" for epoch in keep}
    for path in (run / "checkpoints").glob("epoch_*.pt"):
        if path.name not in names:
            path.unlink()


def compact_selected(run):
    """Publish portable best weights before the completion marker."""
    for filename, weight_key, metric_key in (
        ("best.pt", "model", "validation"),
        ("best_online.pt", "online_model", "online_validation"),
    ):
        path = run / filename
        checkpoint = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
        kept = {key: checkpoint[key] for key in ("config", "epoch", weight_key, metric_key)}
        if weight_key == "model":
            kept["sma"] = {"count": checkpoint["sma"]["count"]}
        save_epoch(path, kept)


def train_sma(
    run,
    config,
    online,
    train_batches,
    validation,
    optimizer,
    scheduler,
    scaler,
    generators,
    mlflow,
    *,
    resume,
):
    """Use the trainer callbacks for ERM updates and unchanged validation metrics."""
    import fcntl

    with (run / ".training.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if resume and (run / "summary.json").is_file():
            raise ValueError("Cannot resume a completed run")
        _train_locked(
            run,
            config,
            online,
            train_batches,
            validation,
            optimizer,
            scheduler,
            scaler,
            generators,
            mlflow,
            resume=resume,
        )


def _train_locked(
    run,
    config,
    online,
    train_batches,
    validation,
    optimizer,
    scheduler,
    scaler,
    generators,
    mlflow,
    *,
    resume,
):
    sma = SimpleMovingAverage(online)
    fixed = config.get("selection") == "fixed_epoch"
    if fixed != (validation is None):
        raise ValueError("Fixed-epoch training requires no validation callback")
    compact = config.get("compact_checkpoints", False)
    journal = {
        name: []
        for name in (
            "history.csv",
            "validation_events.csv",
            "online_history.csv",
            "online_validation_events.csv",
        )
    }
    if fixed:
        journal = {"history.csv": []}
    start = 1
    mlflow_id = None
    if resume:
        saved = sorted((run / "checkpoints").glob("epoch_*.pt"), key=lambda p: int(p.stem[6:]))
        if not saved:
            raise FileNotFoundError("Resume requires at least one completed epoch checkpoint")
        checkpoint = torch.load(saved[-1], weights_only=True, map_location="cpu", mmap=True)
        if checkpoint["config"] != config:
            raise ValueError("Resume requires the same run configuration and source hashes")
        if checkpoint["epoch"] != int(saved[-1].stem[6:]):
            raise ValueError("Checkpoint epoch disagrees with its filename")
        online.load_state_dict(checkpoint["online_model"])
        sma.load_state_dict(checkpoint["sma"])
        restore_state(checkpoint["training_state"], optimizer, scheduler, scaler, generators)
        start = checkpoint["epoch"] + 1
        journal = checkpoint["journal"]
        mlflow_id = (run / "mlflow_run_id.txt").read_text().strip()
        if fixed:
            save_csv(run / "history.csv", journal["history.csv"])
        else:
            publish_epoch(run, journal)
        del checkpoint

    with mlflow.start_run(run_id=mlflow_id, run_name=config["run_id"]) as active:
        (run / "mlflow_run_id.txt").write_text(active.info.run_id)
        mlflow.log_params({key: str(val) for key, val in config.items() if key != "raw"})
        for name in ("config.json", "augmentation.json", "meta/preprocess_config.json"):
            mlflow.log_artifact(str(run / name))
        if (run / "split_manifest.json").is_file():
            mlflow.log_artifact(str(run / "split_manifest.json"))
        if (run / "train_sample_ids.json").is_file():
            mlflow.log_artifact(str(run / "train_sample_ids.json"))

        def log_metrics(index):
            averaged = journal["history.csv"][index]
            if fixed:
                mlflow.log_metrics({key:float(val) for key,val in averaged.items() if key != "epoch"},
                                   step=averaged["epoch"])
                return
            online_row = journal["online_history.csv"][index]
            values = {key: float(val) for key, val in averaged.items() if key != "epoch"}
            values.update(
                {f"online_{key}": float(val) for key, val in online_row.items() if key != "epoch"}
            )
            mlflow.log_metrics(values, step=averaged["epoch"])

        # Recover a metric record if the previous process stopped after saving the epoch.
        if resume:
            log_metrics(-1)
        for epoch in range(start, config["epochs"] + 1):
            train_loss = train_batches(epoch, sma)
            if fixed:
                journal["history.csv"].append(dict(epoch=epoch,train_loss=train_loss,
                    sma_steps=sma.steps,sma_count=sma.count))
                state = sma.state_dict()
                save_epoch(run/"checkpoints"/f"epoch_{epoch:03}.pt", dict(
                    epoch=epoch,config=config,online_model=online.state_dict(),sma=state,
                    training_state=capture_state(optimizer,scheduler,scaler,generators),journal=journal))
                save_csv(run/"history.csv",journal["history.csv"])
                prune_epoch_checkpoints(run,(epoch,))
                log_metrics(-1)
                print(f"epoch={epoch} steps={sma.steps} count={sma.count}",flush=True)
                continue
            online_metrics, online_events = validation(online)
            # Extra validation must leave the online loader RNG at the ERM position.
            generator = generators["validation_loader"]
            generator_state = generator.get_state()
            try:
                averaged_metrics, averaged_events = validation(sma.model)
            finally:
                generator.set_state(generator_state)
            journal["history.csv"].append(
                dict(
                    epoch=epoch,
                    train_loss=train_loss,
                    sma_steps=sma.steps,
                    sma_count=sma.count,
                    **averaged_metrics,
                )
            )
            journal["online_history.csv"].append(
                dict(
                    epoch=epoch,
                    train_loss=train_loss,
                    **online_metrics,
                )
            )
            for name, events in (
                ("validation_events.csv", averaged_events),
                ("online_validation_events.csv", online_events),
            ):
                journal[name].extend(
                    dict(epoch=epoch, event_id=event, **events[event]) for event in sorted(events)
                )
            average_state = sma.state_dict()
            checkpoint = {
                "epoch": epoch,
                "config": config,
                "model": average_state["model"],
                "online_model": online.state_dict(),
                "sma": average_state,
                "validation": averaged_metrics,
                "online_validation": online_metrics,
                "training_state": capture_state(optimizer, scheduler, scaler, generators),
                "journal": journal,
            }
            save_epoch(run / "checkpoints" / f"epoch_{epoch:03}.pt", checkpoint)
            online_best, best = publish_epoch(run, journal)
            if compact:
                keep = (epoch, online_best["epoch"])
                if best is not None:
                    keep += (best["epoch"],)
                prune_epoch_checkpoints(run, keep)
            log_metrics(-1)
            print(
                f"epoch={epoch} val online mIoU={online_metrics['pooled_miou']:.4f} "
                f"SMA mIoU={averaged_metrics['pooled_miou']:.4f} "
                f"steps={sma.steps} count={sma.count} "
                f"best SMA epoch={best['epoch'] if best else None}",
                flush=True,
            )

        if fixed:
            if sma.count == 0 or len(journal["history.csv"]) != config["epochs"]:
                raise RuntimeError("Incomplete fixed-epoch SMA state")
            state = sma.state_dict()
            save_epoch(run/"fixed_sma.pt",dict(epoch=config["epochs"],config=config,model=state["model"],
                                             sma=dict(steps=sma.steps,count=sma.count)))
            for name in ("history.csv","fixed_sma.pt"):
                mlflow.log_artifact(str(run/name))
            summary = dict(selection="fixed_epoch",completed_epochs=config["epochs"],checkpoint_epoch=config["epochs"],
                           sma_steps=sma.steps,sma_count=sma.count)
            temporary = run/"summary.partial"
            temporary.write_text(json.dumps(summary,indent=2)+"\n")
            os.replace(temporary,run/"summary.json")
            prune_epoch_checkpoints(run)
            mlflow.log_artifact(str(run/"summary.json"))
            return

        online_best, best = publish_epoch(run, journal)
        if best is None:
            raise RuntimeError(f"SMA run ended before successful optimizer step {SMA_START_STEP}")
        summary = {
            "best_epoch": best["epoch"],
            "selection": "pooled_miou",
            "best_validation_score": best["pooled_miou"],
            "completed_epochs": config["epochs"],
            "best_online_epoch": online_best["epoch"],
            "best_online_validation_score": online_best["pooled_miou"],
            "sma_steps": sma.steps,
            "sma_count": sma.count,
        }
        if compact:
            compact_selected(run)
        for name in (tuple(journal) if compact else (*journal, "best.pt", "best_online.pt")):
            mlflow.log_artifact(str(run / name))
        temporary = run / "summary.partial"
        temporary.write_text(json.dumps(summary, indent=2) + "\n")
        os.replace(temporary, run / "summary.json")
        if compact:
            prune_epoch_checkpoints(run)
        mlflow.log_artifact(str(run / "summary.json"))
