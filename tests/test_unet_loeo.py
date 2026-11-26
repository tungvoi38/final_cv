"""LOEO safety and resume smoke tests without a full model run."""
from contextlib import nullcontext
from importlib.util import find_spec
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


@unittest.skipUnless(
    all(find_spec(dependency) for dependency in ("torch", "pyarrow", "rasterio", "matplotlib")),
    "requires the U-Net and Kuro Siwo dependencies",
)
class LoeoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        with patch.object(sys, "path", [str(scripts), *sys.path]):
            import loeo_splits
            import unet_sma
            import run_unet_loeo

        cls.splits = loeo_splits
        cls.sma = unet_sma
        cls.runner = run_unet_loeo

    def test_manifest_purges_overlap_and_excludes_original_test(self):
        import numpy as np
        import pandas as pd

        rows = []
        for event in range(43):
            split = "train" if event < 27 else "validation" if event < 33 else "test"
            for suffix, left in enumerate((event * 50_000, event * 50_000 + 3000)):
                rows.append({"actid": event, "tortilla:data_split": split, "tortilla:id": f"{event}-{suffix}",
                             "stac:geotransform": [left, 10, 0, 0, 0, -10],
                             "stac:crs": "EPSG:3857", "stac:raster_shape": [224, 224]})
        rows[2]["stac:geotransform"] = [0, 10, 0, 0, 0, -10]  # source patch intersects target 0
        rows = pd.DataFrame(rows)
        extra = pd.concat([rows.iloc[[4]]] * (5000 - 66), ignore_index=True)
        extra["tortilla:id"] = [f"extra-{i}" for i in range(len(extra))]
        frame = pd.concat([rows, extra], ignore_index=True)
        dev, events = self.splits.development(frame)
        neighbors = self.splits.footprint_neighbors(dev)
        fold = self.splits.make_fold(dev, events, neighbors, 0)
        self.assertEqual(fold["sample_ids"]["target"], ["0-0", "0-1"])
        self.assertEqual(fold["sample_ids"]["excluded_target_overlap"], ["1-0"])
        self.assertEqual(len(fold["source_train_events"]), 26)
        self.assertEqual(len(fold["inner_validation_events"]), 6)
        self.assertTrue(np.array_equal(fold["original_test_events_excluded"], range(33, 43)))
        self.assertNotIn("1-0", fold["sample_ids"]["source_train"])
        replacement = self.splits.make_fold(dev, events, neighbors, 27)
        self.assertEqual(len(set(replacement["inner_validation_events"]) & events["train"]), 1)
        self.assertNotIn(27, replacement["inner_validation_events"])
        with self.assertRaisesRegex(ValueError, "not a development event"):
            self.splits.make_fold(dev, events, neighbors, 33)
        with tempfile.TemporaryDirectory() as tmp, patch.object(self.runner, "KuroDataset") as dataset:
            with self.assertRaises(FileNotFoundError):
                self.runner.evaluate_target(None, dev, Path(tmp), fold, {})
            dataset.assert_not_called()

    def test_compact_run_resumes_and_preserves_both_best_epochs(self):
        import torch

        torch.manual_seed(23)

        def setup():
            model = torch.nn.Linear(1, 1, bias=False)
            model.weight.data.zero_()
            opt = torch.optim.AdamW(model.parameters(), lr=.001)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4)
            scaler = torch.amp.GradScaler("cuda", enabled=False)
            return model, opt, scheduler, scaler, {"validation_loader": torch.Generator().manual_seed(9)}

        def launch(run, resume, interrupt):
            model, opt, scheduler, scaler, generators = setup()
            config = {"run_id": "test", "epochs": 4, "compact_checkpoints": True}
            mlflow = SimpleNamespace(start_run=lambda **_: nullcontext(SimpleNamespace(info=SimpleNamespace(run_id="id"))),
                                     log_params=lambda _: None, log_artifact=lambda _: None,
                                     log_metrics=lambda *_, **__: None)

            def train(epoch, average):
                if epoch == interrupt:
                    raise RuntimeError("simulated stop")
                opt.zero_grad(set_to_none=True)
                (model(torch.ones(1, 1)).sum() - 1).square().backward()
                opt.step()
                scheduler.step()
                with torch.no_grad():
                    model.weight.fill_(epoch if epoch < 3 else -epoch)
                for _ in range(100):
                    average.update()
                return float(torch.rand(()))

            def validate(current):
                return {"pooled_miou": current.weight.item()}, {"event": {"flood_iou": .1}}

            self.sma.train_sma(run, config, model, train, validate, opt, scheduler, scaler,
                               generators, mlflow, resume=resume)

        with tempfile.TemporaryDirectory() as tmp:
            full, resumed = [Path(tmp) / kind for kind in ("full", "resumed")]
            for run in (full, resumed):
                (run / "checkpoints").mkdir(parents=True)
            torch.manual_seed(23)
            launch(full, False, None)
            torch.manual_seed(23)
            with self.assertRaisesRegex(RuntimeError, "simulated stop"):
                launch(resumed, False, 4)
            self.assertLessEqual(len(list((resumed / "checkpoints").glob("*.pt"))), 3)
            launch(resumed, True, None)
            for name, key in (("best_online.pt", "online_model"), ("best.pt", "model")):
                f = torch.load(full / name, weights_only=True)
                r = torch.load(resumed / name, weights_only=True)
                self.assertEqual(f["epoch"], r["epoch"])
                torch.testing.assert_close(f[key]["weight"], r[key]["weight"], rtol=0, atol=0)
                self.assertNotIn("training_state", r)
            self.assertEqual((full / "online_history.csv").read_text(), (resumed / "online_history.csv").read_text())
            self.assertFalse(list((resumed / "checkpoints").glob("*.pt")))
