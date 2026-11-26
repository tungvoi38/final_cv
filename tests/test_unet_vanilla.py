"""Check exact AP and event metrics for the validation-only Vanilla ERM path."""
from importlib.util import find_spec
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


@unittest.skipUnless(
    all(find_spec(dependency) for dependency in ("torch", "pyarrow", "rasterio", "matplotlib")),
    "requires the U-Net evaluation dependencies",
)
class VanillaValidationTest(unittest.TestCase):
    def test_tied_ap_and_ignored_pixels(self):
        import torch

        scripts = Path(__file__).resolve().parents[1] / "scripts"
        with patch.object(sys, "path", [str(scripts), *sys.path]):
            import evaluate_unet_vanilla as vanilla

        # Two batches tie at margin zero. An ignored pixel has the highest score.
        logits = [torch.tensor([[[[1., 1., 0.]], [[0., 0., 0.]], [[3., 1., 99.]]]]),
                  torch.tensor([[[[1., 2.]], [[0., 0.]], [[1., 1.]]]])]
        targets = [torch.tensor([[[2, 0, 255]]]), torch.tensor([[[2, 0]]])]
        batches = [dict(event_id=["411"], x=logits[0], y=targets[0]),
                   dict(event_id=["445"], x=logits[1], y=targets[1])]
        with (
            patch.object(vanilla, "move", lambda batch, device: (batch["x"], batch["y"])),
            patch.object(vanilla, "pre_variability_inputs", lambda x: (x, None)),
        ):
            metrics, events = vanilla.evaluate(torch.nn.Identity(), batches, torch.device("cpu"))
        expected = dict(pooled_flood_ap=0.5 + 0.5 * 2 / 3, pooled_flood_iou=0.5,
                        pooled_flood_precision=1, pooled_flood_recall=0.5, pooled_flood_f1=2 / 3,
                        pooled_miou=(2 / 3 + 0 + 0.5) / 3, event_macro_flood_iou=0.5,
                        event_flood_iou_std=0.5)
        for key, value in expected.items():
            self.assertAlmostEqual(metrics[key], value, msg=key)
        self.assertEqual(events["411"]["flood_tp"], 1)
        self.assertEqual(events["445"]["flood_fn"], 1)
