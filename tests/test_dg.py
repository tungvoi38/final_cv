import numpy as np
import pytest

from geoid.contracts import validate_dg_inputs, validate_dg_output
from geoid.dg import class_coral, coral


def test_dg_boundary():
    features = np.ones((2, 8, 2, 2), dtype=np.float32)
    masks = np.ones((2, 2, 2), dtype=np.int64)
    validate_dg_inputs(features, masks, ["a", "b"], split="train")
    for events, split in [(["a", "a"], "train"), (["a", "b"], "test")]:
        with pytest.raises(ValueError):
            validate_dg_inputs(features, masks, events, split=split)
    validate_dg_output({"loss": np.array(1.0, dtype=np.float32), "metrics": {"pairs": 1}})
    with pytest.raises(ValueError):
        validate_dg_output({"loss": np.array(np.nan, dtype=np.float32), "metrics": {}})


@pytest.mark.parametrize("module", [coral, class_coral])
def test_dg_not_implemented(module):
    with pytest.raises(NotImplementedError):
        module.regularize(None, None, None)
