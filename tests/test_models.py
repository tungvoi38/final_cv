import numpy as np
import pytest

from geoid.contracts import validate_model_output
from geoid.models import resnet_unet, terramind


def test_output_boundary():
    output = {
        "logits": np.ones((2, 3, 4, 4), dtype=np.float32),
        "features": {"alignment": np.ones((2, 8, 2, 2), dtype=np.float32)},
    }
    validate_model_output(output, batch_size=2, height=4, width=4)
    output["logits"] = output["logits"][:, :2]
    with pytest.raises(ValueError, match="shape"):
        validate_model_output(output, batch_size=2, height=4, width=4)


@pytest.mark.parametrize("module", [resnet_unet, terramind])
def test_unimplemented_models_fail_explicitly(module):
    with pytest.raises(NotImplementedError):
        module.build_model({})
