import numpy as np
import pytest

from geoid.contracts import valid_target_pixels, validate_sample


def test_sample_contract(sample):
    validate_sample(sample)
    assert valid_target_pixels(sample["mask"]).sum() == 12


@pytest.mark.parametrize("change", ["dtype", "shape", "label", "nan", "metadata"])
def test_reject_invalid_sample(sample, change):
    if change == "dtype":
        sample["mask"] = sample["mask"].astype(np.int32)
    elif change == "shape":
        sample["sar"] = sample["sar"][:3]
    elif change == "label":
        sample["mask"][0, 0] = 3
    elif change == "nan":
        sample["optical"][0, 0, 0] = np.nan
    else:
        sample["event_id"] = ""
    with pytest.raises(ValueError):
        validate_sample(sample)


def test_all_ignore_is_empty_selection(sample):
    sample["mask"][:] = 255
    assert not valid_target_pixels(sample["mask"]).any()
