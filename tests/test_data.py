import io

import pytest

from geoid.data import download
from geoid.data.manifest import safe_relative, validate_manifest, verify_manifest_files


def test_manifest(manifest):
    validate_manifest(manifest)


@pytest.mark.parametrize("key", ["event_id", "aoi_id", "parent_tile_id"])
def test_split_leakage(manifest, key):
    manifest["records"][1][key] = manifest["records"][0][key]
    with pytest.raises(ValueError, match="crosses splits"):
        validate_manifest(manifest)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "checksum", "size", "split"])
def test_invalid_manifest(manifest, mutation):
    record = manifest["records"][0]
    if mutation == "duplicate":
        manifest["records"].append(record)
    elif mutation == "missing":
        del record["files"]["mask"]
    elif mutation == "checksum":
        record["files"]["mask"]["sha256"] = "bad"
    elif mutation == "size":
        record["files"]["mask"]["size_bytes"] = -1
    else:
        record["split"] = "validation-ish"
    with pytest.raises(ValueError):
        validate_manifest(manifest)


@pytest.mark.parametrize("path", ["../secret", "/tmp/x", "C:/x", "C:x", "a\\b", "x/../z", "x:y"])
def test_path_traversal(path):
    with pytest.raises(ValueError):
        safe_relative(path)


def test_manifest_file_verification(manifest, tmp_path):
    for record in manifest["records"]:
        for item in record["files"].values():
            path = tmp_path / item["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"abc")
    assert verify_manifest_files(manifest, tmp_path) == []
    item = manifest["records"][0]["files"]["mask"]
    item["sha256"] = None
    assert verify_manifest_files(manifest, tmp_path) == [item["path"]]
    (tmp_path / item["path"]).write_bytes(b"wrong length")
    with pytest.raises(ValueError, match="wrong-size"):
        verify_manifest_files(manifest, tmp_path)


def test_dry_run_never_opens_network(acquisition, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("dry run accessed network")

    monkeypatch.setattr(download, "urlopen", forbidden)
    destination = tmp_path / "absent"
    assert download.acquire(acquisition, output=destination)["status"] == "dry-run"
    assert not destination.exists()


def test_unconfigured_and_full_fail_closed(acquisition, tmp_path):
    with pytest.raises(ValueError, match="immutable"):
        download.acquire({}, output=tmp_path)
    with pytest.raises(ValueError, match="confirm-full"):
        download.acquire(acquisition, output=tmp_path, mode="full", execute=True)
    acquisition["acquisition"]["sample_max_bytes"] = 2
    with pytest.raises(ValueError, match="byte cap"):
        download.acquire(acquisition, output=tmp_path)


@pytest.mark.parametrize(
    "url",
    [
        "http://huggingface.co/x",
        "https://example.com/x",
        "https://huggingface.co/datasets/other/resolve/main/x",
    ],
)
def test_reject_unofficial_download(acquisition, url):
    acquisition["acquisition"]["files"][0]["url"] = url
    with pytest.raises(ValueError, match="official"):
        download.plan_acquisition(acquisition)


def test_transfer_verification_and_overwrite(acquisition, tmp_path, monkeypatch):
    item = acquisition["acquisition"]["files"][0]

    def response(*args, **kwargs):
        stream = io.BytesIO(b"abc")
        stream.url = item["url"]
        return stream

    monkeypatch.setattr(download, "urlopen", response)
    download.acquire(acquisition, output=tmp_path, execute=True)
    assert (tmp_path / "sample.bin").read_bytes() == b"abc"
    assert download.acquire(acquisition, output=tmp_path, verify_only=True)["status"] == "verified"
    with pytest.raises(ValueError, match="overwrite"):
        download.acquire(acquisition, output=tmp_path, execute=True)
    item["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        download.acquire(acquisition, output=tmp_path, execute=True, overwrite=True)
    assert (tmp_path / "sample.bin").read_bytes() == b"abc"
    assert not list(tmp_path.glob("*.part"))
