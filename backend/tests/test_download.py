"""Model download on first start: checksum, safe extraction, no work when already present."""

import hashlib
import zipfile
from pathlib import Path

import pytest

from app.ml.download import ensure_model


def make_zip(path: Path, entries: dict[str, bytes]) -> str:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_downloads_checks_and_extracts(tmp_path: Path) -> None:
    archive = tmp_path / "release.zip"
    digest = make_zip(archive, {"model_dir/best.xml": b"<xml/>", "metrics.json": b"{}"})
    weights = tmp_path / "weights"
    ensure_model(weights / "model_dir", archive.as_uri(), digest)
    assert (weights / "model_dir" / "best.xml").read_bytes() == b"<xml/>"
    assert (weights / "metrics.json").exists()
    assert not list(weights.glob(".*download"))  # temporary file removed


def test_wrong_checksum_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "release.zip"
    make_zip(archive, {"model_dir/best.xml": b"tampered"})
    with pytest.raises(ValueError, match="Checksum mismatch"):
        ensure_model(tmp_path / "weights" / "model_dir", archive.as_uri(), "0" * 64)
    assert not (tmp_path / "weights" / "model_dir").exists()


def test_zip_slip_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "evil.zip"
    digest = make_zip(archive, {"../outside.txt": b"x"})
    with pytest.raises(ValueError, match="Unsafe path"):
        ensure_model(tmp_path / "weights" / "model_dir", archive.as_uri(), digest)
    assert not (tmp_path / "outside.txt").exists()


def test_existing_model_is_not_downloaded_again(tmp_path: Path) -> None:
    target = tmp_path / "model_dir"
    target.mkdir()
    ensure_model(target, "https://example.invalid/never-called.zip")  # would fail if called


def test_archive_without_the_model_is_an_error(tmp_path: Path) -> None:
    archive = tmp_path / "release.zip"
    digest = make_zip(archive, {"something_else.txt": b"x"})
    with pytest.raises(FileNotFoundError):
        ensure_model(tmp_path / "weights" / "model_dir", archive.as_uri(), digest)
