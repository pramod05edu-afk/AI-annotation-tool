"""Download model weights on first start, so `docker compose up` works on a fresh clone.

Model files are too big for git, so the fine-tuned detector is published as a GitHub Release
asset (a zip). On start-up, if the model is missing, the zip is downloaded, its SHA-256 hash
is checked (a corrupted or tampered file is rejected), and it is extracted into weights/.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)
CHUNK = 1024 * 1024


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive: zipfile.ZipFile, target_dir: Path) -> None:
    """Extract, refusing entries that would land outside target_dir ("zip slip")."""
    root = target_dir.resolve()
    for member in archive.infolist():
        destination = (root / member.filename).resolve()
        if not destination.is_relative_to(root):
            raise ValueError(f"Unsafe path in archive: {member.filename}")
    archive.extractall(root)


def ensure_model(
    target: Path, url: str | None, sha256: str | None = None, timeout: int = 60
) -> None:
    """Make sure `target` exists, downloading and extracting `url` (a zip) if needed."""
    if target.exists() or not url:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.parent / f".{target.name}.download"
    logger.info("Downloading %s ...", url)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response, partial.open("wb") as f:  # noqa: S310 - URL from our own settings
            shutil.copyfileobj(response, f, CHUNK)
        if sha256 and (actual := sha256_of(partial)) != sha256.lower():
            raise ValueError(f"Checksum mismatch for {url}: expected {sha256}, got {actual}")
        with zipfile.ZipFile(partial) as archive:
            safe_extract(archive, target.parent)
    finally:
        partial.unlink(missing_ok=True)
    if not target.exists():
        raise FileNotFoundError(f"{url} did not contain {target.name}")
    logger.info("Model ready: %s", target)
