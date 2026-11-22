"""The frontend's generated API types must match the real API."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from export_openapi import SCHEMA_FILE, render_schema  # noqa: E402


def test_committed_openapi_schema_is_up_to_date() -> None:
    assert SCHEMA_FILE.exists(), "run: uv run python scripts/export_openapi.py"
    assert SCHEMA_FILE.read_text(encoding="utf-8") == render_schema(), (
        "API changed: run `uv run python scripts/export_openapi.py` (backend) "
        "and `npm run gen:api` (frontend)"
    )
