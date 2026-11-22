"""Write the API's OpenAPI schema to frontend/openapi.json.

The frontend generates its TypeScript types from this file (`npm run gen:api`), so the
browser code and the Python API cannot silently drift apart: change a schema here, re-run
this script and the TypeScript compiler shows every place in the UI that must change.
A test (tests/test_openapi.py) fails if the committed file is out of date.

Usage (from the backend/ folder):
    uv run python scripts/export_openapi.py
"""

from __future__ import annotations

import json
from pathlib import Path

from app.main import app

SCHEMA_FILE = Path(__file__).resolve().parents[2] / "frontend" / "openapi.json"


def render_schema() -> str:
    return json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    SCHEMA_FILE.write_text(render_schema(), encoding="utf-8")
    print(f"Wrote {SCHEMA_FILE}")
