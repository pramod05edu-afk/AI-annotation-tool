"""model task 'detect_open' for the general open-vocabulary detector

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-27 18:30:00

The task column is a VARCHAR with a CHECK constraint (named "modeltask") listing the allowed
values; a new value means replacing that constraint.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_OLD = ("detect", "segment", "embed")
_NEW = ("detect", "detect_open", "segment", "embed")


def _check(values: tuple[str, ...]) -> str:
    return "task IN (" + ", ".join(f"'{v}'" for v in values) + ")"


def upgrade() -> None:
    op.drop_constraint("modeltask", "model_versions", type_="check")
    op.create_check_constraint("modeltask", "model_versions", _check(_NEW))


def downgrade() -> None:
    op.execute("DELETE FROM model_versions WHERE task = 'detect_open'")
    op.drop_constraint("modeltask", "model_versions", type_="check")
    op.create_check_constraint("modeltask", "model_versions", _check(_OLD))
