"""project annotation mode

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-24 15:57:24.736310
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "projects",
        sa.Column(
            "annotation_mode",
            sa.Enum(
                "manual",
                "assisted",
                name="annotationmode",
                native_enum=False,
                create_constraint=True,
                length=20,
            ),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("projects", "annotation_mode")
