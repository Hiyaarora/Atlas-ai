"""document lifecycle and conversation isolation

Revision ID: 92d321f6109f
Revises: 4a5a599df5e8
Create Date: 2026-07-30 00:15:50.175974

HAND-WRITTEN. Autogenerate proposed dropping `documents.user_id` and
`documents.status` and adding `owner_id` and `ingestion_status` — which is
correct as a schema diff and catastrophic as a migration: it would discard
every owner and every ingestion state. Alembic cannot infer that two columns
are the same column renamed, so the renames below are explicit.

New columns are added nullable, backfilled, then constrained. Adding a NOT
NULL column with no default fails outright on a non-empty table, and adding
one with a server default silently rewrites history — this way the values
written are the ones intended.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "92d321f6109f"
down_revision: str | None = "4a5a599df5e8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ---- documents: renames ---------------------------------------------
    op.alter_column("documents", "user_id", new_column_name="owner_id")
    op.alter_column("documents", "status", new_column_name="ingestion_status")

    # The CHECK constraint references the old column name and does not follow
    # the rename, so it is replaced.
    #
    # Bare name, not the full one: `drop_constraint` runs the metadata naming
    # convention over whatever it is given, so passing the already-expanded
    # "ck_documents_valid_status" yields "ck_documents_ck_documents_valid_status".
    # Wrap in op.f() to opt out of the convention instead.
    op.drop_constraint("valid_status", "documents", type_="check")
    op.create_check_constraint(
        "valid_ingestion",
        "documents",
        "ingestion_status IN ('pending', 'processing', 'ready', 'failed')",
    )

    # ---- documents: lifecycle columns -----------------------------------
    op.add_column("documents", sa.Column("lifecycle_status", sa.String(20), nullable=True))
    op.add_column("documents", sa.Column("reference_count", sa.Integer(), nullable=True))
    op.add_column(
        "documents", sa.Column("last_accessed_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column("documents", sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column(
        "documents", sa.Column("deletion_scheduled_at", sa.DateTime(timezone=True), nullable=True)
    )

    # ---- conversations: isolation + activity ----------------------------
    op.add_column("conversations", sa.Column("document_id", sa.UUID(), nullable=True))
    op.add_column(
        "conversations", sa.Column("last_message_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column("conversations", sa.Column("is_archived", sa.Boolean(), nullable=True))

    op.create_foreign_key(
        op.f("fk_conversations_document_id_documents"),
        "conversations",
        "documents",
        ["document_id"],
        ["id"],
        # SET NULL, never CASCADE: purging a document must not destroy a
        # user's conversation history.
        ondelete="SET NULL",
    )

    # ---- backfill --------------------------------------------------------
    # Existing documents are active and have never been archived. Their
    # last-accessed time is unknowable, so creation time is the honest
    # conservative choice: it starts the inactivity clock at the earliest
    # defensible moment rather than pretending they were just used.
    op.execute("""
        UPDATE documents
        SET lifecycle_status = 'active',
            reference_count  = 0,
            last_accessed_at = created_at
        """)

    # Existing conversations keep working as general chat. There is no way to
    # infer retroactively which document they were about, and guessing would
    # be worse than leaving them unbound.
    op.execute("""
        UPDATE conversations
        SET last_message_at = COALESCE(
                (SELECT max(created_at) FROM messages
                  WHERE messages.conversation_id = conversations.id),
                updated_at
            ),
            is_archived = false
        """)

    # Reference counts computed from the links that now exist. Zero today,
    # since no conversation has a document yet — but written by the same
    # query the application uses, so the migration cannot disagree with it.
    op.execute("""
        UPDATE documents
        SET reference_count = (
            SELECT count(*) FROM conversations
             WHERE conversations.document_id = documents.id
        )
        """)

    # ---- constrain now that every row has a value -----------------------
    op.alter_column("documents", "lifecycle_status", nullable=False)
    op.alter_column("documents", "reference_count", nullable=False)
    op.alter_column(
        "documents", "last_accessed_at", nullable=False, server_default=sa.text("now()")
    )
    op.alter_column(
        "conversations", "last_message_at", nullable=False, server_default=sa.text("now()")
    )
    op.alter_column("conversations", "is_archived", nullable=False)

    op.create_check_constraint(
        "valid_lifecycle",
        "documents",
        "lifecycle_status IN ('active', 'archived', 'pending_deletion')",
    )
    op.create_check_constraint("reference_count_non_negative", "documents", "reference_count >= 0")

    # ---- indexes ---------------------------------------------------------
    op.drop_index("ix_documents_user_id_created_at", table_name="documents")
    op.create_index("ix_documents_owner_id_created_at", "documents", ["owner_id", "created_at"])
    op.create_index(
        "ix_documents_lifecycle_archived", "documents", ["lifecycle_status", "archived_at"]
    )
    op.create_index(
        "ix_documents_lifecycle_deletion",
        "documents",
        ["lifecycle_status", "deletion_scheduled_at"],
    )
    op.create_index("ix_conversations_document_id", "conversations", ["document_id"])
    op.create_index(
        "ix_conversations_archived_last_message",
        "conversations",
        ["is_archived", "last_message_at"],
    )


def downgrade() -> None:
    # Order is load-bearing: the old index references `user_id`, which does
    # not exist again until the rename below. Creating it first fails with
    # "column user_id does not exist" — caught by
    # test_migrations_can_be_rolled_all_the_way_back, which is exactly why
    # that test runs the downgrade rather than merely inspecting it.
    op.drop_index("ix_conversations_archived_last_message", table_name="conversations")
    op.drop_index("ix_conversations_document_id", table_name="conversations")
    op.drop_index("ix_documents_lifecycle_deletion", table_name="documents")
    op.drop_index("ix_documents_lifecycle_archived", table_name="documents")
    op.drop_index("ix_documents_owner_id_created_at", table_name="documents")

    op.drop_constraint("reference_count_non_negative", "documents", type_="check")
    op.drop_constraint("valid_lifecycle", "documents", type_="check")

    op.drop_constraint(
        op.f("fk_conversations_document_id_documents"), "conversations", type_="foreignkey"
    )
    op.drop_column("conversations", "is_archived")
    op.drop_column("conversations", "last_message_at")
    op.drop_column("conversations", "document_id")

    op.drop_column("documents", "deletion_scheduled_at")
    op.drop_column("documents", "archived_at")
    op.drop_column("documents", "last_accessed_at")
    op.drop_column("documents", "reference_count")
    op.drop_column("documents", "lifecycle_status")

    op.drop_constraint("valid_ingestion", "documents", type_="check")

    # Renames first; everything that names these columns comes after.
    op.alter_column("documents", "ingestion_status", new_column_name="status")
    op.alter_column("documents", "owner_id", new_column_name="user_id")

    op.create_check_constraint(
        "valid_status",
        "documents",
        "status IN ('pending', 'processing', 'ready', 'failed')",
    )
    op.create_index("ix_documents_user_id_created_at", "documents", ["user_id", "created_at"])
