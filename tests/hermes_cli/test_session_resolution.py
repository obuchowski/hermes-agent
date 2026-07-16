import pytest

from hermes_state import SessionDB
from unittest.mock import MagicMock


def test_named_profile_resolves_matching_gateway_owned_session(tmp_path):
    local_db = SessionDB(db_path=tmp_path / "profiles" / "programmer" / "state.db")
    root_db = SessionDB(db_path=tmp_path / "state.db")
    try:
        root_db.create_session(
            "gateway-session",
            "discord",
            profile_name="programmer",
            cwd="/projects/hermes",
        )
        root_db.append_message("gateway-session", role="user", content="from Discord")

        from hermes_cli.session_resolution import resolve_profile_session

        resolved = resolve_profile_session(
            "gateway-session",
            active_profile="programmer",
            local_db=local_db,
            root_db=root_db,
        )

        assert resolved is not None
        assert resolved.session_id == "gateway-session"
        assert resolved.profile_name == "programmer"
        assert resolved.storage_scope == "gateway_root"
        assert resolved.db is root_db
    finally:
        local_db.close()
        root_db.close()


def test_named_profile_cannot_resolve_another_profiles_gateway_session(tmp_path):
    local_db = SessionDB(db_path=tmp_path / "profiles" / "programmer" / "state.db")
    root_db = SessionDB(db_path=tmp_path / "state.db")
    try:
        root_db.create_session("ula-session", "telegram", profile_name="ula")

        from hermes_cli.session_resolution import resolve_profile_session

        assert resolve_profile_session(
            "ula-session",
            active_profile="programmer",
            local_db=local_db,
            root_db=root_db,
        ) is None
    finally:
        local_db.close()
        root_db.close()


def test_named_profile_local_db_rejects_explicit_foreign_profile_row(tmp_path):
    local_db = SessionDB(db_path=tmp_path / "profiles" / "programmer" / "state.db")
    root_db = SessionDB(db_path=tmp_path / "state.db")
    try:
        local_db.create_session("foreign", "cli", profile_name="other")
        local_db.create_session("legacy", "cli")

        from hermes_cli.session_resolution import resolve_profile_session

        assert resolve_profile_session(
            "foreign",
            active_profile="programmer",
            local_db=local_db,
            root_db=root_db,
        ) is None
        assert resolve_profile_session(
            "legacy",
            active_profile="programmer",
            local_db=local_db,
            root_db=root_db,
        ) is not None
    finally:
        local_db.close()
        root_db.close()


def test_named_profile_listing_excludes_explicit_foreign_local_row(tmp_path):
    local_db = SessionDB(db_path=tmp_path / "profiles" / "programmer" / "state.db")
    root_db = SessionDB(db_path=tmp_path / "state.db")
    try:
        local_db.create_session("foreign", "cli", profile_name="other")
        local_db.create_session("owned", "cli", profile_name="programmer")
        local_db.create_session("legacy", "cli")

        from hermes_cli.session_resolution import merge_profile_session_rows

        rows = merge_profile_session_rows(
            active_profile="programmer",
            local_db=local_db,
            root_db=root_db,
            load_rows=lambda db: [
                row
                for session_id in ("foreign", "owned", "legacy")
                if (row := db.get_session(session_id)) is not None
            ],
        )

        assert {row["id"] for row in rows} == {"owned", "legacy"}
    finally:
        local_db.close()
        root_db.close()


def test_completed_local_handoff_prefers_root_and_listing_deduplicates(tmp_path):
    local_db = SessionDB(db_path=tmp_path / "profiles" / "programmer" / "state.db")
    root_db = SessionDB(db_path=tmp_path / "state.db")
    try:
        local_db.create_session("shared", "cli", profile_name="programmer")
        local_db.request_handoff("shared", "discord", profile_name="programmer")
        local_db.claim_handoff("shared")
        local_db.mark_handoff_transferred("shared")
        local_db.complete_handoff("shared")
        root_db.create_session("shared", "discord", profile_name="programmer")
        root_db.append_message("shared", role="user", content="root continuation")

        from hermes_cli.session_resolution import (
            merge_profile_session_rows,
            resolve_profile_session,
        )

        resolved = resolve_profile_session(
            "shared",
            active_profile="programmer",
            local_db=local_db,
            root_db=root_db,
        )
        assert resolved is not None and resolved.db is root_db

        rows = merge_profile_session_rows(
            active_profile="programmer",
            local_db=local_db,
            root_db=root_db,
            load_rows=lambda db: [db.get_session("shared")],
        )
        assert len(rows) == 1
        assert rows[0]["source"] == "discord"
    finally:
        local_db.close()
        root_db.close()


def test_default_root_policy_accepts_legacy_but_rejects_named(tmp_path):
    local_db = SessionDB(db_path=tmp_path / "state.db")
    try:
        local_db.create_session("legacy", "cli")
        local_db.create_session("named", "discord", profile_name="programmer")

        from hermes_cli.session_resolution import resolve_profile_session

        assert resolve_profile_session(
            "legacy", active_profile="default", local_db=local_db, root_db=local_db
        ) is not None
        # The root/local store is the same for default, so ownership still
        # gates the direct local hit.
        assert resolve_profile_session(
            "named", active_profile="default", local_db=local_db, root_db=local_db
        ) is None
    finally:
        local_db.close()


def test_resolver_closes_every_opened_handle_when_resolution_raises(monkeypatch):
    import hermes_cli.session_resolution as resolution

    local_db = MagicMock()
    root_db = MagicMock()
    monkeypatch.setattr(
        resolution,
        "open_profile_session_dbs",
        lambda **_kwargs: (local_db, root_db, [local_db, root_db]),
    )
    monkeypatch.setattr(
        resolution,
        "resolve_profile_session",
        MagicMock(side_effect=RuntimeError("lookup failed")),
    )

    with pytest.raises(RuntimeError, match="lookup failed"):
        resolution.resolve_active_profile_session("broken", active_profile="programmer")

    local_db.close.assert_called_once_with()
    root_db.close.assert_called_once_with()
