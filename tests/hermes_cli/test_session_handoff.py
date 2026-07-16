"""Tests for session handoff (CLI to gateway platform).

The handoff state machine lives on the ``sessions`` table:

    None  → "pending" → "running" → ("completed" | "failed")

CLI side calls ``request_handoff`` and poll-waits on ``get_handoff_state``.
Gateway side iterates ``list_pending_handoffs``, calls ``claim_handoff`` to
flip pending → running, and finishes with ``complete_handoff`` or
``fail_handoff``.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from contextlib import contextmanager

import pytest

from hermes_state import SessionDB

class TestHandoffStateDB:
    """Test the handoff schema + helper methods on SessionDB."""

    @pytest.fixture
    def db(self, tmp_path, monkeypatch):
        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        return SessionDB(db_path=home / "state.db")

    def _make_session(self, db, session_id, source="cli", title=None):
        """Insert a session row directly for testing."""
        def _do(conn):
            conn.execute(
                "INSERT OR IGNORE INTO sessions (id, source, title, started_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, source, title, time.time()),
            )
        db._execute_write(_do)


    def test_request_handoff_stamps_trusted_profile_without_rewriting_owner(self, db):
        sid = "sess-profile"
        self._make_session(db, sid)
        db._conn.execute(
            "UPDATE sessions SET profile_name = '' WHERE id = ?", (sid,)
        )
        db._conn.commit()

        assert db.request_handoff(sid, "discord", profile_name="programmer") is True
        assert db.get_session(sid)["profile_name"] == "programmer"

        db.fail_handoff(sid, "retry")
        with pytest.raises(ValueError, match="belongs to profile 'programmer'"):
            db.request_handoff(sid, "discord", profile_name="ula")

        assert db.get_session(sid)["profile_name"] == "programmer"





    def test_list_pending_handoffs_excludes_running_and_terminal(self, db):
        a, b, c, d = "sess-a", "sess-b", "sess-c", "sess-d"
        for sid in (a, b, c, d):
            self._make_session(db, sid)

        db.request_handoff(a, "telegram")
        db.request_handoff(b, "discord")
        db.request_handoff(c, "telegram")
        db.claim_handoff(c)  # c is now running, not pending
        db.request_handoff(d, "slack")
        db.claim_handoff(d)
        db.complete_handoff(d)  # d is terminal

        pending = db.list_pending_handoffs()
        ids = [r["id"] for r in pending]
        assert set(ids) == {a, b}

    def test_complete_handoff_clears_error(self, db):
        sid = "sess-complete"
        self._make_session(db, sid)
        db.request_handoff(sid, "telegram")
        db.claim_handoff(sid)
        db.fail_handoff(sid, "transient")
        # User retries; mock the watcher path
        db.request_handoff(sid, "telegram")
        db.claim_handoff(sid)
        db.complete_handoff(sid)

        state = db.get_handoff_state(sid)
        assert state["state"] == "completed"
        assert state["error"] is None


    def test_post_transfer_failure_never_reactivates_local_copy(self, db):
        sid = "sess-transferred-warning"
        self._make_session(db, sid)
        db.request_handoff(sid, "discord", profile_name="programmer")
        db.claim_handoff(sid)
        db.mark_handoff_transferred(sid)

        db.fail_handoff(sid, "late failure must not roll ownership back")
        assert db.get_handoff_state(sid)["state"] == "transferred"

        db.complete_handoff_with_warning(sid, "delivery exhausted retries")
        assert db.get_handoff_state(sid) == {
            "state": "transferred_with_warning",
            "platform": "discord",
            "error": "delivery exhausted retries",
        }
        assert db.request_handoff(sid, "discord", profile_name="programmer") is False




    def test_full_pending_to_completed_flow(self, db):
        """End-to-end sequence the CLI + gateway watcher follow."""
        sid = "sess-flow"
        self._make_session(db, sid, title="my session")
        db.append_message(sid, "user", "Hello")
        db.append_message(sid, "assistant", "Hi there!")

        # CLI: request handoff
        assert db.request_handoff(sid, "telegram") is True
        assert db.get_handoff_state(sid)["state"] == "pending"

        # Gateway watcher: discover + claim
        pending = db.list_pending_handoffs()
        assert len(pending) == 1
        assert pending[0]["id"] == sid
        assert db.claim_handoff(sid) is True
        assert db.get_handoff_state(sid)["state"] == "running"

        # Gateway uses get_messages to load the transcript (real flow uses
        # session_store.switch_session which reads the same table).
        messages = db.get_messages(sid)
        assert [m["role"] for m in messages] == ["user", "assistant"]

        # Gateway: mark completed
        db.complete_handoff(sid)
        assert db.get_handoff_state(sid)["state"] == "completed"
        assert db.list_pending_handoffs() == []

    def test_handoff_reads_use_read_context_during_reconnect(self, db, monkeypatch):
        """Handoff polling must not borrow the reconnectable writer handle."""
        sid = "sess-read-path"
        self._make_session(db, sid)
        db.request_handoff(sid, "discord")

        entered = []
        original = db._read_ctx

        @contextmanager
        def traced_read_ctx():
            entered.append(True)
            with original() as conn:
                yield conn

        monkeypatch.setattr(db, "_read_ctx", traced_read_ctx)
        assert db.get_handoff_state(sid)["state"] == "pending"
        assert [row["id"] for row in db.list_pending_handoffs()] == [sid]
        assert len(entered) == 2


def test_named_profile_cli_command_queues_in_owning_db(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
    from hermes_cli.cli_commands_mixin import CLICommandsMixin

    sid = "named-cli-handoff"
    db = SessionDB(db_path=tmp_path / "profiles" / "programmer" / "state.db")
    db.create_session(sid, "cli")
    config = GatewayConfig(
        platforms={
            Platform.DISCORD: PlatformConfig(
                enabled=True,
                token="test",
                home_channel=HomeChannel(
                    platform=Platform.DISCORD,
                    chat_id="programmer-home",
                    name="Programmer Home",
                ),
            )
        }
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
    monkeypatch.setattr(
        "hermes_cli.profiles.get_active_profile_name", lambda: "programmer"
    )
    monkeypatch.setattr(
        db,
        "get_handoff_state",
        lambda _sid: {"state": "completed", "platform": "discord", "error": None},
    )
    class FakeCLI(CLICommandsMixin):
        pass

    cli = FakeCLI()
    cli.session_id = sid
    cli._session_db = db
    cli._agent_running = False
    cli._should_exit = False

    try:
        keep_running = CLICommandsMixin._handle_handoff_command(cli, "/handoff discord")

        assert keep_running is False
        assert cli._should_exit is True
        assert db.get_session(sid)["profile_name"] == "programmer"
    finally:
        db.close()
