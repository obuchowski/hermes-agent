"""Profile-aware trusted handoff transfer coverage ported from 56a1741f6d."""

import json

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    """Create a SessionDB with a temporary database file."""
    session_db = SessionDB(db_path=tmp_path / "source_state.db")
    yield session_db
    session_db.close()


def test_transfer_session_lineage_copies_parent_chain_and_preserves_fields(db, tmp_path):
    db.create_session(
        "parent", "cli", cwd="/repo", profile_name="programmer",
        model_config={"model": "parent-model"},
    )
    db.set_session_title("parent", "Parent")
    db.append_message("parent", role="user", content="before compression")
    db.end_session("parent", "compression")
    db.create_session(
        "tip", "cli", parent_session_id="parent", cwd="/repo",
        profile_name="programmer", model_config={"model": "tip-model"},
    )
    db.set_session_title("tip", "Tip")
    db.append_message(
        "tip", role="assistant", content="tool call",
        tool_calls=[{"id": "call-1", "function": {"name": "noop"}}],
    )
    db.append_message("tip", role="tool", content="ok", tool_call_id="call-1")
    db.request_handoff("tip", "discord", profile_name="programmer")
    db.claim_handoff("tip")

    target = SessionDB(db_path=tmp_path / "gateway_state.db")
    try:
        result = target.transfer_session_lineage_from(
            db, "tip", profile_name="programmer"
        )
        assert result["transferred_ids"] == ["parent", "tip"]
        assert target.get_compression_lineage("tip") == ["parent", "tip"]
        copied_tip = target.get_session("tip")
        assert copied_tip["parent_session_id"] == "parent"
        assert copied_tip["profile_name"] == "programmer"
        assert copied_tip["handoff_state"] is None
        assert json.loads(copied_tip["model_config"]) == {"model": "tip-model"}
        copied_messages = target.get_messages_as_conversation(
            "tip", include_ancestors=True
        )
        assert [message["role"] for message in copied_messages] == [
            "user", "assistant", "tool"
        ]
        assert copied_messages[-1]["tool_call_id"] == "call-1"
        assert db.get_handoff_state("tip")["state"] == "running"
    finally:
        target.close()


def test_transfer_session_lineage_allows_identical_existing_replay(db, tmp_path):
    db.create_session("handoff-tip", "cli", profile_name="programmer")
    db.append_message("handoff-tip", role="user", content="same")
    target = SessionDB(db_path=tmp_path / "gateway_state.db")
    try:
        first = target.transfer_session_lineage_from(
            db, "handoff-tip", profile_name="programmer"
        )
        second = target.transfer_session_lineage_from(
            db, "handoff-tip", profile_name="programmer"
        )
        assert first["transferred_ids"] == ["handoff-tip"]
        assert first["already_present_ids"] == []
        assert second["transferred_ids"] == []
        assert second["already_present_ids"] == ["handoff-tip"]
        assert target.message_count(session_id="handoff-tip") == 1
    finally:
        target.close()


def test_transfer_session_lineage_accepts_large_trusted_local_session(db, tmp_path):
    db.create_session("large-handoff", "cli", profile_name="programmer")
    large_content = "x" * (SessionDB._IMPORT_MAX_SESSION_BYTES + 1)
    db.append_message("large-handoff", role="user", content=large_content)
    db._conn.execute(
        "UPDATE messages SET active = 0, compacted = 1 WHERE session_id = ?",
        ("large-handoff",),
    )
    db._conn.commit()

    target = SessionDB(db_path=tmp_path / "gateway_state.db")
    try:
        result = target.transfer_session_lineage_from(
            db, "large-handoff", profile_name="programmer"
        )
        assert result["transferred_ids"] == ["large-handoff"]
        messages = target.get_messages("large-handoff", include_inactive=True)
        assert len(messages) == 1
        assert messages[0]["content"] == large_content
        assert messages[0]["active"] == 0
        assert messages[0]["compacted"] == 1
    finally:
        target.close()


def test_transfer_session_lineage_collision_aborts_before_mutation(db, tmp_path):
    db.create_session("parent", "cli", profile_name="programmer")
    db.append_message("parent", role="user", content="source parent")
    db.create_session(
        "handoff-tip", "cli", parent_session_id="parent", profile_name="programmer"
    )
    db.append_message("handoff-tip", role="assistant", content="source tip")
    target = SessionDB(db_path=tmp_path / "gateway_state.db")
    target.create_session("handoff-tip", "discord", profile_name="programmer")
    target.append_message("handoff-tip", role="user", content="destination collision")
    try:
        with pytest.raises(ValueError, match="trusted transfer payload diverges"):
            target.transfer_session_lineage_from(
                db, "handoff-tip", profile_name="programmer"
            )
        assert target.get_session("parent") is None
        assert target.get_messages("handoff-tip")[0]["content"] == "destination collision"
    finally:
        target.close()


def test_transfer_session_lineage_rolls_back_mid_import_failure(db, tmp_path, monkeypatch):
    db.create_session("parent", "cli", profile_name="programmer")
    db.append_message("parent", role="user", content="source parent")
    db.end_session("parent", "compression")
    db.create_session(
        "handoff-tip", "cli", parent_session_id="parent", profile_name="programmer"
    )
    db.append_message("handoff-tip", role="assistant", content="source tip")
    target = SessionDB(db_path=tmp_path / "gateway_state.db")
    original = target._import_session_row
    calls = 0

    def explode_on_second(conn, raw, messages, sid):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected import failure")
        return original(conn, raw, messages, sid)

    monkeypatch.setattr(target, "_import_session_row", explode_on_second)
    try:
        with pytest.raises(RuntimeError, match="injected import failure"):
            target.transfer_session_lineage_from(
                db, "handoff-tip", profile_name="programmer"
            )
        assert target.get_session("parent") is None
        assert target.get_session("handoff-tip") is None
    finally:
        target.close()


def test_transfer_session_lineage_profile_mismatch_aborts(db, tmp_path):
    db.create_session("handoff-tip", "cli", profile_name="programmer")
    target = SessionDB(db_path=tmp_path / "gateway_state.db")
    try:
        with pytest.raises(ValueError, match="belongs to profile.*not.*ula"):
            target.transfer_session_lineage_from(db, "handoff-tip", profile_name="ula")
        assert target.get_session("handoff-tip") is None
    finally:
        target.close()


def test_mark_handoff_transferred_is_staged_and_idempotent(db):
    db.create_session("handoff-tip", "cli", profile_name="programmer")
    db.request_handoff("handoff-tip", "discord", profile_name="programmer")
    db.claim_handoff("handoff-tip")
    db.mark_handoff_transferred("handoff-tip")
    assert db.get_handoff_state("handoff-tip")["state"] == "transferred"
    db.mark_handoff_transferred("handoff-tip")
    assert db.get_handoff_state("handoff-tip")["state"] == "transferred"


@pytest.mark.parametrize(
    ("field", "source_value", "destination_value"),
    [
        ("model_config", {"flag": True}, {"flag": 1}),
        ("tool_calls", [{"function": {"arguments": {"flag": True}}}],
         [{"function": {"arguments": {"flag": 1}}}]),
        ("reasoning_details", [{"type": "detail", "nested": [True]}],
         [{"type": "detail", "nested": [1]}]),
    ],
)
def test_trusted_transfer_retry_uses_type_strict_nested_json_equality(
    db, tmp_path, field, source_value, destination_value
):
    session_kwargs = {"profile_name": "programmer"}
    if field == "model_config":
        session_kwargs["model_config"] = destination_value
    db.create_session("handoff-tip", "cli", **session_kwargs)
    message_kwargs = {}
    if field != "model_config":
        message_kwargs[field] = destination_value
    db.append_message("handoff-tip", role="assistant", content="same text", **message_kwargs)
    target = SessionDB(db_path=tmp_path / "gateway_state.db")
    try:
        target.transfer_session_lineage_from(db, "handoff-tip", profile_name="programmer")
        if field == "model_config":
            db._conn.execute(
                "UPDATE sessions SET model_config = ? WHERE id = ?",
                (json.dumps(source_value), "handoff-tip"),
            )
        else:
            db._conn.execute(
                f"UPDATE messages SET {field} = ? WHERE session_id = ?",
                (json.dumps(source_value), "handoff-tip"),
            )
        db._conn.commit()

        with pytest.raises(ValueError, match="trusted transfer payload diverges"):
            target.transfer_session_lineage_from(
                db, "handoff-tip", profile_name="programmer"
            )
    finally:
        target.close()
