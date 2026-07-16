from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.session import AsyncSessionStore, SessionStore
from hermes_state import AsyncSessionDB, SessionDB


@pytest.fixture
def multiplex_homes(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    programmer = root / "profiles" / "programmer"
    programmer.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    return root, programmer


def _discord_config(*, home_id: str, multiplex: bool = False) -> GatewayConfig:
    return GatewayConfig(
        platforms={
            Platform.DISCORD: PlatformConfig(
                enabled=True,
                token="test-token",
                home_channel=HomeChannel(
                    platform=Platform.DISCORD,
                    chat_id=home_id,
                    name=f"home-{home_id}",
                ),
            )
        },
        multiplex_profiles=multiplex,
    )


def test_handoff_sources_include_each_served_profile_db(multiplex_homes):
    import gateway.run as gateway_run

    root_home, programmer_home = multiplex_homes
    root_db = SessionDB(db_path=root_home / "state.db")
    programmer_db = SessionDB(db_path=programmer_home / "state.db")
    programmer_db.create_session("pending", "cli", profile_name="programmer")
    programmer_db.request_handoff(
        "pending", "discord", profile_name="programmer"
    )
    programmer_db.close()

    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = _discord_config(home_id="default-home", multiplex=True)
    runner._session_db = AsyncSessionDB(root_db)
    runner._handoff_profile_dbs = {}
    try:
        sources = runner._handoff_sources()
        by_profile = {profile: (db, scope) for profile, db, scope in sources}
        assert set(by_profile) == {"default", "programmer"}
        assert by_profile["default"][1] == "gateway_root"
        assert by_profile["programmer"][1] == "profile"
        assert [
            row["id"]
            for row in by_profile["programmer"][0]._db.list_pending_handoffs()
        ] == ["pending"]
    finally:
        for handle in runner._handoff_profile_dbs.values():
            handle._db.close()
        root_db.close()


@pytest.mark.asyncio
async def test_named_profile_handoff_transfers_lineage_and_uses_profile_discord(
    multiplex_homes, monkeypatch
):
    import gateway.run as gateway_run

    root_home, programmer_home = multiplex_homes
    local_db = SessionDB(db_path=programmer_home / "state.db")
    local_db.create_session(
        "parent", "cli", profile_name="programmer", cwd="/projects/hermes"
    )
    local_db.append_message("parent", role="user", content="before compression")
    local_db.end_session("parent", "compression")
    local_db.create_session(
        "tip",
        "cli",
        parent_session_id="parent",
        profile_name="programmer",
        cwd="/projects/hermes",
    )
    local_db.append_message(
        "tip",
        role="assistant",
        content="tool call",
        tool_calls=[{"id": "call-1", "function": {"name": "noop"}}],
    )
    local_db.append_message(
        "tip", role="tool", content="tool result", tool_call_id="call-1"
    )
    local_db.append_message("tip", role="assistant", content="after tool")
    local_db.request_handoff("tip", "discord", profile_name="programmer")
    local_db.claim_handoff("tip")

    default_cfg = _discord_config(home_id="default-home", multiplex=True)
    programmer_cfg = _discord_config(home_id="programmer-home")
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: programmer_cfg)

    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = default_cfg
    runner.adapters = {Platform.DISCORD: MagicMock(name="default-discord")}
    programmer_adapter = MagicMock(name="programmer-discord")
    programmer_adapter.create_handoff_thread = AsyncMock(return_value="thread-1")
    programmer_adapter.send = AsyncMock(
        return_value=SimpleNamespace(success=True)
    )
    runner._profile_adapters = {"programmer": {Platform.DISCORD: programmer_adapter}}
    runner.session_store = SessionStore(root_home / "sessions", default_cfg)
    runner.session_store._db.close()
    runner.session_store._db = SessionDB(db_path=root_home / "state.db")
    runner._async_session_store = AsyncSessionStore(runner.session_store)
    runner._session_db = AsyncSessionDB(runner.session_store._db)
    runner._evict_cached_agent = MagicMock()
    runner._release_running_agent_state = MagicMock()
    runner._evict_cached_agent = MagicMock()
    runner._release_running_agent_state = MagicMock()
    captured = {}

    async def _handle_message(event):
        captured["event"] = event
        captured["history"] = runner.session_store._db.get_messages_as_conversation(
            "tip", include_ancestors=True
        )
        return "continued"

    runner._handle_message = AsyncMock(side_effect=_handle_message)

    try:
        row = local_db.get_session("tip")
        await runner._process_handoff(
            row,
            source_profile="programmer",
            source_db=AsyncSessionDB(local_db),
        )

        event = captured["event"]
        assert event.source.profile == "programmer"
        assert event.source.chat_id == "programmer-home"
        assert event.source.thread_id == "thread-1"
        assert [message["role"] for message in captured["history"]] == [
            "user",
            "assistant",
            "tool",
            "assistant",
        ]
        assert runner.session_store._db.get_compression_lineage("tip") == [
            "parent",
            "tip",
        ]
        assert runner.session_store._db.get_session("tip")["profile_name"] == "programmer"
        assert local_db.get_handoff_state("tip")["state"] == "transferred"
        key = "agent:programmer:discord:thread:programmer-home:thread-1"
        assert runner.session_store._entries[key].session_id == "tip"
        runner.adapters[Platform.DISCORD].create_handoff_thread.assert_not_called()
        programmer_adapter.create_handoff_thread.assert_awaited_once()

        from hermes_cli.session_resolution import resolve_profile_session

        resumed = resolve_profile_session(
            "tip",
            active_profile="programmer",
            local_db=local_db,
            root_db=runner.session_store._db,
        )
        assert resumed is not None and resumed.db is runner.session_store._db
        resumed.db.append_message("tip", role="user", content="back on CLI")
        assert [
            message["content"]
            for message in runner.session_store._db.get_messages("tip")
        ][-1] == "back on CLI"
        assert [
            message["content"] for message in local_db.get_messages("tip")
        ][-1] == "after tool"
    finally:
        runner.session_store._db.close()
        local_db.close()


@pytest.mark.asyncio
async def test_precommit_missing_profile_adapter_keeps_local_owner(
    multiplex_homes, monkeypatch
):
    import gateway.run as gateway_run

    root_home, programmer_home = multiplex_homes
    local_db = SessionDB(db_path=programmer_home / "state.db")
    local_db.create_session("tip", "cli", profile_name="programmer")
    local_db.request_handoff("tip", "discord", profile_name="programmer")
    local_db.claim_handoff("tip")

    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = _discord_config(home_id="default-home", multiplex=True)
    runner.adapters = {Platform.DISCORD: MagicMock()}
    runner._profile_adapters = {}
    runner._session_db = AsyncSessionDB(SessionDB(db_path=root_home / "state.db"))

    try:
        with pytest.raises(RuntimeError, match="profile 'programmer'.*not active"):
            await runner._process_handoff(
                local_db.get_session("tip"),
                source_profile="programmer",
                source_db=AsyncSessionDB(local_db),
            )
        assert runner._session_db._db.get_session("tip") is None
        assert local_db.get_handoff_state("tip")["state"] == "running"
    finally:
        runner._session_db._db.close()
        local_db.close()


@pytest.mark.asyncio
async def test_transfer_commit_survives_source_state_update_failure(
    multiplex_homes, monkeypatch
):
    import gateway.run as gateway_run

    root_home, programmer_home = multiplex_homes
    local_db = SessionDB(db_path=programmer_home / "state.db")
    local_db.create_session("tip", "cli", profile_name="programmer")
    local_db.append_message("tip", role="user", content="still canonical")
    local_db.request_handoff("tip", "discord", profile_name="programmer")
    local_db.claim_handoff("tip")

    default_cfg = _discord_config(home_id="default-home", multiplex=True)
    programmer_cfg = _discord_config(home_id="programmer-home")
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: programmer_cfg)

    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = default_cfg
    runner.adapters = {}
    adapter = MagicMock()
    adapter.create_handoff_thread = AsyncMock(return_value="thread-1")
    runner._profile_adapters = {"programmer": {Platform.DISCORD: adapter}}
    runner.session_store = SessionStore(root_home / "sessions", default_cfg)
    runner.session_store._db.close()
    runner.session_store._db = SessionDB(db_path=root_home / "state.db")
    runner._async_session_store = AsyncSessionStore(runner.session_store)
    runner._session_db = AsyncSessionDB(runner.session_store._db)
    runner._evict_cached_agent = MagicMock()
    runner._release_running_agent_state = MagicMock()
    source_db = AsyncSessionDB(local_db)
    source_db.mark_handoff_transferred = AsyncMock(
        side_effect=OSError("source state write failed")
    )
    runner._handle_message = AsyncMock(return_value="delivered continuation")
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True))

    try:
        with pytest.raises(gateway_run.HandoffPostCommitError):
            await runner._process_handoff(
                local_db.get_session("tip"),
                source_profile="programmer",
                source_db=source_db,
            )

        runner._handle_message.assert_awaited_once()
        adapter.send.assert_awaited_once()
        assert runner.session_store._db.get_session("tip")["profile_name"] == "programmer"
        assert local_db.get_handoff_state("tip")["state"] == "running"
        await source_db.complete_handoff_with_warning(
            "tip", "source state write failed"
        )
        assert local_db.get_handoff_state("tip")["state"] == "transferred_with_warning"
    finally:
        runner.session_store._db.close()
        local_db.close()
