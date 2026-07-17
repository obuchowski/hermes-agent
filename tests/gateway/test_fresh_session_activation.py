"""Behavior contracts for plugin-requested fresh gateway sessions."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import hermes_state

from agent.prompt_builder import build_context_files_prompt
from gateway.config import GatewayConfig, Platform, PlatformConfig, SessionResetPolicy
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import (
    AsyncSessionStore,
    SessionSource,
    SessionStore,
    canonicalize_session_cwd,
)
from hermes_cli.commands import gateway_help_lines, is_gateway_known_command
from hermes_cli.plugins import PluginManager
from hermes_state import SessionDB


_PROGRAMMING_PLUGIN = '''\
"""Programmer-profile slash command for fresh workspace sessions."""

from hermes_cli.plugins import FreshSessionRequest, PluginCommandContext


async def _programming(raw_args: str, context: PluginCommandContext) -> str:
    if context.effective_profile_name != "programmer":
        return "`/programming` is available only in the programmer profile."
    try:
        result = await context.create_fresh_session(FreshSessionRequest(cwd=raw_args))
    except (ValueError, RuntimeError) as exc:
        return str(exc)
    return f"Started a fresh programming session in `{result.cwd}`."


def register(ctx) -> None:
    ctx.register_command(
        "programming",
        _programming,
        description="Start a fresh Programmer session in a workspace",
        args_hint="<cwd>",
        gateway_context=True,
    )
'''


@pytest.fixture
def programming_plugin(tmp_path, monkeypatch):
    """Discover the thin plugin from a hermetic root HERMES_HOME."""
    import hermes_cli.plugins as plugins

    home = tmp_path / "plugin-home"
    plugin_dir = home / "plugins" / "programming"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "__init__.py").write_text(_PROGRAMMING_PLUGIN)
    (plugin_dir / "plugin.yaml").write_text(
        'name: programming\nversion: "1.0.0"\nkind: standalone\n'
    )
    bundled = tmp_path / "bundled-plugins"
    bundled.mkdir()

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugins, "get_bundled_plugins_dir", lambda: bundled)
    monkeypatch.setattr(plugins, "_get_enabled_plugins", lambda: {"programming"})
    monkeypatch.setattr(plugins, "_get_disabled_plugins", lambda: set())
    monkeypatch.setattr(PluginManager, "_scan_entry_points", lambda self: [])
    manager = PluginManager()
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    manager.discover_and_load()
    return manager


def _source(profile: str = "programmer", chat_id: str = "chat-1") -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type="dm",
        user_id="owner",
        profile=profile,
    )


def _store(tmp_path: Path, monkeypatch) -> tuple[SessionStore, SessionDB]:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", home / "state.db")
    config = GatewayConfig(
        sessions_dir=tmp_path / "sessions",
        default_reset_policy=SessionResetPolicy(mode="none"),
    )
    store = SessionStore(config.sessions_dir, config)
    assert store._db is not None
    return store, store._db


def test_valid_cwd_creates_empty_programmer_session_and_preserves_old(
    tmp_path, monkeypatch
):
    store, db = _store(tmp_path, monkeypatch)
    source = _source()
    old = store.get_or_create_session(source)
    db.append_message(old.session_id, "user", "keep this transcript")
    old_row = db.get_session(old.session_id)

    workspace = tmp_path / "repo"
    workspace.mkdir()
    canonical = canonicalize_session_cwd(str(workspace / "."))
    new = store.create_and_activate_fresh_session(
        source, cwd=canonical, profile_name="programmer"
    )

    assert new.session_id != old.session_id
    assert store.peek_session_id(new.session_key) == new.session_id
    new_row = db.get_session(new.session_id)
    assert new_row["cwd"] == str(workspace.resolve())
    assert new_row["profile_name"] == "programmer"
    assert new_row["parent_session_id"] is None
    assert db.get_messages(new.session_id) == []
    assert db.get_messages(old.session_id)[0]["content"] == "keep this transcript"
    assert db.get_session(old.session_id) == old_row


def test_invalid_and_missing_cwd_leave_route_and_database_untouched(
    tmp_path, monkeypatch
):
    store, db = _store(tmp_path, monkeypatch)
    source = _source()
    old = store.get_or_create_session(source)
    before_count = db.session_count()

    with pytest.raises(ValueError, match="Usage"):
        canonicalize_session_cwd("")
    with pytest.raises(ValueError, match="does not exist"):
        canonicalize_session_cwd(str(tmp_path / "missing"))

    assert store.peek_session_id(old.session_key) == old.session_id
    assert db.session_count() == before_count


def test_relative_cwd_is_defined_against_current_durable_workspace(tmp_path):
    current = tmp_path / "repo"
    child = current / "packages" / "api"
    child.mkdir(parents=True)

    assert canonicalize_session_cwd(
        "packages/api", current_cwd=str(current)
    ) == str(child.resolve())
    with pytest.raises(ValueError, match="Relative cwd requires"):
        canonicalize_session_cwd("packages/api")


def test_transaction_failure_keeps_old_route_authoritative(tmp_path, monkeypatch):
    store, db = _store(tmp_path, monkeypatch)
    source = _source()
    old = store.get_or_create_session(source)
    before_count = db.session_count()
    workspace = tmp_path / "repo"
    workspace.mkdir()

    def fail_between_create_and_route(**kwargs):
        def _partial(conn):
            conn.execute(
                "INSERT INTO sessions (id, source, cwd, profile_name, started_at) "
                "VALUES (?, ?, ?, ?, 0)",
                (
                    kwargs["session_id"],
                    kwargs["source"],
                    kwargs["cwd"],
                    kwargs["profile_name"],
                ),
            )
            raise RuntimeError("injected before route activation")

        db._execute_write(_partial)

    monkeypatch.setattr(db, "create_gateway_session_and_activate", fail_between_create_and_route)
    with pytest.raises(RuntimeError, match="injected"):
        store.create_and_activate_fresh_session(
            source, cwd=str(workspace.resolve()), profile_name="programmer"
        )

    assert store.peek_session_id(old.session_key) == old.session_id
    assert db.session_count() == before_count


def test_restart_reloads_route_profile_and_cwd(tmp_path, monkeypatch):
    store, db = _store(tmp_path, monkeypatch)
    source = _source()
    store.get_or_create_session(source)
    workspace = tmp_path / "repo"
    workspace.mkdir()
    created = store.create_and_activate_fresh_session(
        source, cwd=str(workspace.resolve()), profile_name="programmer"
    )

    restarted = SessionStore(store.sessions_dir, store.config)
    restored = restarted.get_or_create_session(source)
    row = restarted._db.get_session(restored.session_id)

    assert restored.session_id == created.session_id
    assert row["profile_name"] == "programmer"
    assert row["cwd"] == str(workspace.resolve())
    assert db.get_messages(created.session_id) == []


def test_standard_project_context_loader_uses_hermes_precedence(tmp_path):
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / ".git").mkdir()
    (workspace / "HERMES.md").write_text("STANDARD HERMES CONTEXT")
    (workspace / "AGENTS.md").write_text("SHOULD NOT WIN")

    prompt = build_context_files_prompt(cwd=str(workspace), skip_soul=True)

    assert "STANDARD HERMES CONTEXT" in prompt
    assert "SHOULD NOT WIN" not in prompt


def test_concurrent_routes_keep_distinct_cwds(tmp_path, monkeypatch):
    store, db = _store(tmp_path, monkeypatch)
    sources = [_source(chat_id="alpha"), _source(chat_id="beta")]
    workspaces = [tmp_path / "alpha", tmp_path / "beta"]
    for workspace in workspaces:
        workspace.mkdir()
    for source in sources:
        store.get_or_create_session(source)

    def activate(source, workspace):
        return store.create_and_activate_fresh_session(
            source,
            cwd=str(workspace.resolve()),
            profile_name="programmer",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        entries = list(pool.map(activate, sources, workspaces))
    rows = [db.get_session(entry.session_id) for entry in entries]
    assert [row["cwd"] for row in rows] == [str(path.resolve()) for path in workspaces]


def test_root_plugin_discovery_registers_catalog_and_enforces_profile_guard(
    programming_plugin,
):
    """The root loader owns discovery; the command itself owns profile scope."""
    command = programming_plugin._plugin_commands["programming"]
    denied = asyncio.run(command["handler"]("/tmp", MagicMock(
        effective_profile_name="default"
    )))

    assert "only in the programmer profile" in denied
    assert command["gateway_context"] is True
    assert is_gateway_known_command("programming") is True
    assert any(line.startswith("`/programming <cwd>`") for line in gateway_help_lines())


@pytest.mark.asyncio
async def test_gateway_dispatch_returns_direct_ack_and_keeps_fresh_transcript_empty(
    tmp_path, monkeypatch, programming_plugin
):
    """Exercise real gateway plugin dispatch through its typed command context."""
    from tests.gateway.test_gateway_command_dispatch_minimal import _make_runner

    store, db = _store(tmp_path, monkeypatch)
    source = _source()
    old = store.get_or_create_session(source)
    db.append_message(old.session_id, "user", "old transcript")
    workspace = tmp_path / "repo"
    workspace.mkdir()

    runner, _adapter = _make_runner()
    runner.config = store.config
    runner.config.platforms = {
        Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")
    }
    runner.session_store = store
    runner._async_session_store = AsyncSessionStore(store)
    runner._session_db = db
    runner._last_resolved_model = {}
    runner._set_session_reasoning_override = lambda *_args: None
    runner._is_telegram_topic_lane = lambda _source: False
    runner._clear_session_boundary_security_state = MagicMock()
    # A cache cleanup failure is post-commit and must not produce a false
    # command failure after the durable route has already switched.
    runner._evict_cached_agent = MagicMock(side_effect=RuntimeError("cache cleanup"))
    runner._handle_message_with_agent = AsyncMock(
        side_effect=AssertionError("plugin acknowledgement must not enter the agent")
    )
    event = MessageEvent(
        text=f"/programming {workspace}",
        message_type=MessageType.TEXT,
        source=source,
        message_id="m1",
        internal=True,
    )

    acknowledgement = await runner._handle_message(event)
    active_id = store.peek_session_id(store._generate_session_key(source))

    assert acknowledgement == f"Started a fresh programming session in `{workspace.resolve()}`."
    assert active_id != old.session_id
    assert db.get_messages(active_id) == []
    assert db.get_messages(old.session_id)[0]["content"] == "old transcript"
    assert db.get_session(active_id)["profile_name"] == "programmer"
    runner._clear_session_boundary_security_state.assert_called_once()
    runner._handle_message_with_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_busy_gateway_plugin_command_is_guarded_without_dispatch_or_mutation(
    tmp_path, monkeypatch, programming_plugin
):
    """A registered plugin command must never enter a running agent mid-turn."""
    from tests.gateway.test_gateway_command_dispatch_minimal import _make_runner

    store, db = _store(tmp_path, monkeypatch)
    source = _source()
    active = store.get_or_create_session(source)
    db.append_message(active.session_id, "user", "existing transcript")
    route_before = store.peek_session_id(active.session_key)
    transcript_before = db.get_messages(active.session_id)

    runner, adapter = _make_runner()
    runner.config = store.config
    runner.config.platforms = {
        Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")
    }
    runner.session_store = store
    runner._async_session_store = AsyncSessionStore(store)
    runner._session_db = db
    running_agent = MagicMock()
    runner._running_agents[active.session_key] = running_agent

    handler = MagicMock(side_effect=AssertionError("plugin handler ran mid-turn"))
    programming_plugin._plugin_commands["programming"]["handler"] = handler
    event = MessageEvent(
        text=f"/programming {tmp_path}",
        message_type=MessageType.TEXT,
        source=source,
        message_id="m-busy",
        internal=True,
    )

    response = await runner._handle_message(event)

    assert response == (
        "⏳ Agent is running — `/programming` can't run mid-turn. "
        "Wait for the current response or `/stop` first."
    )
    handler.assert_not_called()
    running_agent.steer.assert_not_called()
    running_agent.interrupt.assert_not_called()
    assert store.peek_session_id(active.session_key) == route_before
    assert db.get_messages(active.session_id) == transcript_before
    assert runner._pending_messages == {}
    assert adapter._pending_messages == {}


def test_programming_plugin_contains_no_prompt_discovery_copy():
    plugin = _PROGRAMMING_PLUGIN
    for forbidden in ("AGENTS.md", "HERMES.md", "CLAUDE.md", ".cursorrules", "system prompt"):
        assert forbidden not in plugin
