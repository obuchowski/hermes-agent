import asyncio
import os
import threading
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionContext, SessionSource
from gateway.session_context import (
    get_session_env,
    set_session_vars,
    clear_session_vars,
    reset_session_vars,
    restore_session_workspace,
    _VAR_MAP,
    _UNSET,
)


@pytest.fixture(autouse=True)
def _reset_contextvars():
    """Reset all session contextvars to _UNSET between tests.

    In production each asyncio.Task gets a fresh context copy where the
    defaults are _UNSET.  In tests all functions share the same thread
    context, so a clear_session_vars() from test A (which sets vars to "")
    would leak into test B.  This fixture ensures each test starts clean.
    """
    yield
    for var in _VAR_MAP.values():
        # Can't use var.reset() without a token; just set back to sentinel.
        var.set(_UNSET)


def test_set_session_env_sets_contextvars(monkeypatch):
    """_set_session_env should populate contextvars, not os.environ."""
    runner = object.__new__(GatewayRunner)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_name="Group",
        chat_type="group",
        user_id="123456",
        user_name="alice",
        thread_id="17585",
    )
    context = SessionContext(source=source, connected_platforms=[], home_channels={})

    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_NAME", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_TYPE", raising=False)
    monkeypatch.delenv("HERMES_SESSION_USER_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_USER_NAME", raising=False)
    monkeypatch.delenv("HERMES_SESSION_THREAD_ID", raising=False)

    tokens = runner._set_session_env(context)

    # Values should be readable via get_session_env (contextvar path)
    assert get_session_env("HERMES_SESSION_PLATFORM") == "telegram"
    assert get_session_env("HERMES_SESSION_SOURCE") == ""
    assert get_session_env("HERMES_SESSION_CHAT_ID") == "-1001"
    assert get_session_env("HERMES_SESSION_CHAT_NAME") == "Group"
    assert get_session_env("HERMES_SESSION_CHAT_TYPE") == "group"
    assert get_session_env("HERMES_SESSION_USER_ID") == "123456"
    assert get_session_env("HERMES_SESSION_USER_NAME") == "alice"
    assert get_session_env("HERMES_SESSION_THREAD_ID") == "17585"

    # os.environ should NOT be touched
    assert os.getenv("HERMES_SESSION_PLATFORM") is None
    assert os.getenv("HERMES_SESSION_SOURCE") is None
    assert os.getenv("HERMES_SESSION_CHAT_TYPE") is None
    assert os.getenv("HERMES_SESSION_THREAD_ID") is None

    # Clean up
    runner._clear_session_env(tokens)


def test_clear_session_env_restores_previous_state(monkeypatch):
    """_clear_session_env should restore contextvars to their pre-handler values."""
    runner = object.__new__(GatewayRunner)

    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_NAME", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_TYPE", raising=False)
    monkeypatch.delenv("HERMES_SESSION_USER_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_USER_NAME", raising=False)
    monkeypatch.delenv("HERMES_SESSION_THREAD_ID", raising=False)

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_name="Group",
        chat_type="group",
        user_id="123456",
        user_name="alice",
        thread_id="17585",
    )
    context = SessionContext(source=source, connected_platforms=[], home_channels={})

    tokens = runner._set_session_env(context)
    assert get_session_env("HERMES_SESSION_PLATFORM") == "telegram"
    assert get_session_env("HERMES_SESSION_USER_ID") == "123456"
    assert get_session_env("HERMES_SESSION_CHAT_TYPE") == "group"

    runner._clear_session_env(tokens)

    # After clear, contextvars should return to defaults (empty)
    assert get_session_env("HERMES_SESSION_PLATFORM") == ""
    assert get_session_env("HERMES_SESSION_CHAT_ID") == ""
    assert get_session_env("HERMES_SESSION_CHAT_NAME") == ""
    assert get_session_env("HERMES_SESSION_CHAT_TYPE") == ""
    assert get_session_env("HERMES_SESSION_USER_ID") == ""
    assert get_session_env("HERMES_SESSION_USER_NAME") == ""
    assert get_session_env("HERMES_SESSION_THREAD_ID") == ""


def test_get_session_env_falls_back_to_os_environ(monkeypatch):
    """get_session_env should fall back to os.environ when contextvar is unset."""
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "discord")

    # No contextvar set — should read from os.environ
    assert get_session_env("HERMES_SESSION_PLATFORM") == "discord"

    # Now set a contextvar — should prefer it
    tokens = set_session_vars(platform="telegram")
    assert get_session_env("HERMES_SESSION_PLATFORM") == "telegram"

    # After clear — should return "" (explicitly cleared), NOT fall back
    # to os.environ.  This is the fix for #10304: stale os.environ values
    # must not leak through after a gateway session is cleaned up.
    clear_session_vars(tokens)
    assert get_session_env("HERMES_SESSION_PLATFORM") == ""


# ---------------------------------------------------------------------------
# SESSION_KEY contextvars tests
# ---------------------------------------------------------------------------


def test_session_key_falls_back_to_os_environ(monkeypatch):
    """get_session_env for SESSION_KEY should fall back to os.environ."""
    monkeypatch.setenv("HERMES_SESSION_KEY", "env-session-123")

    # No contextvar set — should read from os.environ
    assert get_session_env("HERMES_SESSION_KEY") == "env-session-123"

    # Set contextvar — should prefer it
    tokens = set_session_vars(session_key="ctx-session-456")
    assert get_session_env("HERMES_SESSION_KEY") == "ctx-session-456"

    # After clear — should return "" (explicitly cleared), not os.environ (#10304)
    clear_session_vars(tokens)
    assert get_session_env("HERMES_SESSION_KEY") == ""


def test_handed_off_session_row_restores_cwd_and_raw_session_id(tmp_path, monkeypatch):
    from agent.runtime_cwd import resolve_context_cwd
    from tools.terminal_tool import clear_task_env_overrides, get_session_cwd

    process_cwd, workspace = tmp_path / "gateway-default", tmp_path / "workspace"
    process_cwd.mkdir()
    workspace.mkdir()
    monkeypatch.chdir(process_cwd)
    context = SessionContext(
        source=SessionSource(platform=Platform.DISCORD, chat_id="chan", chat_type="channel"),
        connected_platforms=[], home_channels={}, session_key="route", session_id="handoff-session",
    )
    assert restore_session_workspace(
        context, {"id": "handoff-session", "cwd": str(workspace)}
    ) == str(workspace)
    runner = object.__new__(GatewayRunner)
    tokens = runner._set_session_env(context)
    try:
        assert get_session_env("HERMES_SESSION_ID") == "handoff-session"
        assert resolve_context_cwd() == workspace
        assert get_session_cwd("handoff-session") == str(workspace)
    finally:
        runner._clear_session_env(tokens)
        clear_task_env_overrides("handoff-session")


@pytest.mark.asyncio
async def test_gateway_queries_raw_durable_session_id_before_binding(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    context = SessionContext(
        source=SessionSource(platform=Platform.DISCORD, chat_id="chan"),
        connected_platforms=[], home_channels={}, session_id="raw-handoff-id",
    )
    runner = object.__new__(GatewayRunner)
    runner._session_db = type("AsyncDB", (), {})()
    runner._session_db.get_session = AsyncMock(
        return_value={"id": "raw-handoff-id", "cwd": str(workspace)}
    )
    assert await runner._restore_session_workspace(context) == str(workspace)
    runner._session_db.get_session.assert_awaited_once_with("raw-handoff-id")
    assert context.cwd == str(workspace)


@pytest.mark.asyncio
async def test_gateway_workspace_validation_runs_off_event_loop(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    context = SessionContext(
        source=SessionSource(platform=Platform.DISCORD, chat_id="chan"),
        connected_platforms=[], home_channels={}, session_id="off-loop-session",
    )
    runner = object.__new__(GatewayRunner)
    runner._session_db = type("AsyncDB", (), {})()
    runner._session_db.get_session = AsyncMock(
        return_value={"id": "off-loop-session", "cwd": str(workspace)}
    )
    loop_thread, validation_threads = threading.get_ident(), []
    real_is_dir = Path.is_dir

    def recording_is_dir(path):
        validation_threads.append(threading.get_ident())
        return real_is_dir(path)

    monkeypatch.setattr(Path, "is_dir", recording_is_dir)
    assert await runner._restore_session_workspace(context) == str(workspace)
    assert validation_threads and validation_threads[0] != loop_thread


@pytest.mark.asyncio
async def test_gateway_workspace_lookup_error_preserves_known_good_state(tmp_path):
    from tools.terminal_tool import clear_task_env_overrides, get_session_cwd, register_task_env_overrides

    session_id = "lookup-error-session"
    known_good = tmp_path / "known-good"
    known_good.mkdir()
    context = SessionContext(
        source=SessionSource(platform=Platform.DISCORD, chat_id="chan"),
        connected_platforms=[], home_channels={}, session_id=session_id,
    )
    context.cwd = str(known_good)
    register_task_env_overrides(session_id, {"cwd": str(known_good)})
    runner = object.__new__(GatewayRunner)
    runner._session_db = type("AsyncDB", (), {})()
    runner._session_db.get_session = AsyncMock(side_effect=RuntimeError("sqlite unavailable"))
    try:
        with pytest.raises(RuntimeError, match="sqlite unavailable"):
            await runner._restore_session_workspace(context)
        assert context.cwd == str(known_good)
        assert get_session_cwd(session_id) == str(known_good)
    finally:
        clear_task_env_overrides(session_id)


def test_invalid_handed_off_cwd_clears_stale_task_state(tmp_path):
    from tools.terminal_tool import get_session_cwd, register_task_env_overrides

    session_id = "invalid-cwd-session"
    stale = tmp_path / "stale"
    stale.mkdir()
    register_task_env_overrides(session_id, {"cwd": str(stale)})
    context = SessionContext(
        source=SessionSource(platform=Platform.DISCORD, chat_id="chan"),
        connected_platforms=[], home_channels={}, session_id=session_id,
    )
    assert restore_session_workspace(
        context, {"id": session_id, "cwd": str(tmp_path / "missing")}
    ) == ""
    assert context.cwd == ""
    assert get_session_cwd(session_id) is None


def test_session_key_no_race_condition_with_contextvars(monkeypatch):
    """Prove contextvars isolates SESSION_KEY across concurrent async tasks.

    Two tasks set different session keys. With contextvars each task
    reads back its own value. With os.environ the second task would
    overwrite the first (the old bug).
    """
    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)

    results = {}

    async def handler(key: str, delay: float):
        tokens = set_session_vars(session_key=key)
        try:
            await asyncio.sleep(delay)
            read_back = get_session_env("HERMES_SESSION_KEY")
            results[key] = read_back
        finally:
            clear_session_vars(tokens)

    async def run():
        task_a = asyncio.create_task(handler("session-A", 0.15))
        await asyncio.sleep(0.05)
        task_b = asyncio.create_task(handler("session-B", 0.05))
        await asyncio.gather(task_a, task_b)

    asyncio.run(run())

    # Both tasks must read back their own session key
    assert results["session-A"] == "session-A", (
        f"Session A got '{results['session-A']}' instead of 'session-A' — race condition!"
    )
    assert results["session-B"] == "session-B", (
        f"Session B got '{results['session-B']}' instead of 'session-B' — race condition!"
    )


@pytest.mark.asyncio
async def test_run_in_executor_with_context_preserves_session_env(monkeypatch):
    """Gateway executor work should inherit session contextvars for tool routing."""
    runner = object.__new__(GatewayRunner)
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_THREAD_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_USER_ID", raising=False)

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="2144471399",
        chat_type="dm",
        user_id="123456",
        user_name="alice",
        thread_id=None,
    )
    context = SessionContext(
        source=source,
        connected_platforms=[],
        home_channels={},
        session_key="agent:main:telegram:dm:2144471399",
    )

    tokens = runner._set_session_env(context)
    try:
        result = await runner._run_in_executor_with_context(
            lambda: {
                "platform": get_session_env("HERMES_SESSION_PLATFORM"),
                "chat_id": get_session_env("HERMES_SESSION_CHAT_ID"),
                "user_id": get_session_env("HERMES_SESSION_USER_ID"),
                "session_key": get_session_env("HERMES_SESSION_KEY"),
            }
        )
    finally:
        runner._clear_session_env(tokens)
        runner._shutdown_executor()

    assert result == {
        "platform": "telegram",
        "chat_id": "2144471399",
        "user_id": "123456",
        "session_key": "agent:main:telegram:dm:2144471399",
    }




def test_cron_session_contextvar_preserves_legacy_env_fallback(monkeypatch):
    """Unset cron ContextVar keeps old env-only cron callers working."""
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")

    assert get_session_env("HERMES_CRON_SESSION") == "1"


def test_cron_session_explicit_blank_masks_leaked_env(monkeypatch):
    """Non-cron session bindings must override a stale process cron env flag."""
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")

    tokens = set_session_vars(platform="api_server", cron_session="")
    try:
        assert get_session_env("HERMES_CRON_SESSION") == ""
    finally:
        clear_session_vars(tokens)

    assert get_session_env("HERMES_CRON_SESSION") == ""


def test_cron_session_set_clear_and_reset_tristate(monkeypatch):
    """Cron marker supports _UNSET fallback, 1 cron, and  explicit clear."""
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")

    tokens = set_session_vars(cron_session="1")
    assert get_session_env("HERMES_CRON_SESSION") == "1"

    clear_session_vars(tokens)
    assert get_session_env("HERMES_CRON_SESSION") == ""

    reset_session_vars()
    assert get_session_env("HERMES_CRON_SESSION") == "1"


@pytest.mark.asyncio
async def test_plugin_slash_command_sees_session_env(monkeypatch):
    """A plugin-registered slash command handler must see the same HERMES_SESSION_*
    contextvars an agent turn would for that event (#108698): the agent-turn path binds
    them via _set_session_env before running, but plugin command dispatch is a separate,
    earlier path that previously called the handler with nothing bound."""
    from gateway.config import GatewayConfig, PlatformConfig
    from gateway.platforms.event import MessageEvent

    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_ID", raising=False)

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")})
    runner._draining = False

    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="c1", user_id="u1", user_name="tester", chat_type="dm",
    )
    event = MessageEvent(text="/gsd bind", source=source, message_id="m1")

    seen = {}

    def _handler(raw_args):
        seen["session_key"] = get_session_env("HERMES_SESSION_KEY")
        seen["chat_id"] = get_session_env("HERMES_SESSION_CHAT_ID")
        return f"Bound: {raw_args}"

    from hermes_cli import plugins as _plugins_mod
    monkeypatch.setattr(_plugins_mod, "get_plugin_command_handler",
                         lambda name: _handler if name == "gsd-bind" else None)

    handled, result, command = await runner._hm_dispatch_quick_and_plugin_commands(event, source, "gsd_bind")

    assert handled is True
    assert result == "Bound: bind"
    assert seen["session_key"] == runner._session_key_for_source(source)
    assert seen["session_key"] != ""
    assert seen["chat_id"] == "c1"
    # Bound only for the handler call, not leaked past dispatch
    assert get_session_env("HERMES_SESSION_KEY") == ""

