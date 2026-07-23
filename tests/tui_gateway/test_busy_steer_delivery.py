"""Regression coverage for Desktop busy-time steer delivery."""

from __future__ import annotations

import importlib
import threading
import time
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def server():
    with patch.dict(
        "sys.modules",
        {
            "hermes_constants": MagicMock(
                get_hermes_home=MagicMock(return_value="/tmp/hermes_test_busy_steer")
            ),
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
            "hermes_state": MagicMock(),
        },
    ):
        mod = importlib.import_module("tui_gateway.server")
        yield mod
        mod._sessions.clear()
        mod._pending.clear()
        mod._answers.clear()


class _FinishingAgent:
    """A live turn that finishes after accepting late steering."""

    model = "test-model"
    provider = "test-provider"
    base_url = ""
    api_mode = ""
    interim_assistant_callback = None

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.first_turn_started = threading.Event()
        self.finish_first_turn = threading.Event()
        self.second_turn_finished = threading.Event()
        self._pending_steer: str | None = None
        self._steer_lock = threading.Lock()

    def clear_interrupt(self) -> None:
        pass

    def steer(self, text: str) -> bool:
        cleaned = text.strip()
        if not cleaned:
            return False
        with self._steer_lock:
            self._pending_steer = (
                f"{self._pending_steer}\n{cleaned}"
                if self._pending_steer
                else cleaned
            )
        return True

    def run_conversation(self, text: str, **_kwargs):
        self.calls.append(text)
        if len(self.calls) == 1:
            self.first_turn_started.set()
            assert self.finish_first_turn.wait(timeout=2)
            with self._steer_lock:
                pending = self._pending_steer
                self._pending_steer = None
            return {
                "final_response": "",
                "interrupted": False,
                "pending_steer": pending,
            }

        self.second_turn_finished.set()
        return {"final_response": "", "interrupted": False}


def _patch_turn_side_effects(server, monkeypatch) -> None:
    from tools.process_registry import process_registry

    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_set_session_context", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(server, "_clear_session_context", lambda _tokens: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda *_args: None)
    monkeypatch.setattr(server, "_session_cwd", lambda _session: "/tmp")
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    monkeypatch.setattr(server, "make_stream_renderer", lambda _cols: None)
    monkeypatch.setattr(server, "_load_interim_assistant_messages", lambda: False)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_get_usage", lambda _agent: {})
    monkeypatch.setattr(server, "render_message", lambda *_args: "")
    monkeypatch.setattr(server, "_session_info", lambda *_args: {"running": False})
    monkeypatch.setattr(process_registry, "drain_notifications", lambda **_kwargs: [])


def test_two_late_busy_steers_fall_back_to_one_ordered_next_turn(server, monkeypatch):
    """Accepted steers that miss the last tool boundary must not disappear."""
    _patch_turn_side_effects(server, monkeypatch)
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "steer")

    sid = "busy-steer"
    agent = _FinishingAgent()
    session = {
        "agent": agent,
        "attached_images": [],
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "last_active": time.time(),
        "running": True,
        "session_key": "stored-busy-steer",
    }
    server._sessions[sid] = session
    settle_submit_responses: list[dict] = []

    def emit(event: str, _sid: str, _payload=None) -> None:
        # Desktop can send the next optimistic queue entry as soon as the
        # first turn's message.complete paints. The agent has already returned
        # by then, so this must take the queue fallback instead of falsely
        # accepting another live steer.
        if event == "message.complete" and not settle_submit_responses:
            settle_submit_responses.append(
                server._handle_busy_submit("r2", sid, session, "ну что?", None)
            )

    monkeypatch.setattr(server, "_emit", emit)

    server._run_prompt_submit("rid", sid, session, "original")
    assert agent.first_turn_started.wait(timeout=2)

    first = server._handle_busy_submit("r1", sid, session, "да", None)
    assert first["result"]["status"] == "steered"

    agent.finish_first_turn.set()
    assert agent.second_turn_finished.wait(timeout=2)

    assert settle_submit_responses[0]["result"]["status"] == "queued"
    assert agent.calls == ["original", "да\n\nну что?"]
    assert session.get("queued_prompt") is None


def test_steer_unavailable_queues_multiple_busy_messages_losslessly(server, monkeypatch):
    """The steer policy's queue fallback preserves every arrival in order."""
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "steer")

    session = {
        "agent": object(),
        "history_lock": threading.Lock(),
        "last_active": 0,
        "running": True,
    }

    first = server._handle_busy_submit("r1", "sid", session, "да", None)
    second = server._handle_busy_submit("r2", "sid", session, "ну что?", None)

    assert first["result"]["status"] == "queued"
    assert second["result"]["status"] == "queued"
    assert session["queued_prompt"]["text"] == "да\n\nну что?"


def test_steer_transport_failure_falls_back_to_queue(server, monkeypatch):
    """A failed live injection is reported as queued, never falsely steered."""
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "steer")

    class _BrokenSteerAgent:
        def steer(self, _text: str) -> bool:
            raise RuntimeError("live steer unavailable")

        def interrupt(self) -> None:
            pass

    session = {
        "agent": _BrokenSteerAgent(),
        "history_lock": threading.Lock(),
        "last_active": 0,
        "running": True,
    }

    response = server._handle_busy_submit("rid", "sid", session, "still deliver this", None)

    assert response["result"]["status"] == "queued"
    assert session["queued_prompt"]["text"] == "still deliver this"


def test_session_steer_rejects_after_live_injection_window_closes(server):
    """Desktop only paints acceptance when a live steer can still land."""
    agent = MagicMock()
    sid = "closed-steer-window"
    server._sessions[sid] = {
        "_steer_open": False,
        "agent": agent,
        "history_lock": threading.Lock(),
        "running": True,
    }

    response = server._methods["session.steer"](
        "rid",
        {"session_id": sid, "text": "too late"},
    )

    assert response["result"]["status"] == "rejected"
    agent.steer.assert_not_called()
