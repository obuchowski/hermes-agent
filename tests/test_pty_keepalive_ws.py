import pytest

from hermes_cli import web_server


@pytest.mark.asyncio
async def test_snapshot_disconnect_ends_attach_without_asgi_error():
    from starlette.websockets import WebSocketDisconnect

    class DisconnectedSession:
        async def attach(self, ws):
            raise WebSocketDisconnect(code=1006)

    attached = await web_server._attach_pty_session(DisconnectedSession(), object())

    assert attached is False


@pytest.mark.asyncio
async def test_snapshot_disconnected_state_runtime_error_ends_attach_cleanly():
    class DisconnectedSession:
        async def attach(self, ws):
            raise RuntimeError('Cannot call "send" once a close message has been sent.')

    attached = await web_server._attach_pty_session(DisconnectedSession(), object())

    assert attached is False


@pytest.mark.asyncio
async def test_snapshot_attach_does_not_swallow_unrelated_failure():
    class BrokenSession:
        async def attach(self, ws):
            raise RuntimeError("unexpected attach failure")

    with pytest.raises(RuntimeError, match="unexpected attach failure"):
        await web_server._attach_pty_session(BrokenSession(), object())


@pytest.mark.asyncio
async def test_attach_token_reuses_same_session(monkeypatch):
    """Two connects with the same ?attach= token hit one spawned bridge."""
    spawned = []

    class FakeBridge:
        def __init__(self):
            self.alive = True

        def read(self, timeout):
            return b""        # idle forever

        def write(self, data):
            pass

        def resize(self, cols, rows):
            pass

        def close(self):
            self.alive = False

    def fake_spawn(argv, cwd=None, env=None):
        b = FakeBridge()
        spawned.append(b)
        return b

    monkeypatch.setattr(web_server.PtyBridge, "spawn", staticmethod(fake_spawn))
    # bypass auth + argv resolution for the test
    monkeypatch.setattr(web_server, "_ws_auth_reason", lambda ws: (None, "test"))
    monkeypatch.setattr(web_server, "_ws_host_origin_reason", lambda ws: None)
    monkeypatch.setattr(web_server, "_ws_client_reason", lambda ws: None)

    async def fake_argv(**kw):
        return (["x"], "/tmp", {})

    monkeypatch.setattr(web_server, "_resolve_chat_argv_async", fake_argv)

    from starlette.testclient import TestClient

    try:
        client = TestClient(web_server.app)
        with client.websocket_connect("/api/pty?attach=TOK1") as ws1:
            ws1.send_bytes(b"hi")
        with client.websocket_connect("/api/pty?attach=TOK1") as ws2:
            ws2.send_bytes(b"again")
        assert len(spawned) == 1                # reattached, did not respawn
    finally:
        web_server.PTY_REGISTRY._sessions.clear()
