import asyncio
import time

import pytest

from hermes_cli.pty_session import RingBuffer


def test_ringbuffer_keeps_everything_under_capacity():
    rb = RingBuffer(10)
    rb.append(b"abc")
    rb.append(b"def")
    assert rb.snapshot() == b"abcdef"
    assert rb.truncated is False


def test_ringbuffer_drops_oldest_over_capacity():
    rb = RingBuffer(4)
    rb.append(b"abcdef")          # 6 bytes into a 4-byte buffer
    assert rb.snapshot() == b"cdef"
    assert rb.truncated is True


def test_ringbuffer_truncation_across_appends():
    rb = RingBuffer(3)
    rb.append(b"ab")
    rb.append(b"cd")             # now "abcd" -> keep "bcd"
    assert rb.snapshot() == b"bcd"
    assert rb.truncated is True


class FakeBridge:
    """Implements the bridge contract PtySession depends on."""

    def __init__(self, chunks):
        self._chunks = list(chunks)   # bytes; b"" = idle tick; None = EOF
        self.written = bytearray()
        self.closed = False
        self.resized = None

    def read(self, timeout):
        if not self._chunks:
            return b""                # idle
        return self._chunks.pop(0)

    def write(self, data):
        self.written.extend(data)

    def resize(self, cols, rows):
        self.resized = (cols, rows)

    def close(self):
        self.closed = True


class FakeWS:
    def __init__(self):
        self.sent = []               # list of ("bytes"|"text", payload)
        self.close_code = None

    async def send_bytes(self, data):
        self.sent.append(("bytes", bytes(data)))

    async def send_text(self, text):
        self.sent.append(("text", text))

    async def close(self, code=1000, reason=""):
        self.close_code = code


class SnapshotFailingWS(FakeWS):
    async def send_bytes(self, data):
        raise ConnectionError("socket closed during snapshot replay")


class SnapshotCancelledWS(FakeWS):
    async def send_bytes(self, data):
        raise asyncio.CancelledError


class CoordinatedWS(FakeWS):
    def __init__(self):
        super().__init__()
        self.snapshot_started = asyncio.Event()
        self.release_snapshot = asyncio.Event()
        self.live_started = asyncio.Event()
        self.active_sends = 0
        self.max_active_sends = 0

    async def send_bytes(self, data):
        self.active_sends += 1
        self.max_active_sends = max(self.max_active_sends, self.active_sends)
        try:
            if data == b"history":
                self.snapshot_started.set()
                await self.release_snapshot.wait()
            elif data == b"live":
                self.live_started.set()
            self.sent.append(("bytes", bytes(data)))
        finally:
            self.active_sends -= 1


@pytest.mark.asyncio
async def test_attach_replays_buffer_then_streams_live():
    from hermes_cli.pty_session import PtySession
    bridge = FakeBridge([b"hello ", b"world", None])
    s = PtySession("k", bridge, buffer_cap=1024, read_timeout=0.01)
    await s.start()
    await asyncio.sleep(0.05)                      # drain consumes "hello world"
    ws = FakeWS()
    await s.attach(ws)
    replay = b"".join(p for kind, p in ws.sent if kind == "bytes")
    assert replay == b"hello world"
    await s.close()


@pytest.mark.asyncio
async def test_attach_serializes_snapshot_before_concurrent_live_drain():
    from hermes_cli.pty_session import PtySession

    bridge = FakeBridge([b"live", None])
    s = PtySession("k", bridge, buffer_cap=1024, read_timeout=0.01)
    s.buffer.append(b"history")
    ws = CoordinatedWS()

    attach_task = asyncio.create_task(s.attach(ws))
    await ws.snapshot_started.wait()
    await s.start()
    try:
        await asyncio.wait_for(ws.live_started.wait(), timeout=0.05)
    except asyncio.TimeoutError:
        pass
    ws.release_snapshot.set()
    await attach_task
    await s._drain_task

    assert ws.max_active_sends == 1
    assert ws.sent == [("bytes", b"history"), ("bytes", b"live")]
    await s.close()


@pytest.mark.asyncio
async def test_attach_rolls_back_when_snapshot_replay_fails():
    from hermes_cli.pty_session import PtySession

    bridge = FakeBridge([b"buffered", b""])
    s = PtySession("k", bridge, buffer_cap=1024, read_timeout=0.01)
    await s.start()
    await asyncio.sleep(0.05)
    ws = SnapshotFailingWS()

    with pytest.raises(ConnectionError, match="socket closed"):
        await s.attach(ws)

    assert s.attached is False
    assert s.last_detached_at is not None
    assert s._ws is None
    await s.close()


@pytest.mark.asyncio
async def test_attach_rolls_back_when_snapshot_replay_is_cancelled():
    from hermes_cli.pty_session import PtySession

    bridge = FakeBridge([b"buffered", b""])
    s = PtySession("k", bridge, buffer_cap=1024, read_timeout=0.01)
    await s.start()
    await asyncio.sleep(0.05)
    ws = SnapshotCancelledWS()

    with pytest.raises(asyncio.CancelledError):
        await s.attach(ws)

    assert s.attached is False
    assert s.last_detached_at is not None
    assert s._ws is None
    await s.close()


@pytest.mark.asyncio
async def test_detach_keeps_draining_into_buffer():
    from hermes_cli.pty_session import PtySession
    bridge = FakeBridge([b"one", b"", b"two"])
    s = PtySession("k", bridge, buffer_cap=1024, read_timeout=0.01)
    await s.start()
    ws = FakeWS()
    await s.attach(ws)
    s.detach(ws)
    assert s.attached is False
    assert s.last_detached_at is not None
    await asyncio.sleep(0.05)                      # "two" drains while detached
    ws2 = FakeWS()
    await s.attach(ws2)
    replay = b"".join(p for kind, p in ws2.sent if kind == "bytes")
    assert replay == b"onetwo"
    await s.close()


@pytest.mark.asyncio
async def test_eof_marks_dead_and_closes_socket_4410():
    from hermes_cli.pty_session import PtySession
    bridge = FakeBridge([b"bye", None])
    s = PtySession("k", bridge, buffer_cap=1024, read_timeout=0.01)
    await s.start()
    ws = FakeWS()
    await s.attach(ws)
    await asyncio.sleep(0.05)                      # drain hits None (EOF)
    assert s.alive is False
    assert ws.close_code == 4410
    await s.close()


from hermes_cli.pty_session import PtySessionRegistry, RegistryFull


def make_registry(ttl=1800.0, max_sessions=16):
    return PtySessionRegistry(ttl=ttl, max_sessions=max_sessions,
                              buffer_cap=1024, read_timeout=0.01)


@pytest.mark.asyncio
async def test_same_key_reattaches_same_session():
    reg = make_registry()
    b1 = FakeBridge([b"", b"", b""])
    s1, created1 = await reg.attach_or_spawn("tok", spawn=lambda: b1)
    s2, created2 = await reg.attach_or_spawn("tok", spawn=lambda: FakeBridge([]))
    assert created1 is True and created2 is False
    assert s1 is s2
    assert s2.bridge is b1                     # second spawn callable was NOT used
    await reg.close_all()


@pytest.mark.asyncio
async def test_reap_idle_closes_sessions_past_ttl():
    reg = make_registry(ttl=10.0)
    b = FakeBridge([b"", b""])
    s, _ = await reg.attach_or_spawn("tok", spawn=lambda: b)
    ws = FakeWS()
    await s.attach(ws)
    s.detach(ws)
    s.last_detached_at = time.monotonic() - 11.0   # detached 11s ago, ttl 10s
    await reg.reap_idle()
    assert b.closed is True
    s2, created = await reg.attach_or_spawn("tok", spawn=lambda: FakeBridge([]))
    assert created is True
    await reg.close_all()


@pytest.mark.asyncio
async def test_new_key_at_capacity_raises_when_none_reapable():
    reg = make_registry(max_sessions=1)
    b = FakeBridge([b"", b""])
    s, _ = await reg.attach_or_spawn("a", spawn=lambda: b)
    await s.attach(FakeWS())                    # attached → not reapable
    with pytest.raises(RegistryFull):
        await reg.attach_or_spawn("b", spawn=lambda: FakeBridge([]))
    await reg.close_all()


@pytest.mark.asyncio
async def test_reaper_loop_invokes_reap(monkeypatch):
    from hermes_cli.pty_session import run_reaper
    reg = make_registry()
    calls = {"n": 0}

    async def fake_reap(now=None):
        calls["n"] += 1

    monkeypatch.setattr(reg, "reap_idle", fake_reap)
    task = asyncio.create_task(run_reaper(reg, interval=0.01))
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert calls["n"] >= 2
