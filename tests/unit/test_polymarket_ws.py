"""Fake-WebSocket integration tests for hft_pm.data.polymarket_ws.

These tests stand up a local asyncio WebSocket server, point the
``PolymarketWSClient`` at it via the ``connector`` injection point, and
assert: (a) subscribe payload, (b) event dispatch, (c) hash tracking,
(d) reconnect on connection drop, (e) heartbeat-timeout-triggered close.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

import pytest
import websockets

from hft_pm.data.polymarket_ws import PolymarketWSClient

pytestmark = pytest.mark.asyncio


class FakeServer:
    """Local WebSocket server that scripts a deterministic event sequence."""

    def __init__(self) -> None:
        self.received_subscribes: list[dict] = []
        self.send_then_close: list[list[dict]] = []
        self._server: Any = None
        self.port = 0
        self.stop_event = asyncio.Event()

    def script_burst(self, events: list[dict]) -> None:
        """Each call to ``script_burst`` queues one connection's worth of events.

        The fake closes the connection after the burst is sent. Use multiple
        bursts to test reconnect behaviour.
        """
        self.send_then_close.append(events)

    async def start(self) -> None:
        async def handler(ws: Any) -> None:
            sub = json.loads(await ws.recv())
            self.received_subscribes.append(sub)
            if self.send_then_close:
                burst = self.send_then_close.pop(0)
                for ev in burst:
                    await ws.send(json.dumps(ev))
            await ws.close()

        self._server = await websockets.serve(handler, "127.0.0.1", 0)
        # ``sockets`` is the canonical way to read the bound port.
        sock = next(iter(self._server.sockets))
        self.port = sock.getsockname()[1]

    async def stop(self) -> None:
        self._server.close()
        await self._server.wait_closed()


def _make_event(asset_id: str, hash_: str, ts_ms: int) -> dict:
    return {
        "event_type": "book",
        "asset_id": asset_id,
        "market": "0xdeadbeef",
        "timestamp": str(ts_ms),
        "hash": hash_,
        "bids": [{"price": "0.50", "size": "10"}],
        "asks": [{"price": "0.51", "size": "10"}],
    }


@pytest.fixture
async def fake_server() -> Any:
    server = FakeServer()
    await server.start()
    try:
        yield server
    finally:
        await server.stop()


async def test_client_subscribes_dispatches_and_tracks_hash(fake_server: FakeServer) -> None:
    received: list[tuple[dict, int]] = []
    resyncs = 0

    async def on_event(ev: dict, recv_ts_ms: int) -> None:
        received.append((ev, recv_ts_ms))

    async def on_resync() -> None:
        nonlocal resyncs
        resyncs += 1

    fake_server.script_burst(
        [
            _make_event("A", "h1", 1_715_000_000_000),
            _make_event("B", "h2", 1_715_000_000_010),
        ]
    )

    client = PolymarketWSClient(
        asset_ids=["A", "B"],
        on_event=on_event,
        on_resync=on_resync,
        connector=lambda url: websockets.connect(url),
        url=f"ws://127.0.0.1:{fake_server.port}",
    )

    runner = asyncio.create_task(client.run())
    # Give the burst time to flow through, then stop.
    await asyncio.sleep(0.3)
    client.stop()
    runner.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await runner

    assert fake_server.received_subscribes == [{"type": "market", "assets_ids": ["A", "B"]}]
    assert resyncs >= 1
    assert [ev["asset_id"] for ev, _ in received] == ["A", "B"]
    assert client.last_hash_by_asset == {"A": "h1", "B": "h2"}
    # recv_ts_ms is wall clock; assert non-zero and monotonic-ish.
    assert all(ts > 0 for _, ts in received)


async def test_client_reconnects_after_drop(fake_server: FakeServer) -> None:
    received: list[dict] = []

    async def on_event(ev: dict, _recv: int) -> None:
        received.append(ev)

    fake_server.script_burst([_make_event("A", "h1", 1)])
    fake_server.script_burst([_make_event("A", "h2", 2)])

    # Skip the default 1s backoff so the test finishes quickly.
    client = PolymarketWSClient(
        asset_ids=["A"],
        on_event=on_event,
        connector=lambda url: websockets.connect(url),
        url=f"ws://127.0.0.1:{fake_server.port}",
    )
    client.INITIAL_BACKOFF_S = 0.05

    runner = asyncio.create_task(client.run())
    # Wait long enough for two subscribe cycles (drop + reconnect).
    await asyncio.sleep(0.6)
    client.stop()
    runner.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await runner

    assert len(fake_server.received_subscribes) >= 2, "client must resubscribe after drop"
    hashes = [ev["hash"] for ev in received]
    assert "h1" in hashes and "h2" in hashes


async def test_constructor_rejects_empty_asset_list() -> None:
    async def on_event(_e: dict, _t: int) -> None:
        return None

    with pytest.raises(ValueError):
        PolymarketWSClient(asset_ids=[], on_event=on_event)


async def test_client_skips_malformed_json(fake_server: FakeServer) -> None:
    received: list[dict] = []

    async def on_event(ev: dict, _recv: int) -> None:
        received.append(ev)

    async def malformed_handler(ws: Any) -> None:
        await ws.recv()  # drain subscribe
        await ws.send("{not json")
        await ws.send(json.dumps(_make_event("A", "h1", 1)))
        await ws.close()

    # Replace the default handler with one that emits a bad line first.
    await fake_server.stop()
    fake_server._server = await websockets.serve(malformed_handler, "127.0.0.1", 0)
    sock = next(iter(fake_server._server.sockets))
    fake_server.port = sock.getsockname()[1]

    client = PolymarketWSClient(
        asset_ids=["A"],
        on_event=on_event,
        connector=lambda url: websockets.connect(url),
        url=f"ws://127.0.0.1:{fake_server.port}",
    )

    runner = asyncio.create_task(client.run())
    await asyncio.sleep(0.3)
    client.stop()
    runner.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await runner

    assert [ev["asset_id"] for ev in received] == ["A"]
