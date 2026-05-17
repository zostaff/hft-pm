"""Unit tests for hft_pm.live.paper_trade.

The tests drive :class:`PaperTrader` against an in-memory fake WebSocket
connector instead of a real network — same approach as
``test_polymarket_ws.py`` but lighter (no asyncio server). Each test
scripts a finite sequence of raw WS messages, runs the trader long
enough for them to drain, then stops it and asserts on the resulting
log and fills.

Coverage:

* DoNothing strategy → no fills, PnL stays at zero, log includes
  resync + per-event PnL snapshots.
* place_limit → trade at our level produces a maker fill with the
  Polymarket SPORTS-category rebate.
* Latency injection: a place that has not yet "arrived" must not fill.
* Cancel before the matching trade prevents the fill.
* Kill switch halt cancels all locally-resting orders and writes a
  halt record before the runner exits.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from typing import Any

import pytest

from hft_pm.data.polymarket_ws import PolymarketWSClient
from hft_pm.fees.polymarket import FeeCategory
from hft_pm.live.paper_trade import PaperTrader
from hft_pm.orderbook.events import SimEvent
from hft_pm.risk.limits import HaltReason, KillSwitch, RiskLimits
from hft_pm.simulator.latency import ConstantLatency
from hft_pm.strategies.base import DoNothing, SimulatorAPI, Strategy

pytestmark = pytest.mark.asyncio

ASSET = "12345"
MARKET = "0xdeadbeef"


# ---------------------------------------------------------------------------
# In-memory fake WS connector
# ---------------------------------------------------------------------------


class _FakeWS:
    """Minimal ``_WSLike`` that yields scripted strings then ends."""

    def __init__(self, messages: list[str]) -> None:
        self.messages = list(messages)
        self.sent: list[str] = []
        self._closed = False
        self._idx = 0

    async def send(self, data: str) -> None:
        self.sent.append(data)

    async def close(self) -> None:
        self._closed = True

    def __aiter__(self) -> _FakeWS:
        return self

    async def __anext__(self) -> str:
        # Yield control so concurrent tasks (heartbeat watchdog, test
        # driver) get a chance to run between messages.
        await asyncio.sleep(0)
        if self._closed or self._idx >= len(self.messages):
            raise StopAsyncIteration
        msg = self.messages[self._idx]
        self._idx += 1
        return msg


class _FakeConnector:
    """Single-burst connector: first connection drains all scripted
    messages, subsequent connections yield empty (so reconnect loops
    don't replay)."""

    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self._payload = [json.dumps(m) for m in messages]
        self.connections: list[_FakeWS] = []

    def __call__(self, _url: str) -> _ConnectorCM:
        return _ConnectorCM(self)


class _ConnectorCM:
    def __init__(self, parent: _FakeConnector) -> None:
        self._parent = parent

    async def __aenter__(self) -> _FakeWS:
        payload = self._parent._payload if not self._parent.connections else []
        ws = _FakeWS(payload)
        self._parent.connections.append(ws)
        return ws

    async def __aexit__(self, *_: object) -> None:
        return None


# ---------------------------------------------------------------------------
# Wire-format helpers
# ---------------------------------------------------------------------------


def _book_event(
    ts: int,
    bids: list[tuple[float, float]],
    asks: list[tuple[float, float]],
) -> dict[str, Any]:
    return {
        "event_type": "book",
        "asset_id": ASSET,
        "market": MARKET,
        "timestamp": str(ts),
        "hash": f"h{ts}",
        "bids": [{"price": str(p), "size": str(s)} for p, s in bids],
        "asks": [{"price": str(p), "size": str(s)} for p, s in asks],
    }


def _trade_event(ts: int, price: float, size: float, side: str) -> dict[str, Any]:
    return {
        "event_type": "last_trade_price",
        "asset_id": ASSET,
        "market": MARKET,
        "timestamp": str(ts),
        "price": str(price),
        "size": str(size),
        "side": side,
    }


def _read_log(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


async def _drive_until_idle(trader: PaperTrader, *, drain_s: float = 0.3) -> None:
    """Run ``trader.run()`` long enough for the scripted burst to drain."""
    task = asyncio.create_task(trader.run())
    await asyncio.sleep(drain_s)
    trader.stop()
    try:
        await asyncio.wait_for(task, timeout=3.0)
    except TimeoutError:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        raise


@pytest.fixture(autouse=True)
def _short_ws_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shorten WS reconnect backoff so tests don't burn 1 s per connection cycle."""
    monkeypatch.setattr(PolymarketWSClient, "INITIAL_BACKOFF_S", 0.02)
    monkeypatch.setattr(PolymarketWSClient, "MAX_BACKOFF_S", 0.1)


# ---------------------------------------------------------------------------
# Test strategies
# ---------------------------------------------------------------------------


class _PlaceOnceBid(Strategy):
    """Place exactly one bid as soon as the book has a mid."""

    def __init__(self, price: float, size: float = 1.0, *, post_only: bool = True) -> None:
        self.price = price
        self.size = size
        self.post_only = post_only
        self.order_id: int | None = None

    def on_event(self, sim: SimulatorAPI, _event: SimEvent) -> None:
        if self.order_id is None and sim.book.mid() is not None:
            self.order_id = sim.place_limit("bid", self.price, self.size, post_only=self.post_only)


class _PlaceThenCancel(Strategy):
    """Place on the first event with a mid; cancel on the next event."""

    def __init__(self, price: float) -> None:
        self.price = price
        self.order_id: int | None = None
        self._cancelled = False

    def on_event(self, sim: SimulatorAPI, _event: SimEvent) -> None:
        if self.order_id is None and sim.book.mid() is not None:
            self.order_id = sim.place_limit("bid", self.price, 1.0)
            return
        if self.order_id is not None and not self._cancelled:
            sim.cancel(self.order_id)
            self._cancelled = True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_do_nothing_strategy_returns_zero_pnl(tmp_path: Path) -> None:
    msgs = [
        _book_event(1000, [(0.49, 100.0)], [(0.51, 100.0)]),
        _trade_event(2000, 0.50, 5.0, "BUY"),
    ]
    connector = _FakeConnector(msgs)
    trader = PaperTrader(
        token_id=ASSET,
        tick=0.01,
        fee_category=FeeCategory.GEOPOLITICS,
        strategy=DoNothing(),
        kill_switch=KillSwitch(RiskLimits()),
        latency=ConstantLatency(0),
        log_path=tmp_path / "paper.jsonl",
        connector=connector,
        ws_url="ws://test",
        heartbeat_check_interval_s=10.0,
    )

    await _drive_until_idle(trader)

    assert trader.cash == 0.0
    assert trader.inventory == 0.0
    assert trader.fills == []
    log = _read_log(tmp_path / "paper.jsonl")
    types = [r["type"] for r in log]
    assert "ws_resync" in types
    assert types.count("pnl") >= 2  # one snapshot per processed event


async def test_maker_fill_when_trade_hits_our_level(tmp_path: Path) -> None:
    # Empty public size at 0.49 → we sit at the front of the queue.
    # SELL trade at 0.49 hits our bid for the full size.
    msgs = [
        _book_event(1000, [(0.48, 50.0)], [(0.51, 100.0)]),
        _trade_event(2000, 0.49, 1.0, "SELL"),
    ]
    connector = _FakeConnector(msgs)
    strat = _PlaceOnceBid(price=0.49, size=1.0)
    trader = PaperTrader(
        token_id=ASSET,
        tick=0.01,
        fee_category=FeeCategory.SPORTS,
        strategy=strat,
        kill_switch=KillSwitch(RiskLimits()),
        latency=ConstantLatency(0),
        log_path=tmp_path / "paper.jsonl",
        connector=connector,
        ws_url="ws://test",
        heartbeat_check_interval_s=10.0,
    )

    await _drive_until_idle(trader)

    assert strat.order_id is not None, "strategy never placed"
    assert len(trader.fills) == 1
    fill = trader.fills[0]
    assert fill.is_maker is True
    assert fill.side == "bid"
    assert fill.price == 0.49
    assert fill.size == 1.0
    # SPORTS: peak_rate=0.0075, rebate_share=0.25. Expected rebate ≈ 0.00187
    assert fill.fee_paid == 0.0
    assert 0.001 < fill.rebate_received < 0.003
    assert trader.inventory == 1.0
    # cash = -0.49 * 1 + rebate
    assert -0.49 + fill.rebate_received == pytest.approx(trader.cash, abs=1e-9)

    log = _read_log(tmp_path / "paper.jsonl")
    types = [r["type"] for r in log]
    assert types.count("order_place") == 1
    assert types.count("order_arrival") == 1
    assert types.count("fill") == 1


async def test_latency_injection_blocks_premature_fill(tmp_path: Path) -> None:
    # Place at t=1000 with 200 ms latency → arrival at 1200.
    # Trade arrives at t=1100 (BEFORE the place_arrival lands) → no fill.
    msgs = [
        _book_event(1000, [(0.48, 50.0)], [(0.51, 100.0)]),
        _trade_event(1100, 0.49, 1.0, "SELL"),
        _book_event(1300, [(0.48, 50.0)], [(0.51, 100.0)]),
    ]
    connector = _FakeConnector(msgs)
    strat = _PlaceOnceBid(price=0.49, size=1.0)
    trader = PaperTrader(
        token_id=ASSET,
        tick=0.01,
        fee_category=FeeCategory.SPORTS,
        strategy=strat,
        kill_switch=KillSwitch(RiskLimits()),
        latency=ConstantLatency(200),
        log_path=tmp_path / "paper.jsonl",
        connector=connector,
        ws_url="ws://test",
        heartbeat_check_interval_s=10.0,
    )

    await _drive_until_idle(trader)

    assert strat.order_id is not None
    assert len(trader.fills) == 0
    # By the third event (ts=1300), the place_arrival at 1200 should have
    # landed, so our order is now resting in the book.
    assert strat.order_id in trader.book.our_orders


async def test_cancel_before_trade_prevents_fill(tmp_path: Path) -> None:
    msgs = [
        _book_event(1000, [(0.48, 50.0)], [(0.51, 100.0)]),  # strategy places
        _book_event(1500, [(0.48, 50.0)], [(0.51, 100.0)]),  # strategy cancels
        _trade_event(2000, 0.49, 1.0, "SELL"),  # would have hit if not cancelled
    ]
    connector = _FakeConnector(msgs)
    strat = _PlaceThenCancel(price=0.49)
    trader = PaperTrader(
        token_id=ASSET,
        tick=0.01,
        fee_category=FeeCategory.SPORTS,
        strategy=strat,
        kill_switch=KillSwitch(RiskLimits()),
        latency=ConstantLatency(0),
        log_path=tmp_path / "paper.jsonl",
        connector=connector,
        ws_url="ws://test",
        heartbeat_check_interval_s=10.0,
    )

    await _drive_until_idle(trader)

    assert strat.order_id is not None
    assert strat._cancelled is True
    assert len(trader.fills) == 0
    assert strat.order_id not in trader.book.our_orders


class _HaltAfterNTicks(KillSwitch):
    """KillSwitch test double that trips MAX_DRAWDOWN after N ticks."""

    def __init__(self, limits: RiskLimits, *, trip_after: int) -> None:
        super().__init__(limits)
        self._n = 0
        self._trip_after = trip_after

    def tick(self, *, now_s: float, current_pnl: float, inventory: float) -> None:
        super().tick(now_s=now_s, current_pnl=current_pnl, inventory=inventory)
        self._n += 1
        if self._n >= self._trip_after and not self.halted:
            # Mirror the real switch's halt API.
            self._trip(HaltReason.MAX_DRAWDOWN)


async def test_kill_switch_halt_cancels_local_orders_and_logs(tmp_path: Path) -> None:
    # First event: strategy places a bid. Second event: kill switch trips
    # *before* the strategy callback runs, so the bid (still resting from
    # the previous tick) must be cancelled locally.
    msgs = [
        _book_event(1000, [(0.48, 50.0)], [(0.51, 100.0)]),  # tick 1: place
        _book_event(2000, [(0.48, 50.0)], [(0.51, 100.0)]),  # tick 2: halt
    ]
    connector = _FakeConnector(msgs)
    strat = _PlaceOnceBid(price=0.49, size=1.0)
    ks = _HaltAfterNTicks(RiskLimits(), trip_after=2)

    trader = PaperTrader(
        token_id=ASSET,
        tick=0.01,
        fee_category=FeeCategory.SPORTS,
        strategy=strat,
        kill_switch=ks,
        latency=ConstantLatency(0),
        log_path=tmp_path / "paper.jsonl",
        connector=connector,
        ws_url="ws://test",
        heartbeat_check_interval_s=10.0,
    )

    await _drive_until_idle(trader)

    assert ks.halted is True
    assert ks.halt_reason == HaltReason.MAX_DRAWDOWN
    # Local resting order cleared by _on_halt.
    assert strat.order_id not in trader.book.our_orders
    # Halt record present in the log.
    log = _read_log(tmp_path / "paper.jsonl")
    halts = [r for r in log if r["type"] == "halt"]
    assert len(halts) >= 1
    assert halts[0]["reason"] == HaltReason.MAX_DRAWDOWN.value


async def test_kill_switch_heartbeat_timeout_halts_on_silent_feed(tmp_path: Path) -> None:
    # One book event, then no further messages. With a 0.1 s heartbeat
    # timeout the kill switch's watchdog should detect the silence.
    msgs = [_book_event(1000, [(0.49, 100.0)], [(0.51, 100.0)])]
    connector = _FakeConnector(msgs)
    ks = KillSwitch(RiskLimits(heartbeat_timeout_s=0.1))

    trader = PaperTrader(
        token_id=ASSET,
        tick=0.01,
        fee_category=FeeCategory.GEOPOLITICS,
        strategy=DoNothing(),
        kill_switch=ks,
        latency=ConstantLatency(0),
        log_path=tmp_path / "paper.jsonl",
        connector=connector,
        ws_url="ws://test",
        heartbeat_check_interval_s=0.03,
    )

    task = asyncio.create_task(trader.run())
    # Give the heartbeat watchdog enough time to fire.
    try:
        await asyncio.wait_for(task, timeout=2.0)
    except TimeoutError:
        trader.stop()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        raise

    assert ks.halted is True
    assert ks.halt_reason == HaltReason.HEARTBEAT_TIMEOUT
    log = _read_log(tmp_path / "paper.jsonl")
    assert any(r["type"] == "halt" for r in log)


async def test_foreign_asset_events_are_ignored(tmp_path: Path) -> None:
    msgs = [
        # Event for a different asset id — must be filtered out.
        {**_book_event(1000, [(0.49, 100.0)], [(0.51, 100.0)]), "asset_id": "OTHER"},
        _trade_event(2000, 0.50, 1.0, "BUY"),
    ]
    connector = _FakeConnector(msgs)
    trader = PaperTrader(
        token_id=ASSET,
        tick=0.01,
        fee_category=FeeCategory.GEOPOLITICS,
        strategy=DoNothing(),
        kill_switch=KillSwitch(RiskLimits()),
        latency=ConstantLatency(0),
        log_path=tmp_path / "paper.jsonl",
        connector=connector,
        ws_url="ws://test",
        heartbeat_check_interval_s=10.0,
    )

    await _drive_until_idle(trader)

    log = _read_log(tmp_path / "paper.jsonl")
    # Only the trade event for ASSET should have produced a pnl snapshot.
    pnls = [r for r in log if r["type"] == "pnl"]
    assert len(pnls) == 1


# ---------------------------------------------------------------------------
# Regression tests for bugs found during 2026-05-17 live paper-trade
# ---------------------------------------------------------------------------


class _StreamingFakeWS:
    """``_WSLike`` that streams a single canned payload at a steady cadence
    until :meth:`close` is called, with no natural end.

    The :class:`_FakeWS` used by the burst tests exits as soon as its
    scripted list is exhausted, which masks two bugs we hit on a real WS:
    (a) ``stop()`` doesn't actually break ``async for raw in ws`` and
    (b) ``_on_halt`` re-fires on every event after the halt. This streaming
    variant exposes both.
    """

    def __init__(self, payload: str, *, cadence_s: float = 0.005) -> None:
        self._payload = payload
        self._cadence_s = cadence_s
        self._closed = False

    async def send(self, _data: str) -> None:
        return None

    async def close(self) -> None:
        self._closed = True

    def __aiter__(self) -> _StreamingFakeWS:
        return self

    async def __anext__(self) -> str:
        await asyncio.sleep(self._cadence_s)
        if self._closed:
            raise StopAsyncIteration
        return self._payload


class _StreamingConnector:
    def __init__(self, payload: dict[str, Any], *, cadence_s: float = 0.005) -> None:
        self._payload_str = json.dumps(payload)
        self._cadence_s = cadence_s
        self.connections: list[_StreamingFakeWS] = []

    def __call__(self, _url: str) -> _StreamingCM:
        return _StreamingCM(self)


class _StreamingCM:
    def __init__(self, parent: _StreamingConnector) -> None:
        self._parent = parent

    async def __aenter__(self) -> _StreamingFakeWS:
        ws = _StreamingFakeWS(self._parent._payload_str, cadence_s=self._parent._cadence_s)
        self._parent.connections.append(ws)
        return ws

    async def __aexit__(self, *_: object) -> None:
        return None


async def test_foreign_asset_traffic_does_not_trip_kill_switch_heartbeat(
    tmp_path: Path,
) -> None:
    """Regression: 2026-05-17 paper-trade on Thunder NBA halted within 30 s
    because the kill-switch heartbeat was tracking events on the strategy's
    own token only. Foreign-asset events flowing steadily on the same WS
    must keep ``KillSwitch._last_event_s`` fresh — otherwise a low-volume
    market is unusable for paper trading even when the WS itself is healthy.
    """
    # Stream foreign-asset events at ~200 / sec for the full test duration.
    foreign = {**_book_event(1000, [(0.5, 100.0)], [(0.51, 100.0)]), "asset_id": "OTHER"}
    connector = _StreamingConnector(foreign, cadence_s=0.005)

    # Heartbeat timeout 0.2 s but we'll let the test run for 0.8 s: with the
    # bug fixed the switch must NOT trip; with the bug, it trips ~0.2 s in.
    ks = KillSwitch(RiskLimits(heartbeat_timeout_s=0.2))
    trader = PaperTrader(
        token_id=ASSET,
        tick=0.01,
        fee_category=FeeCategory.GEOPOLITICS,
        strategy=DoNothing(),
        kill_switch=ks,
        latency=ConstantLatency(0),
        log_path=tmp_path / "paper.jsonl",
        connector=connector,
        ws_url="ws://test",
        heartbeat_check_interval_s=0.05,
    )

    task = asyncio.create_task(trader.run())
    try:
        await asyncio.sleep(0.8)  # well past 0.2 s heartbeat_timeout
        assert ks.halted is False, "heartbeat tripped despite live foreign-asset traffic"
    finally:
        trader.stop()
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except TimeoutError:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            raise


async def test_stop_force_closes_active_streaming_ws(tmp_path: Path) -> None:
    """Regression: 2026-05-17 paper-trade stayed in ``async for raw in ws``
    for 2.5 hours after a halt because ``client.stop()`` set a flag but did
    not close the live WS. With the fix, ``stop()`` must terminate the
    runner promptly even on a continuously-streaming feed.
    """
    foreign = {**_book_event(1000, [(0.5, 100.0)], [(0.51, 100.0)]), "asset_id": "OTHER"}
    connector = _StreamingConnector(foreign, cadence_s=0.005)
    trader = PaperTrader(
        token_id=ASSET,
        tick=0.01,
        fee_category=FeeCategory.GEOPOLITICS,
        strategy=DoNothing(),
        kill_switch=KillSwitch(RiskLimits()),
        latency=ConstantLatency(0),
        log_path=tmp_path / "paper.jsonl",
        connector=connector,
        ws_url="ws://test",
        heartbeat_check_interval_s=10.0,
    )

    task = asyncio.create_task(trader.run())
    await asyncio.sleep(0.1)  # let the stream settle
    trader.stop()
    # Must exit within 0.5 s. Pre-fix this hung until WS naturally closed.
    await asyncio.wait_for(task, timeout=0.5)


async def test_halt_logs_only_once_under_continued_traffic(tmp_path: Path) -> None:
    """Regression: 2026-05-17 paper-trade wrote 35 halt records because every
    post-halt event re-entered ``_on_halt`` and the "already cleaned up"
    branch always re-logged with ``"repeat": True``. Exactly one halt record
    must be emitted regardless of how many events arrive after the trip.
    """
    # First event on our asset triggers halt (via _HaltAfterNTicks).
    # Then stream foreign events that, before the fix, would each cause a
    # repeat halt log via _process_market_event re-entering _on_halt.
    own_event = _book_event(1000, [(0.48, 50.0)], [(0.51, 100.0)])
    foreign = {**_book_event(2000, [(0.5, 100.0)], [(0.51, 100.0)]), "asset_id": "OTHER"}

    class _DualConnector:
        """One own-asset event + sustained foreign-asset stream on the same WS."""

        def __init__(self) -> None:
            self.connections: list[Any] = []

        def __call__(self, _url: str) -> _DualConnectorCM:
            return _DualConnectorCM(self)

    class _DualConnectorCM:
        def __init__(self, parent: _DualConnector) -> None:
            self._parent = parent

        async def __aenter__(self) -> _DualWS:
            ws = _DualWS(json.dumps(own_event), json.dumps(foreign))
            self._parent.connections.append(ws)
            return ws

        async def __aexit__(self, *_: object) -> None:
            return None

    class _DualWS:
        def __init__(self, own: str, foreign: str) -> None:
            self._own = own
            self._foreign = foreign
            self._sent_own = False
            self._closed = False

        async def send(self, _data: str) -> None:
            return None

        async def close(self) -> None:
            self._closed = True

        def __aiter__(self) -> _DualWS:
            return self

        async def __anext__(self) -> str:
            await asyncio.sleep(0.005)
            if self._closed:
                raise StopAsyncIteration
            if not self._sent_own:
                self._sent_own = True
                return self._own
            return self._foreign

    connector = _DualConnector()
    ks = _HaltAfterNTicks(RiskLimits(), trip_after=1)  # halt on the first own-asset tick
    trader = PaperTrader(
        token_id=ASSET,
        tick=0.01,
        fee_category=FeeCategory.GEOPOLITICS,
        strategy=DoNothing(),
        kill_switch=ks,
        latency=ConstantLatency(0),
        log_path=tmp_path / "paper.jsonl",
        connector=connector,
        ws_url="ws://test",
        heartbeat_check_interval_s=10.0,
    )

    task = asyncio.create_task(trader.run())
    try:
        # Let foreign events stream after the halt. Pre-fix each one
        # re-entered _on_halt and wrote a repeat record; we expect the run
        # to terminate quickly (because stop() now force-closes the WS).
        await asyncio.wait_for(task, timeout=1.5)
    except TimeoutError:
        trader.stop()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        raise

    log = _read_log(tmp_path / "paper.jsonl")
    halts = [r for r in log if r["type"] == "halt"]
    assert len(halts) == 1, f"expected exactly 1 halt record, got {len(halts)}: {halts}"
    assert "repeat" not in halts[0], "no record should carry the legacy 'repeat' flag"
