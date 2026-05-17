"""Tier 2 paper-trading runner: live WS + simulated fills + KillSwitch + JSONL log.

Subscribes to the Polymarket market channel for one token, maintains a
local :class:`L2OrderBook`, and runs a :class:`Strategy` against the
live feed. Orders are not sent to Polymarket: they live only inside
this process and fill when the public feed prints a trade against our
resting price level (via ``L2OrderBook.process_trade``).

The runner exposes the same :class:`SimulatorAPI` surface as
:class:`hft_pm.simulator.engine.Backtester`, so any strategy that runs
in backtest plugs into paper trading without changes.

Lifecycle::

    trader = PaperTrader(token_id=..., tick=..., fee_category=...,
                        strategy=..., kill_switch=..., latency=...,
                        log_path=Path("data/paper/.../asset.jsonl"))
    await trader.run()   # blocks until stop() or kill switch halt

Per-event pipeline:

1. Parse raw WS dict to a typed :class:`MarketEvent` (filter foreign assets).
2. Drain pending internal arrivals (latency-delayed place/cancel) whose
   ``arrival_ms`` is ≤ ``event.timestamp_ms``.
3. Apply the market event to the book; for ``trade`` events,
   :meth:`L2OrderBook.process_trade` walks our resting orders and any
   that get hit are recorded as maker fills with Polymarket V2 fees.
4. Tick the :class:`KillSwitch` with wall-clock time and mark-to-mid PnL.
   If halted, cancel locals and stop the runner.
5. Call ``strategy.on_event(self, sim_ev)``.
6. Append a PnL snapshot record to the JSONL log.

Kill-switch heartbeat runs on a separate asyncio task using wall-clock
time (not server timestamps) so a frozen feed actually trips the halt.

Reconnect handling: ``PolymarketWSClient`` already reconnects with
exponential backoff. On every (re)subscribe we log a ``resync`` record;
the next ``book`` snapshot replaces the public book. Our local resting
orders are *kept* across resync — paper mode has no broker to cancel
them, and the strategy's queue tracking should pick up where it left
off when the snapshot lands. This is a v1 approximation; the Tier 3
live runner will reconcile against the broker's open-orders endpoint.
"""

from __future__ import annotations

import asyncio
import contextlib
import heapq
import itertools
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Literal

from ..data.polymarket_ws import Connector, PolymarketWSClient
from ..data.schemas import (
    BookEvent,
    LastTradePriceEvent,
    MarketEvent,
    PriceChangeEvent,
    TickSizeChangeEvent,
    UnknownEventTypeError,
    parse_event,
)
from ..fees.polymarket import FeeCategory, maker_rebate, taker_fee
from ..orderbook.events import FillRecord, SimEvent
from ..orderbook.l2_book import L2OrderBook
from ..risk.limits import KillSwitch
from ..simulator.latency import LatencyModel
from ..simulator.metrics import mark_to_mid
from ..strategies.base import SimulatorAPI, Strategy

logger = logging.getLogger(__name__)

Side = Literal["bid", "ask"]


@dataclass
class _PlacePayload:
    order_id: int
    side: Side
    price: float
    size: float
    post_only: bool


@dataclass
class _CancelPayload:
    order_id: int


class PaperTrader(SimulatorAPI):
    """Live WS + simulated fills + KillSwitch + JSONL log.

    Parameters
    ----------
    token_id:
        Polymarket asset id (token id) to subscribe to. One token per
        instance; for two-sided coverage run two ``PaperTrader``\\ s.
    tick:
        Tick size for the local book (used for queue accounting and
        snap-to-grid by quoting strategies).
    fee_category:
        Polymarket fee tier; rebates/fees on simulated fills follow
        :mod:`hft_pm.fees.polymarket`.
    strategy:
        :class:`Strategy` instance. Same interface as backtest.
    kill_switch:
        :class:`KillSwitch` instance. The runner halts when it trips
        and writes a ``halt`` record to the log.
    latency:
        :class:`LatencyModel` controlling when our place/cancel actions
        become effective on the local book. Use ``ConstantLatency(0)``
        only if you want zero-latency paper, which underestimates real
        adverse selection.
    log_path:
        Destination JSONL file. Parent directories are created on demand.
        Append-only; one record per line; flushed after every write.
    ws_url:
        Optional override for the WS URL (default: production Polymarket).
    connector:
        Optional WS connector factory; tests pass a local fake here.
    heartbeat_check_interval_s:
        How often the kill-switch heartbeat watchdog runs (seconds).

    Notes
    -----
    The runner is single-asset by design — multi-token would require a
    multi-book state machine and we want Tier 2 to be debuggable. Run
    multiple instances under a supervisor for portfolio paper trading.
    """

    DEFAULT_HEARTBEAT_CHECK_INTERVAL_S = 5.0

    def __init__(
        self,
        *,
        token_id: str,
        tick: float,
        fee_category: FeeCategory,
        strategy: Strategy,
        kill_switch: KillSwitch,
        latency: LatencyModel,
        log_path: Path,
        ws_url: str | None = None,
        connector: Connector | None = None,
        heartbeat_check_interval_s: float | None = None,
    ) -> None:
        self._token_id = token_id
        self._book = L2OrderBook(tick=tick)
        self._fee_category = fee_category
        self._strategy = strategy
        self._kill_switch = kill_switch
        self._latency = latency
        self._ws_url = ws_url
        self._connector = connector
        self._heartbeat_interval_s = (
            heartbeat_check_interval_s
            if heartbeat_check_interval_s is not None
            else self.DEFAULT_HEARTBEAT_CHECK_INTERVAL_S
        )

        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        # Line-buffered text mode + explicit flush per write so the log
        # survives a hard kill. The handle is owned by PaperTrader and
        # closed in run()'s finally block.
        self._log_fh: IO[str] = open(  # noqa: SIM115
            self._log_path, "a", encoding="utf-8", buffering=1
        )

        # Engine state mirrors Backtester.
        self._now_ms: int = 0
        self._cash: float = 0.0
        self._inventory: float = 0.0
        self._fees_paid: float = 0.0
        self._rebates_received: float = 0.0
        self._fills: list[FillRecord] = []
        self._last_mid: float | None = None

        self._pending_arrivals: list[SimEvent] = []  # min-heap by (ts, seq)
        self._pending_places: dict[int, _PlacePayload] = {}
        self._our_order_ids: set[int] = set()

        self._next_order_id = itertools.count(start=1)
        self._seq_counter = itertools.count()

        self._client: PolymarketWSClient | None = None
        self._stop = False
        # _on_halt is idempotent: cleanup runs once, then subsequent calls
        # just re-assert client.stop() without re-logging. This prevents the
        # log spam observed on 2026-05-17 when a slow market plus a flawed
        # heartbeat semantics produced 35 halt records over 2.5 hours.
        self._halt_logged: bool = False

    # ------------------------------------------------------------------
    # SimulatorAPI surface
    # ------------------------------------------------------------------

    @property
    def now_ms(self) -> int:
        return self._now_ms

    @property
    def inventory(self) -> float:
        return self._inventory

    @property
    def book(self) -> L2OrderBook:
        return self._book

    def place_limit(
        self,
        side: Side,
        price: float,
        size: float,
        *,
        post_only: bool = True,
    ) -> int:
        if size <= 0:
            raise ValueError("size must be positive")
        if side not in ("bid", "ask"):
            raise ValueError(f"invalid side: {side!r}")
        order_id = next(self._next_order_id)
        arrival_ms = self._latency.sample(self._now_ms)
        payload = _PlacePayload(order_id, side, price, size, post_only)
        self._pending_places[order_id] = payload
        self._push_arrival(arrival_ms, "place_arrival", payload)
        self._log(
            {
                "type": "order_place",
                "ts_ms": self._now_ms,
                "order_id": order_id,
                "side": side,
                "price": price,
                "size": size,
                "post_only": post_only,
                "arrival_ms": arrival_ms,
            }
        )
        return order_id

    def cancel(self, order_id: int) -> None:
        arrival_ms = self._latency.sample(self._now_ms)
        self._push_arrival(arrival_ms, "cancel_arrival", _CancelPayload(order_id))
        self._log(
            {
                "type": "order_cancel",
                "ts_ms": self._now_ms,
                "order_id": order_id,
                "arrival_ms": arrival_ms,
            }
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Connect, subscribe, and drive the strategy until stop / halt."""
        kwargs: dict[str, Any] = {}
        if self._connector is not None:
            kwargs["connector"] = self._connector
        if self._ws_url is not None:
            kwargs["url"] = self._ws_url

        self._client = PolymarketWSClient(
            asset_ids=[self._token_id],
            on_event=self._on_ws_event,
            on_disconnect=self._on_ws_disconnect,
            on_resync=self._on_ws_resync,
            **kwargs,
        )

        heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        try:
            await self._client.run()
        finally:
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task
            try:
                self._log_fh.flush()
            finally:
                self._log_fh.close()

    def stop(self) -> None:
        """Request a graceful shutdown."""
        self._stop = True
        if self._client is not None:
            self._client.stop()

    # ------------------------------------------------------------------
    # WS callbacks
    # ------------------------------------------------------------------

    async def _on_ws_event(self, raw: dict[str, Any], recv_ts_ms: int) -> None:
        # Update WS-liveness heartbeat on EVERY raw message, before any
        # filtering. The kill-switch heartbeat-timeout halt means "feed is
        # dead", not "my own token went quiet" — a low-volume token can
        # legitimately have hours between trades while the WS itself is
        # streaming healthily for other assets the runner doesn't trade.
        # See risk.limits.KillSwitch.note_ws_message for the rationale.
        self._kill_switch.note_ws_message(now_s=time.time())

        # Filter foreign assets and control messages without asset_id.
        if raw.get("asset_id") != self._token_id:
            return
        try:
            event = parse_event(raw, recv_ts_ms)
        except UnknownEventTypeError as e:
            logger.debug("unknown event_type: %s", e)
            return
        except (ValueError, TypeError) as e:
            logger.warning("event parse failed: %s", e)
            self._log({"type": "parse_error", "error": str(e), "wall_ts": time.time()})
            return
        self._process_market_event(event)

    async def _on_ws_disconnect(self, reason: str) -> None:
        logger.warning("ws disconnect: %s", reason)
        self._log({"type": "ws_disconnect", "reason": reason})

    async def _on_ws_resync(self) -> None:
        logger.info("ws (re)subscribed")
        self._log({"type": "ws_resync"})

    # ------------------------------------------------------------------
    # Event processing
    # ------------------------------------------------------------------

    def _process_market_event(self, ev: MarketEvent) -> None:
        ts = ev.timestamp_ms
        # Drain latency-pending arrivals that should have already landed.
        self._drain_arrivals_up_to(ts)

        # Advance the clock and apply the event to the book.
        if ts > self._now_ms:
            self._now_ms = ts
        sim_kind = _market_event_to_kind(ev)
        sim_ev = SimEvent(ts, next(self._seq_counter), sim_kind, ev)  # type: ignore[arg-type]
        self._apply_market_event(sim_ev)

        # Risk check first, then strategy callback.
        self._tick_kill_switch()
        if self._kill_switch.halted:
            self._on_halt()
            return

        try:
            self._strategy.on_event(self, sim_ev)
        except Exception as e:
            # A buggy strategy must not crash the long-running runner; log and continue.
            logger.exception("strategy raised on_event")
            self._log({"type": "strategy_error", "ts_ms": ts, "error": repr(e)})

        self._log_pnl_snapshot()

    def _drain_arrivals_up_to(self, ts_ms: int) -> None:
        while self._pending_arrivals and self._pending_arrivals[0].timestamp_ms <= ts_ms:
            sim_ev = heapq.heappop(self._pending_arrivals)
            if sim_ev.timestamp_ms > self._now_ms:
                self._now_ms = sim_ev.timestamp_ms
            if sim_ev.kind == "place_arrival":
                assert isinstance(sim_ev.payload, _PlacePayload)
                self._handle_place_arrival(sim_ev.payload)
            elif sim_ev.kind == "cancel_arrival":
                assert isinstance(sim_ev.payload, _CancelPayload)
                self._handle_cancel_arrival(sim_ev.payload)

    def _apply_market_event(self, sim_ev: SimEvent) -> None:
        kind = sim_ev.kind
        payload = sim_ev.payload
        if kind == "book":
            assert isinstance(payload, BookEvent)
            bids = [(lvl.price, lvl.size) for lvl in payload.bids]
            asks = [(lvl.price, lvl.size) for lvl in payload.asks]
            self._book.apply_book_snapshot(bids, asks)
            self._refresh_last_mid()
        elif kind == "price_change":
            assert isinstance(payload, PriceChangeEvent)
            for change in payload.changes:
                self._book.apply_price_change(change.side, change.price, change.size)
            self._refresh_last_mid()
        elif kind == "trade":
            assert isinstance(payload, LastTradePriceEvent)
            fills = self._book.process_trade(
                payload.price, payload.size, payload.side, payload.timestamp_ms
            )
            for oid, side, fill_size, qa_at_fill, time_in_book in fills:
                self._record_fill(
                    oid=oid,
                    side=side,
                    price=payload.price,
                    size=fill_size,
                    is_maker=True,
                    queue_ahead_at_fill=qa_at_fill,
                    time_in_book_ms=time_in_book,
                    ts_ms=payload.timestamp_ms,
                )
            self._refresh_last_mid()
        elif kind == "tick_size_change":
            assert isinstance(payload, TickSizeChangeEvent)
            self._book.tick = float(payload.new_tick_size)

    def _handle_place_arrival(self, p: _PlacePayload) -> None:
        # The order may have been cancelled before the arrival fired.
        if p.order_id not in self._pending_places:
            return
        del self._pending_places[p.order_id]

        if not p.post_only and _crosses_spread(self._book, p.side, p.price):
            opp = self._book.best_ask() if p.side == "bid" else self._book.best_bid()
            if opp is not None:
                fill_price, available = opp
                fill_size = min(p.size, available)
                self._record_fill(
                    oid=p.order_id,
                    side=p.side,
                    price=fill_price,
                    size=fill_size,
                    is_maker=False,
                    queue_ahead_at_fill=0.0,
                    time_in_book_ms=0,
                    ts_ms=self._now_ms,
                )
                remainder = p.size - fill_size
                if remainder > 0:
                    self._book.add_our_order(p.order_id, p.side, p.price, remainder, self._now_ms)
                    self._our_order_ids.add(p.order_id)
                    self._log_order_arrival(p.order_id, p, partial_after_taker=True)
                return

        self._book.add_our_order(p.order_id, p.side, p.price, p.size, self._now_ms)
        self._our_order_ids.add(p.order_id)
        self._log_order_arrival(p.order_id, p, partial_after_taker=False)

    def _handle_cancel_arrival(self, c: _CancelPayload) -> None:
        self._book.remove_our_order(c.order_id)
        self._our_order_ids.discard(c.order_id)
        self._pending_places.pop(c.order_id, None)
        self._log({"type": "order_cancel_arrival", "ts_ms": self._now_ms, "order_id": c.order_id})

    # ------------------------------------------------------------------
    # PnL accounting (mirrors Backtester._record_fill)
    # ------------------------------------------------------------------

    def _record_fill(
        self,
        *,
        oid: int,
        side: Side,
        price: float,
        size: float,
        is_maker: bool,
        queue_ahead_at_fill: float,
        time_in_book_ms: int,
        ts_ms: int,
    ) -> None:
        if is_maker:
            fee = 0.0
            rebate = maker_rebate(price, size, self._fee_category)
        else:
            fee = taker_fee(price, size, self._fee_category)
            rebate = 0.0

        sign = 1 if side == "bid" else -1
        self._cash += -sign * price * size + rebate - fee
        self._inventory += sign * size
        self._fees_paid += fee
        self._rebates_received += rebate

        fill = FillRecord(
            timestamp_ms=ts_ms,
            order_id=oid,
            side=side,
            price=price,
            size=size,
            is_maker=is_maker,
            fee_paid=fee,
            rebate_received=rebate,
            queue_ahead_at_fill=queue_ahead_at_fill,
            time_in_book_ms=time_in_book_ms,
        )
        self._fills.append(fill)
        # L2OrderBook.process_trade prunes fully-filled orders from
        # our_orders; mirror that into our_order_ids.
        if oid not in self._book.our_orders:
            self._our_order_ids.discard(oid)

        self._log(
            {
                "type": "fill",
                "ts_ms": ts_ms,
                "order_id": oid,
                "side": side,
                "price": price,
                "size": size,
                "is_maker": is_maker,
                "fee": fee,
                "rebate": rebate,
                "queue_ahead": queue_ahead_at_fill,
                "time_in_book_ms": time_in_book_ms,
            }
        )

    def _refresh_last_mid(self) -> None:
        mid = self._book.mid()
        if mid is not None:
            self._last_mid = mid

    # ------------------------------------------------------------------
    # Risk + heartbeat
    # ------------------------------------------------------------------

    def _tick_kill_switch(self) -> None:
        pnl = mark_to_mid(self._cash, self._inventory, self._last_mid)
        self._kill_switch.tick(
            now_s=time.time(),
            current_pnl=pnl,
            inventory=self._inventory,
        )

    async def _heartbeat_loop(self) -> None:
        """Periodic watchdog that lets the kill switch trip on a frozen feed."""
        while not self._stop:
            try:
                await asyncio.sleep(self._heartbeat_interval_s)
            except asyncio.CancelledError:
                return
            self._kill_switch.heartbeat_check(now_s=time.time())
            if self._kill_switch.halted:
                self._on_halt()
                if self._client is not None:
                    self._client.stop()
                return

    def _on_halt(self) -> None:
        """Cancel local resting orders, log once, and signal the WS client to stop.

        Idempotent. The first call cleans up state and writes a single
        ``halt`` record. Subsequent calls only re-assert ``client.stop()`` —
        no log spam, no redundant cleanup. This matters in practice because
        ``_on_halt`` can be re-entered both from :meth:`_heartbeat_loop`
        (periodic tick) and from :meth:`_process_market_event` (a new event
        arrived after halt while the WS was being closed), and on a live
        feed those re-entries can fire dozens of times within a second.
        """
        if self._halt_logged:
            if self._client is not None:
                self._client.stop()
            return

        for oid in list(self._our_order_ids):
            self._book.remove_our_order(oid)
        self._our_order_ids.clear()
        self._pending_places.clear()
        self._pending_arrivals.clear()

        self._log(
            {
                "type": "halt",
                "ts_ms": self._now_ms,
                "reason": self._kill_switch.halt_reason.value,
                "pnl": mark_to_mid(self._cash, self._inventory, self._last_mid),
                "inventory": self._inventory,
                "cash": self._cash,
                "fees_paid": self._fees_paid,
                "rebates_received": self._rebates_received,
            }
        )
        self._halt_logged = True

        if self._client is not None:
            self._client.stop()

    # ------------------------------------------------------------------
    # Heap + logging helpers
    # ------------------------------------------------------------------

    def _push_arrival(self, ts_ms: int, kind: str, payload: object) -> None:
        heapq.heappush(
            self._pending_arrivals,
            SimEvent(ts_ms, next(self._seq_counter), kind, payload),  # type: ignore[arg-type]
        )

    def _log_order_arrival(self, oid: int, p: _PlacePayload, *, partial_after_taker: bool) -> None:
        info = self._book.our_orders.get(oid)
        queue_ahead = info[3] if info else 0.0
        self._log(
            {
                "type": "order_arrival",
                "ts_ms": self._now_ms,
                "order_id": oid,
                "side": p.side,
                "price": p.price,
                "size": p.size,
                "post_only": p.post_only,
                "queue_ahead": queue_ahead,
                "partial_after_taker": partial_after_taker,
            }
        )

    def _log_pnl_snapshot(self) -> None:
        pnl = mark_to_mid(self._cash, self._inventory, self._last_mid)
        self._log(
            {
                "type": "pnl",
                "ts_ms": self._now_ms,
                "cash": self._cash,
                "inventory": self._inventory,
                "mid": self._last_mid,
                "pnl": pnl,
                "n_open_orders": len(self._book.our_orders),
                "n_pending_arrivals": len(self._pending_arrivals),
            }
        )

    def _log(self, record: dict[str, Any]) -> None:
        # wall_ts (epoch seconds) makes the log useful even when server
        # ts is stale or absent (e.g. ws_resync before any event lands).
        out = {"wall_ts": time.time(), **record}
        self._log_fh.write(json.dumps(out, separators=(",", ":")) + "\n")

    # ------------------------------------------------------------------
    # Read-only accessors for tests / diagnostics
    # ------------------------------------------------------------------

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def fills(self) -> list[FillRecord]:
        return list(self._fills)

    @property
    def fees_paid(self) -> float:
        return self._fees_paid

    @property
    def rebates_received(self) -> float:
        return self._rebates_received


# ----------------------------------------------------------------------
# Module-level helpers (duplicated from simulator.engine to avoid coupling
# the live runner to private engine internals — both are 3-line bodies).
# ----------------------------------------------------------------------


def _market_event_to_kind(ev: MarketEvent) -> str:
    if isinstance(ev, BookEvent):
        return "book"
    if isinstance(ev, PriceChangeEvent):
        return "price_change"
    if isinstance(ev, LastTradePriceEvent):
        return "trade"
    if isinstance(ev, TickSizeChangeEvent):
        return "tick_size_change"
    raise TypeError(f"unsupported market event type: {type(ev).__name__}")


def _crosses_spread(book: L2OrderBook, side: Side, price: float) -> bool:
    if side == "bid":
        best_ask = book.best_ask()
        return best_ask is not None and price >= best_ask[0]
    best_bid = book.best_bid()
    return best_bid is not None and price <= best_bid[0]


# ----------------------------------------------------------------------
# CLI entrypoint
# ----------------------------------------------------------------------


# Same parameter mapping as scripts/run_backtest._merge_calibrated_params.
# Kept here so the live runner does not depend on the scripts/ directory.
_PARAMS_BY_STRATEGY: dict[str, set[str]] = {
    "constant_spread": {"half_spread", "size", "tick"},
    "avellaneda_stoikov": {"gamma", "sigma", "kappa", "horizon_ms", "size"},
    "avellaneda_stoikov_with_signals": {
        "gamma",
        "sigma",
        "kappa",
        "horizon_ms",
        "size",
        "use_microprice",
        "alpha_beta",
        "ofi_window_s",
        "vpin_bucket_volume",
        "vpin_n_buckets",
        "vpin_max",
        "jump_schedule_ms",
        "pre_jump_withdraw_ms",
        "post_jump_resume_ms",
    },
    "glt": {"gamma", "sigma", "kappa", "A", "size"},
}


def _merge_calibrated_params(
    strategy_kind: str, strategy_params: dict[str, Any], calibrated: dict[str, Any]
) -> dict[str, Any]:
    """Map calibrated keys onto strategy parameter names, filtered by kind."""
    allowed = _PARAMS_BY_STRATEGY.get(strategy_kind, set())
    out = dict(strategy_params)
    mapping = {
        "sigma_per_sqrts": "sigma",
        "kappa": "kappa",
        "A_per_side": "A",
        "alpha_beta": "alpha_beta",
    }
    for src, dst in mapping.items():
        if src in calibrated and dst in allowed:
            out[dst] = calibrated[src]
    return out


def _default_log_path(root: Path, token_id: str) -> Path:
    """``{root}/{YYYY-MM-DD UTC}/{token_id}.jsonl`` — partitioning mirrors the WS capture writer."""
    from datetime import UTC, datetime

    date_str = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    return root / date_str / f"{token_id}.jsonl"


async def _run_cli(
    config_path: Path,
    log_root: Path,
    latency_ms: int,
    params_path: Path | None,
) -> None:
    """Wire AppConfig + KillSwitch + strategy + PaperTrader and run forever."""
    from ..config import load_config
    from ..simulator.latency import ConstantLatency
    from ..strategies.factory import build_strategy

    cfg = load_config(config_path)
    strat_params = dict(cfg.strategy.params)
    if params_path is not None:
        calibrated = json.loads(params_path.read_text())
        strat_params = _merge_calibrated_params(cfg.strategy.kind, strat_params, calibrated)

    strategy = build_strategy(cfg.strategy.kind, strat_params)
    kill_switch = KillSwitch(cfg.risk.to_limits())
    log_path = _default_log_path(log_root, cfg.market.token_id)

    trader = PaperTrader(
        token_id=cfg.market.token_id,
        tick=cfg.market.tick,
        fee_category=cfg.market.fee_category,
        strategy=strategy,
        kill_switch=kill_switch,
        latency=ConstantLatency(latency_ms),
        log_path=log_path,
    )

    loop = asyncio.get_running_loop()
    stop_evt = asyncio.Event()

    def _request_stop() -> None:
        trader.stop()
        stop_evt.set()

    for sig_name in ("SIGINT", "SIGTERM"):
        try:
            import signal

            loop.add_signal_handler(getattr(signal, sig_name), _request_stop)
        except (NotImplementedError, RuntimeError):
            pass

    runner = asyncio.create_task(trader.run())
    try:
        await asyncio.wait(
            {runner, asyncio.create_task(stop_evt.wait())},
            return_when=asyncio.FIRST_COMPLETED,
        )
    except KeyboardInterrupt:
        trader.stop()
    finally:
        if not runner.done():
            runner.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await runner
    logger.info(
        "paper trader exited: pnl=%.4f fills=%d fees=%.4f rebates=%.4f halted=%s reason=%s",
        mark_to_mid(trader.cash, trader.inventory, trader._last_mid),
        len(trader.fills),
        trader.fees_paid,
        trader.rebates_received,
        kill_switch.halted,
        kill_switch.halt_reason.value,
    )


def _build_cli_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m hft_pm.live.paper_trade",
        description="Paper-trade against the live Polymarket feed.",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--log-root",
        default="data/paper",
        help="Root for the JSONL paper-trade log (partitioned by date / token).",
    )
    parser.add_argument(
        "--latency-ms",
        type=int,
        default=50,
        help="Constant order-arrival latency in milliseconds.",
    )
    parser.add_argument(
        "--params",
        help="Optional calibrated-params JSON from scripts/calibrate_strategy.py.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main() -> None:
    args = _build_cli_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    asyncio.run(
        _run_cli(
            config_path=Path(args.config),
            log_root=Path(args.log_root),
            latency_ms=int(args.latency_ms),
            params_path=Path(args.params) if args.params else None,
        )
    )


if __name__ == "__main__":
    main()


__all__ = ["PaperTrader"]
