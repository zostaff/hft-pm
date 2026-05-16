"""Production Polymarket WebSocket market-channel client (docs §10.5).

Three additions over the minimal subscriber in docs §7:

1. Auto-reconnect with exponential backoff.
2. Heartbeat timeout watchdog (Polymarket's silent-freeze bug).
3. Per-asset ``hash`` tracking so gaps can be detected and a REST
   reconciliation is triggered after each (re)subscribe.

The class accepts a connector callable so tests can substitute a local
asyncio echo server without monkey-patching ``websockets``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

import websockets

logger = logging.getLogger(__name__)


# Type for the per-event sink. recv_ts_ms is set on the receive side; the
# raw server dict is forwarded verbatim so the sink can persist it untouched.
EventCallback = Callable[[dict[str, Any], int], Awaitable[None]]
DisconnectCallback = Callable[[str], Awaitable[None]]
ResyncCallback = Callable[[], Awaitable[None]]


class _WSLike(Protocol):
    """Minimal subset of ``websockets`` connections we need.

    Declared so the fake-WS tests can pass any object that supports the
    same surface area.
    """

    async def send(self, data: str) -> None: ...
    async def close(self) -> None: ...
    def __aiter__(self) -> Any: ...
    def __anext__(self) -> Awaitable[str]: ...


Connector = Callable[[str], Any]
"""Async context manager factory returning a ``_WSLike``. Defaults to ``websockets.connect``."""


def _default_connector(url: str) -> Any:
    # ``ping_interval=15`` / ``ping_timeout=10`` enable transport-level keepalive
    # in addition to our application-level heartbeat watchdog. See docs §10.5.
    return websockets.connect(url, ping_interval=15, ping_timeout=10, close_timeout=5)


async def _noop_disconnect(_reason: str) -> None:
    return None


async def _noop_resync() -> None:
    return None


class PolymarketWSClient:
    """Polymarket market-channel client (docs §10.5).

    Parameters
    ----------
    asset_ids:
        Token ids to subscribe to on connect / reconnect.
    on_event:
        Coroutine called with ``(raw_event_dict, recv_ts_ms)`` per event.
        recv_ts_ms is local epoch-ms at read time.
    on_disconnect:
        Optional notifier for connection drops, heartbeat timeouts, and
        unexpected exceptions. Used to log and trigger external bookkeeping.
    on_resync:
        Called once per successful subscribe. The live system uses this to
        kick off a REST snapshot reconciliation so the local book is rebuilt
        from authoritative state after every reconnect.
    connector:
        Async context manager factory. Defaults to ``websockets.connect``.
        Override in tests to inject a fake server.
    """

    URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    HEARTBEAT_TIMEOUT_S = 30.0
    HEARTBEAT_POLL_S = 5.0
    INITIAL_BACKOFF_S = 1.0
    MAX_BACKOFF_S = 60.0

    def __init__(
        self,
        asset_ids: list[str],
        on_event: EventCallback,
        on_disconnect: DisconnectCallback | None = None,
        on_resync: ResyncCallback | None = None,
        *,
        connector: Connector | None = None,
        url: str | None = None,
    ) -> None:
        if not asset_ids:
            raise ValueError("asset_ids must be non-empty")
        self.asset_ids = list(asset_ids)
        self.on_event = on_event
        self.on_disconnect = on_disconnect or _noop_disconnect
        self.on_resync = on_resync or _noop_resync
        self._connector = connector or _default_connector
        self._url = url or self.URL
        self.last_msg_ts: float = 0.0
        self.last_hash_by_asset: dict[str, str] = {}
        self._stop = False

    async def run(self) -> None:
        """Main loop. Reconnects forever with exponential backoff until :meth:`stop`."""
        backoff = self.INITIAL_BACKOFF_S
        while not self._stop:
            try:
                async with self._connector(self._url) as ws:
                    await self._subscribe(ws)
                    backoff = self.INITIAL_BACKOFF_S  # reset on successful subscribe
                    await self._read_loop(ws)
            except (TimeoutError, websockets.ConnectionClosed, OSError) as e:
                await self.on_disconnect(f"connection lost: {e}")
            except Exception as e:
                await self.on_disconnect(f"unexpected: {e!r}")

            if self._stop:
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self.MAX_BACKOFF_S)

    async def _subscribe(self, ws: _WSLike) -> None:
        await ws.send(json.dumps({"type": "market", "assets_ids": self.asset_ids}))
        self.last_msg_ts = time.time()
        await self.on_resync()

    async def _read_loop(self, ws: _WSLike) -> None:
        """Drain messages until the connection drops or the watchdog fires."""

        async def watchdog() -> None:
            while True:
                await asyncio.sleep(self.HEARTBEAT_POLL_S)
                if time.time() - self.last_msg_ts > self.HEARTBEAT_TIMEOUT_S:
                    await self.on_disconnect("heartbeat timeout (silent freeze)")
                    await ws.close()
                    return

        wd_task = asyncio.create_task(watchdog())
        try:
            async for raw in ws:
                recv_ts = int(time.time() * 1000)
                self.last_msg_ts = time.time()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    logger.debug("dropping non-JSON message", extra={"raw": raw})
                    continue
                events = msg if isinstance(msg, list) else [msg]
                for ev in events:
                    if not isinstance(ev, dict):
                        continue
                    await self._handle_event(ev, recv_ts)
        finally:
            wd_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await wd_task

    async def _handle_event(self, ev: dict[str, Any], recv_ts: int) -> None:
        """Track per-asset hash and forward the raw event to the sink."""
        asset = ev.get("asset_id")
        new_hash = ev.get("hash")
        if isinstance(asset, str) and isinstance(new_hash, str):
            # The public feed is not strictly hash-chained; periodic REST
            # reconciliation (see docs §10.5) is the authoritative safety net.
            self.last_hash_by_asset[asset] = new_hash
        await self.on_event(ev, recv_ts)

    def stop(self) -> None:
        """Request shutdown after the next read iteration."""
        self._stop = True


def _build_cli_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m hft_pm.data.polymarket_ws",
        description="Capture Polymarket market-channel events to JSONL.",
    )
    parser.add_argument(
        "--assets",
        required=True,
        help="Comma-separated Polymarket token ids to subscribe to.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output root. JSONL is written to {out}/{YYYY-MM-DD}/{asset}.jsonl.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


async def _run_cli(asset_ids: list[str], out_root: str) -> None:
    """CLI entrypoint: wire the WS client to a JsonlWriter and run forever."""
    from pathlib import Path

    from .writer import JsonlWriter

    writer = JsonlWriter(Path(out_root))
    latencies_ms: list[int] = []
    n_dropped_non_market = 0

    async def on_event(ev: dict[str, Any], recv_ts_ms: int) -> None:
        nonlocal n_dropped_non_market
        # Polymarket's market channel occasionally sends control messages
        # without ``asset_id`` (subscription acks, server pings). The writer
        # requires asset_id, so filter them here rather than letting the
        # exception kill the run loop.
        if not isinstance(ev.get("asset_id"), str):
            n_dropped_non_market += 1
            return
        try:
            writer.write(ev, recv_ts_ms)
        except (OSError, ValueError) as e:
            logger.warning("writer rejected event: %s", e)
            return
        server_ts = ev.get("timestamp")
        if isinstance(server_ts, str) and server_ts.isdigit():
            latencies_ms.append(recv_ts_ms - int(server_ts))

    async def on_disconnect(reason: str) -> None:
        logger.warning("disconnect: %s", reason)

    async def on_resync() -> None:
        logger.info("(re)subscribed; REST reconciliation expected at this point")

    client = PolymarketWSClient(asset_ids, on_event, on_disconnect, on_resync)

    loop = asyncio.get_running_loop()
    stop_evt = asyncio.Event()

    def _request_stop() -> None:
        client.stop()
        stop_evt.set()

    for sig_name in ("SIGINT", "SIGTERM"):
        try:
            import signal

            loop.add_signal_handler(getattr(signal, sig_name), _request_stop)
        except (NotImplementedError, RuntimeError):
            # add_signal_handler is unsupported on Windows; SIGINT still
            # works via KeyboardInterrupt in the await below.
            pass

    runner = asyncio.create_task(client.run())
    try:
        await stop_evt.wait()
    except KeyboardInterrupt:
        client.stop()
    finally:
        runner.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await runner
        writer.close()
        _print_latency_profile(latencies_ms)


def _print_latency_profile(latencies_ms: list[int]) -> None:
    if not latencies_ms:
        logger.info("no events received; nothing to summarise")
        return
    samples = sorted(latencies_ms)
    n = len(samples)

    def pct(p: float) -> int:
        # Linear-interpolation percentile; tiny samples are fine.
        if n == 1:
            return samples[0]
        k = (n - 1) * p
        lo = int(k)
        hi = min(lo + 1, n - 1)
        frac = k - lo
        return int(samples[lo] * (1 - frac) + samples[hi] * frac)

    logger.info(
        "latency profile (recv_ts - server_ts, ms): n=%d  median=%d  p95=%d  p99=%d  max=%d",
        n,
        pct(0.5),
        pct(0.95),
        pct(0.99),
        samples[-1],
    )


def main() -> None:
    args = _build_cli_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    asset_ids = [a.strip() for a in args.assets.split(",") if a.strip()]
    if not asset_ids:
        raise SystemExit("--assets produced an empty list")
    asyncio.run(_run_cli(asset_ids, args.out))


if __name__ == "__main__":
    main()


__all__ = ["PolymarketWSClient"]
