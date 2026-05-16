"""Append-only JSONL sink for captured market events.

Partitioned by UTC date and asset id::

    {root}/{YYYY-MM-DD}/{asset_id}.jsonl

Each line is a JSON object::

    {"recv_ts_ms": <int>, "event": <raw server dict>}

We store the raw server dict (not the parsed Pydantic model) so old captures
survive future schema additions. Pydantic parsing happens at replay time.

Files are kept open between writes and flushed after every line. If a write
crosses UTC midnight, the writer transparently rotates to the new date.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import IO, Any


def _utc_date_str(recv_ts_ms: int) -> str:
    return datetime.fromtimestamp(recv_ts_ms / 1000.0, tz=UTC).strftime("%Y-%m-%d")


class JsonlWriter:
    """Per-asset, per-UTC-date JSONL appender.

    Parameters
    ----------
    root:
        Directory under which dated subdirectories are created. Created on
        first write if absent.
    clock:
        Optional override that returns the current epoch in milliseconds.
        Tests use it to simulate the UTC-midnight rollover; production code
        always passes ``recv_ts_ms`` explicitly, so the clock is only used
        when the caller omits it (which production code does not).
    """

    def __init__(
        self,
        root: Path,
        *,
        clock: Callable[[], int] | None = None,
    ) -> None:
        self._root = Path(root)
        self._clock = clock
        self._handles: dict[tuple[str, str], IO[str]] = {}

    def write(self, event: dict[str, Any], recv_ts_ms: int | None = None) -> None:
        """Append one event to the appropriate dated file.

        ``recv_ts_ms`` determines partitioning. If omitted, ``clock`` is used.
        Production callers always pass ``recv_ts_ms`` from the WS read loop
        because that is the timestamp under which the event is logged.
        """
        if recv_ts_ms is None:
            if self._clock is None:
                raise ValueError("recv_ts_ms required when no clock is configured")
            recv_ts_ms = self._clock()

        asset_id = event.get("asset_id")
        if not isinstance(asset_id, str):
            raise ValueError("event missing string 'asset_id'")

        date_str = _utc_date_str(recv_ts_ms)
        handle = self._handle_for(date_str, asset_id)
        record = {"recv_ts_ms": recv_ts_ms, "event": event}
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        handle.flush()

    def _handle_for(self, date_str: str, asset_id: str) -> IO[str]:
        key = (date_str, asset_id)
        handle = self._handles.get(key)
        if handle is not None:
            return handle
        dir_path = self._root / date_str
        dir_path.mkdir(parents=True, exist_ok=True)
        path = dir_path / f"{asset_id}.jsonl"
        # ``buffering=1`` gives line-buffered text mode; combined with
        # ``flush()`` per write, capture survives an OS-level kill. The
        # handle is owned by JsonlWriter and closed via close()/__exit__,
        # so SIM115's "use a context manager" doesn't apply.
        new_handle = open(path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115
        self._handles[key] = new_handle
        return new_handle

    def close(self) -> None:
        """Close all open file handles. Safe to call multiple times."""
        for handle in self._handles.values():
            try:
                handle.flush()
                os.fsync(handle.fileno())
            except OSError:
                # Best-effort durability; do not mask close errors below.
                pass
            handle.close()
        self._handles.clear()

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


__all__ = ["JsonlWriter"]
