"""Deterministic replay of captured Polymarket events.

Reads the JSONL files produced by :class:`hft_pm.data.writer.JsonlWriter`,
parses each line into a typed :class:`MarketEvent`, and yields events in
global timestamp order across all (asset, date) shards via k-way heap merge.

The replay also exposes :meth:`Replay.verify_no_gaps`, which scans the same
data and reports out-of-order events per asset, parse errors, and event
counts. This is the Phase 1 acceptance check.
"""

from __future__ import annotations

import heapq
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from .schemas import MarketEvent, UnknownEventTypeError, parse_event


@dataclass(frozen=True)
class GapReport:
    """Details of one timestamp regression observed inside an asset's tape."""

    asset_id: str
    file_path: str
    line_number: int
    previous_timestamp_ms: int
    current_timestamp_ms: int


@dataclass
class ReplayReport:
    """Result of :meth:`Replay.verify_no_gaps`."""

    n_events: int = 0
    n_parse_errors: int = 0
    n_unknown_event_types: int = 0
    n_out_of_order: int = 0
    gaps: list[GapReport] = field(default_factory=list)


def _iter_dates(start: date, end: date) -> Iterator[date]:
    if end < start:
        raise ValueError(f"end date {end} precedes start {start}")
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _shards(root: Path, assets: list[str], date_range: tuple[date, date]) -> list[Path]:
    """Return existing per-(date, asset) JSONL paths within the requested range."""
    start, end = date_range
    paths: list[Path] = []
    for d in _iter_dates(start, end):
        dir_path = root / d.strftime("%Y-%m-%d")
        if not dir_path.is_dir():
            continue
        for asset_id in assets:
            path = dir_path / f"{asset_id}.jsonl"
            if path.is_file():
                paths.append(path)
    return paths


def _decode_line(raw_line: str) -> tuple[int, dict[str, Any]]:
    """Decode one JSONL line into (recv_ts_ms, raw event dict).

    Raises
    ------
    json.JSONDecodeError
        On malformed JSON.
    ValueError
        On structurally wrong envelopes.
    """
    record = json.loads(raw_line)
    if not isinstance(record, dict):
        raise ValueError("envelope must be a JSON object")
    recv_ts_ms = record.get("recv_ts_ms")
    event = record.get("event")
    if not isinstance(recv_ts_ms, int) or not isinstance(event, dict):
        raise ValueError("envelope missing recv_ts_ms or event")
    return recv_ts_ms, event


class Replay:
    """Deterministic timestamp-ordered replay over captured JSONL shards.

    Parameters
    ----------
    root:
        Directory passed to :class:`JsonlWriter` during capture.
    assets:
        Token ids to replay. Other shards in ``root`` are ignored.
    date_range:
        Inclusive ``(start, end)`` range of UTC capture dates to scan.
    """

    def __init__(
        self,
        root: Path,
        assets: list[str],
        date_range: tuple[date, date],
    ) -> None:
        self._root = Path(root)
        self._assets = list(assets)
        self._date_range = date_range

    def __iter__(self) -> Iterator[MarketEvent]:
        """Yield events in global timestamp order, parsing each lazily."""
        shards = _shards(self._root, self._assets, self._date_range)
        if not shards:
            return

        # heap entries: (timestamp_ms, recv_ts_ms, tie_seq, parsed_event, gen)
        # tie_seq disambiguates events with identical timestamps from different
        # shards in a stable, reproducible way.
        heap: list[tuple[int, int, int, MarketEvent, Iterator[MarketEvent]]] = []
        tie_seq = 0
        for path in shards:
            gen = self._iter_shard(path)
            try:
                ev = next(gen)
            except StopIteration:
                continue
            heapq.heappush(heap, (ev.timestamp_ms, ev.recv_ts_ms, tie_seq, ev, gen))
            tie_seq += 1

        while heap:
            _, _, _, ev, gen = heapq.heappop(heap)
            yield ev
            try:
                nxt = next(gen)
            except StopIteration:
                continue
            heapq.heappush(heap, (nxt.timestamp_ms, nxt.recv_ts_ms, tie_seq, nxt, gen))
            tie_seq += 1

    def _iter_shard(self, path: Path) -> Iterator[MarketEvent]:
        """Yield parsed events from one JSONL shard, skipping bad lines."""
        with path.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                if not raw_line.strip():
                    continue
                try:
                    recv_ts_ms, event_dict = _decode_line(raw_line)
                    parsed = parse_event(event_dict, recv_ts_ms)
                except (json.JSONDecodeError, ValueError, UnknownEventTypeError):
                    # Bad lines surface in verify_no_gaps(); the iterator
                    # skips them so a corrupt line does not abort replay.
                    continue
                yield parsed

    def verify_no_gaps(self) -> ReplayReport:
        """Scan every shard once and return a structural integrity report.

        Returns counts of total events, JSON parse errors, unknown event
        types, and per-asset timestamp regressions. Acceptance for Phase 1
        is ``n_out_of_order == 0`` and ``n_parse_errors == 0``.
        """
        report = ReplayReport()
        for path in _shards(self._root, self._assets, self._date_range):
            asset_id = path.stem
            prev_ts: int | None = None
            with path.open("r", encoding="utf-8") as fh:
                for line_number, raw_line in enumerate(fh, start=1):
                    if not raw_line.strip():
                        continue
                    try:
                        recv_ts_ms, event_dict = _decode_line(raw_line)
                    except (json.JSONDecodeError, ValueError):
                        report.n_parse_errors += 1
                        continue
                    try:
                        parsed = parse_event(event_dict, recv_ts_ms)
                    except UnknownEventTypeError:
                        report.n_unknown_event_types += 1
                        continue
                    except ValueError:
                        report.n_parse_errors += 1
                        continue

                    report.n_events += 1
                    if prev_ts is not None and parsed.timestamp_ms < prev_ts:
                        report.n_out_of_order += 1
                        report.gaps.append(
                            GapReport(
                                asset_id=asset_id,
                                file_path=str(path),
                                line_number=line_number,
                                previous_timestamp_ms=prev_ts,
                                current_timestamp_ms=parsed.timestamp_ms,
                            )
                        )
                    prev_ts = parsed.timestamp_ms
        return report


__all__ = ["GapReport", "Replay", "ReplayReport"]
