"""Unit tests for hft_pm.data.replay."""

from __future__ import annotations

import calendar
import json
from datetime import date
from pathlib import Path

import pytest

from hft_pm.data.replay import Replay
from hft_pm.data.writer import JsonlWriter

ASSET_A = "A"
ASSET_B = "B"
DATE_1 = date(2026, 5, 16)
DATE_2 = date(2026, 5, 17)

DAY1_BASE_MS = calendar.timegm((2026, 5, 16, 0, 0, 0)) * 1000
DAY2_BASE_MS = calendar.timegm((2026, 5, 17, 0, 0, 0)) * 1000


def _make_book(asset: str, ts_ms: int) -> dict:
    return {
        "event_type": "book",
        "asset_id": asset,
        "market": "0xdeadbeef",
        "timestamp": str(ts_ms),
        "hash": f"hash-{asset}-{ts_ms}",
        "bids": [{"price": "0.5", "size": "10"}],
        "asks": [{"price": "0.51", "size": "10"}],
    }


def _seed_dataset(root: Path) -> None:
    writer = JsonlWriter(root)
    # Asset A, day 1: 3 in-order events.
    for i in range(3):
        writer.write(_make_book(ASSET_A, DAY1_BASE_MS + i * 1000), DAY1_BASE_MS + i * 1000 + 50)
    # Asset B, day 1: 2 events interleaved with A's by server timestamp.
    writer.write(_make_book(ASSET_B, DAY1_BASE_MS + 500), DAY1_BASE_MS + 600)
    writer.write(_make_book(ASSET_B, DAY1_BASE_MS + 2500), DAY1_BASE_MS + 2600)
    # Asset A, day 2: 1 event.
    writer.write(_make_book(ASSET_A, DAY2_BASE_MS), DAY2_BASE_MS + 25)
    writer.close()


def test_replay_yields_global_timestamp_order(tmp_path: Path) -> None:
    _seed_dataset(tmp_path)
    replay = Replay(tmp_path, [ASSET_A, ASSET_B], (DATE_1, DATE_2))
    events = list(replay)
    timestamps = [e.timestamp_ms for e in events]
    assert timestamps == sorted(timestamps), "events must be globally non-decreasing"
    assert len(events) == 6


def test_replay_filters_to_requested_assets(tmp_path: Path) -> None:
    _seed_dataset(tmp_path)
    replay = Replay(tmp_path, [ASSET_A], (DATE_1, DATE_2))
    assets = {e.asset_id for e in replay}
    assert assets == {ASSET_A}


def test_replay_respects_date_range(tmp_path: Path) -> None:
    _seed_dataset(tmp_path)
    replay = Replay(tmp_path, [ASSET_A, ASSET_B], (DATE_1, DATE_1))
    events = list(replay)
    assert all(e.timestamp_ms < DAY2_BASE_MS for e in events)
    assert len(events) == 5


def test_verify_no_gaps_clean_dataset(tmp_path: Path) -> None:
    _seed_dataset(tmp_path)
    replay = Replay(tmp_path, [ASSET_A, ASSET_B], (DATE_1, DATE_2))
    report = replay.verify_no_gaps()
    assert report.n_events == 6
    assert report.n_out_of_order == 0
    assert report.n_parse_errors == 0
    assert report.gaps == []


def test_verify_no_gaps_detects_regression(tmp_path: Path) -> None:
    """Inject an out-of-order event on asset A and confirm it is reported."""
    writer = JsonlWriter(tmp_path)
    writer.write(_make_book(ASSET_A, DAY1_BASE_MS + 1000), DAY1_BASE_MS + 1050)
    writer.write(_make_book(ASSET_A, DAY1_BASE_MS + 2000), DAY1_BASE_MS + 2050)
    # Regression: timestamp goes backwards on the same asset/file.
    writer.write(_make_book(ASSET_A, DAY1_BASE_MS + 1500), DAY1_BASE_MS + 2100)
    writer.close()

    replay = Replay(tmp_path, [ASSET_A], (DATE_1, DATE_1))
    report = replay.verify_no_gaps()
    assert report.n_out_of_order == 1
    assert len(report.gaps) == 1
    gap = report.gaps[0]
    assert gap.asset_id == ASSET_A
    assert gap.previous_timestamp_ms == DAY1_BASE_MS + 2000
    assert gap.current_timestamp_ms == DAY1_BASE_MS + 1500


def test_verify_no_gaps_counts_parse_errors(tmp_path: Path) -> None:
    # Write a valid event, then a malformed line, then a structurally bad envelope.
    writer = JsonlWriter(tmp_path)
    writer.write(_make_book(ASSET_A, DAY1_BASE_MS), DAY1_BASE_MS + 10)
    writer.close()

    path = tmp_path / DATE_1.strftime("%Y-%m-%d") / f"{ASSET_A}.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{not json\n")
        fh.write(json.dumps({"recv_ts_ms": "not_an_int", "event": {}}) + "\n")

    replay = Replay(tmp_path, [ASSET_A], (DATE_1, DATE_1))
    report = replay.verify_no_gaps()
    assert report.n_events == 1
    assert report.n_parse_errors == 2


def test_verify_no_gaps_counts_unknown_event_types(tmp_path: Path) -> None:
    writer = JsonlWriter(tmp_path)
    writer.write(_make_book(ASSET_A, DAY1_BASE_MS), DAY1_BASE_MS + 10)
    writer.close()

    path = tmp_path / DATE_1.strftime("%Y-%m-%d") / f"{ASSET_A}.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "recv_ts_ms": DAY1_BASE_MS + 20,
                    "event": {
                        "event_type": "mystery_future_event",
                        "asset_id": ASSET_A,
                        "timestamp": str(DAY1_BASE_MS + 5),
                    },
                }
            )
            + "\n"
        )

    replay = Replay(tmp_path, [ASSET_A], (DATE_1, DATE_1))
    report = replay.verify_no_gaps()
    assert report.n_events == 1
    assert report.n_unknown_event_types == 1


def test_replay_handles_empty_directory(tmp_path: Path) -> None:
    replay = Replay(tmp_path, [ASSET_A], (DATE_1, DATE_2))
    assert list(replay) == []
    report = replay.verify_no_gaps()
    assert report.n_events == 0


def test_replay_rejects_inverted_date_range(tmp_path: Path) -> None:
    replay = Replay(tmp_path, [ASSET_A], (DATE_2, DATE_1))
    with pytest.raises(ValueError):
        list(replay)
