"""Unit tests for hft_pm.data.writer."""

from __future__ import annotations

import calendar
import json
from pathlib import Path

import pytest

from hft_pm.data.writer import JsonlWriter


def _utc_ms(y: int, mo: int, d: int, h: int = 0, mi: int = 0, s: int = 0, ms: int = 0) -> int:
    """Build an epoch-ms timestamp from UTC wall-clock components."""
    return calendar.timegm((y, mo, d, h, mi, s)) * 1000 + ms


DAY1_MIDNIGHT_MS = _utc_ms(2026, 5, 16, 0, 0, 0)
DAY1_EOD_MS = _utc_ms(2026, 5, 16, 23, 59, 59, 999)
DAY2_MIDNIGHT_MS = _utc_ms(2026, 5, 17, 0, 0, 0)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_writer_partitions_by_utc_date_and_asset(tmp_path: Path) -> None:
    writer = JsonlWriter(tmp_path)
    writer.write({"asset_id": "A", "event_type": "book"}, DAY1_MIDNIGHT_MS)
    writer.write({"asset_id": "B", "event_type": "book"}, DAY1_EOD_MS)
    writer.close()

    a_path = tmp_path / "2026-05-16" / "A.jsonl"
    b_path = tmp_path / "2026-05-16" / "B.jsonl"
    assert a_path.is_file()
    assert b_path.is_file()
    a_records = _read_jsonl(a_path)
    assert a_records == [
        {"recv_ts_ms": DAY1_MIDNIGHT_MS, "event": {"asset_id": "A", "event_type": "book"}}
    ]


def test_writer_rotates_across_utc_midnight(tmp_path: Path) -> None:
    writer = JsonlWriter(tmp_path)
    writer.write({"asset_id": "A", "event_type": "book"}, DAY1_EOD_MS)
    writer.write({"asset_id": "A", "event_type": "book"}, DAY2_MIDNIGHT_MS)
    writer.close()

    day1 = tmp_path / "2026-05-16" / "A.jsonl"
    day2 = tmp_path / "2026-05-17" / "A.jsonl"
    assert len(_read_jsonl(day1)) == 1
    assert len(_read_jsonl(day2)) == 1


def test_writer_flushes_per_write(tmp_path: Path) -> None:
    # We require flush-per-write so capture survives SIGKILL. Read the file
    # without closing the writer to verify the line is on disk already.
    writer = JsonlWriter(tmp_path)
    writer.write({"asset_id": "A", "event_type": "book"}, DAY2_MIDNIGHT_MS)
    path = tmp_path / "2026-05-17" / "A.jsonl"
    assert path.read_text().endswith("\n")
    writer.close()


def test_writer_context_manager_closes_handles(tmp_path: Path) -> None:
    with JsonlWriter(tmp_path) as writer:
        writer.write({"asset_id": "A", "event_type": "book"}, DAY2_MIDNIGHT_MS)
    # No assertion needed beyond not raising; the contextmanager exit
    # calls close(), which is idempotent.


def test_writer_rejects_missing_asset_id(tmp_path: Path) -> None:
    writer = JsonlWriter(tmp_path)
    with pytest.raises(ValueError):
        writer.write({"event_type": "book"}, DAY2_MIDNIGHT_MS)
    writer.close()


def test_writer_requires_recv_ts_without_clock(tmp_path: Path) -> None:
    writer = JsonlWriter(tmp_path)
    with pytest.raises(ValueError):
        writer.write({"asset_id": "A", "event_type": "book"})
    writer.close()


def test_writer_uses_clock_when_recv_ts_omitted(tmp_path: Path) -> None:
    writer = JsonlWriter(tmp_path, clock=lambda: DAY2_MIDNIGHT_MS)
    writer.write({"asset_id": "A", "event_type": "book"})
    writer.close()
    path = tmp_path / "2026-05-17" / "A.jsonl"
    assert len(_read_jsonl(path)) == 1
