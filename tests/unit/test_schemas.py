"""Unit tests for hft_pm.data.schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hft_pm.data.schemas import (
    BookEvent,
    LastTradePriceEvent,
    PriceChangeEvent,
    TickSizeChangeEvent,
    UnknownEventTypeError,
    parse_event,
)

ASSET = "12345"
MARKET = "0xdeadbeef"
RECV_TS = 1_715_000_000_500


def _wire_book() -> dict:
    return {
        "event_type": "book",
        "asset_id": ASSET,
        "market": MARKET,
        "timestamp": "1715000000000",
        "hash": "abc",
        "bids": [{"price": "0.42", "size": "100"}],
        "asks": [{"price": "0.43", "size": "75"}],
    }


def _wire_price_change() -> dict:
    return {
        "event_type": "price_change",
        "asset_id": ASSET,
        "market": MARKET,
        "timestamp": "1715000000001",
        "hash": "def",
        "changes": [
            {"price": "0.42", "side": "BUY", "size": "120"},
            {"price": "0.43", "side": "SELL", "size": "0"},
        ],
    }


def _wire_last_trade() -> dict:
    return {
        "event_type": "last_trade_price",
        "asset_id": ASSET,
        "market": MARKET,
        "timestamp": "1715000000002",
        "price": "0.425",
        "size": "10",
        "side": "BUY",
        "fee_rate_bps": "0",
    }


def _wire_tick_size() -> dict:
    return {
        "event_type": "tick_size_change",
        "asset_id": ASSET,
        "market": MARKET,
        "timestamp": "1715000000003",
        "old_tick_size": "0.01",
        "new_tick_size": "0.001",
    }


def test_parse_book_event() -> None:
    ev = parse_event(_wire_book(), RECV_TS)
    assert isinstance(ev, BookEvent)
    assert ev.asset_id == ASSET
    assert ev.timestamp_ms == 1_715_000_000_000
    assert ev.recv_ts_ms == RECV_TS
    assert ev.bids[0].price == 0.42
    assert ev.bids[0].size == 100.0


def test_parse_price_change_event() -> None:
    ev = parse_event(_wire_price_change(), RECV_TS)
    assert isinstance(ev, PriceChangeEvent)
    assert len(ev.changes) == 2
    assert ev.changes[0].side == "BUY"
    assert ev.changes[1].size == 0.0


def test_parse_last_trade_event() -> None:
    ev = parse_event(_wire_last_trade(), RECV_TS)
    assert isinstance(ev, LastTradePriceEvent)
    assert ev.price == 0.425
    assert ev.side == "BUY"


def test_parse_tick_size_event() -> None:
    ev = parse_event(_wire_tick_size(), RECV_TS)
    assert isinstance(ev, TickSizeChangeEvent)
    assert ev.old_tick_size == "0.01"
    assert ev.new_tick_size == "0.001"


def test_parse_event_does_not_mutate_caller_dict() -> None:
    raw = _wire_book()
    snapshot = dict(raw)
    parse_event(raw, RECV_TS)
    assert raw == snapshot, "parse_event must leave the raw dict untouched"


def test_unknown_event_type_raises() -> None:
    with pytest.raises(UnknownEventTypeError):
        parse_event({"event_type": "mystery", "asset_id": ASSET, "timestamp": "0"}, RECV_TS)


def test_missing_timestamp_raises_validation_error() -> None:
    bad = _wire_book()
    del bad["timestamp"]
    with pytest.raises(ValidationError):
        parse_event(bad, RECV_TS)


def test_dispatcher_routes_by_event_type() -> None:
    # The dispatcher selects model by event_type; payload fields not relevant
    # to that model are tolerated (extra='allow'). Guards against refactors
    # that try to choose model class from payload shape instead.
    raw = _wire_book()
    raw["event_type"] = "price_change"
    raw["changes"] = []  # legal for a PriceChangeEvent
    ev = parse_event(raw, RECV_TS)
    assert isinstance(ev, PriceChangeEvent)
