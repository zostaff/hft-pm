"""Pydantic v2 models for Polymarket market-channel events.

The market channel publishes four event kinds (see docs §7 and §10.5):

- ``book``              full L2 snapshot for an asset
- ``price_change``      incremental updates (one or more price levels)
- ``last_trade_price``  recent trade print
- ``tick_size_change``  exchange-side tick size change

Polymarket sends prices and sizes as strings on the wire. We parse them to
floats at the schema boundary so downstream code never has to think about it.

The ``parse_event`` dispatcher is the only intended entry point for turning a
raw server dict into a typed model. Storage is intentionally untyped (raw
dict + ``recv_ts_ms``) so old captures survive schema additions.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Side = Literal["BUY", "SELL"]


class _Base(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)


class PriceLevel(_Base):
    """One price level in a book snapshot."""

    price: float
    size: float

    @field_validator("price", "size", mode="before")
    @classmethod
    def _coerce_numeric(cls, v: Any) -> Any:
        return float(v) if isinstance(v, str) else v


class PriceLevelChange(_Base):
    """One incremental level update inside a ``price_change`` event."""

    price: float
    side: Side
    size: float

    @field_validator("price", "size", mode="before")
    @classmethod
    def _coerce_numeric(cls, v: Any) -> Any:
        return float(v) if isinstance(v, str) else v


class MarketEvent(_Base):
    """Common fields shared by every market-channel event."""

    event_type: str
    asset_id: str
    market: str | None = None
    timestamp_ms: int = Field(
        description="Server-side wall-clock timestamp in ms (parsed from 'timestamp')."
    )
    recv_ts_ms: int = Field(description="Local wall-clock receive time in ms.")
    hash: str | None = None

    @field_validator("timestamp_ms", mode="before")
    @classmethod
    def _coerce_ts(cls, v: Any) -> Any:
        # Polymarket sends 'timestamp' as a decimal string of ms since epoch.
        if isinstance(v, str):
            return int(v)
        return v


class BookEvent(MarketEvent):
    event_type: Literal["book"] = "book"
    bids: list[PriceLevel] = Field(default_factory=list)
    asks: list[PriceLevel] = Field(default_factory=list)


class PriceChangeEvent(MarketEvent):
    event_type: Literal["price_change"] = "price_change"
    changes: list[PriceLevelChange] = Field(default_factory=list)


class LastTradePriceEvent(MarketEvent):
    event_type: Literal["last_trade_price"] = "last_trade_price"
    price: float
    size: float
    side: Side
    fee_rate_bps: float | None = None

    @field_validator("price", "size", "fee_rate_bps", mode="before")
    @classmethod
    def _coerce_numeric(cls, v: Any) -> Any:
        return float(v) if isinstance(v, str) else v


class TickSizeChangeEvent(MarketEvent):
    event_type: Literal["tick_size_change"] = "tick_size_change"
    old_tick_size: str
    new_tick_size: str


# Mapping from the wire's ``event_type`` discriminator to the parsed model.
_REGISTRY: dict[str, type[MarketEvent]] = {
    "book": BookEvent,
    "price_change": PriceChangeEvent,
    "last_trade_price": LastTradePriceEvent,
    "tick_size_change": TickSizeChangeEvent,
}


class UnknownEventTypeError(ValueError):
    """Raised when the server sends an event_type we don't have a model for."""


def parse_event(raw: dict[str, Any], recv_ts_ms: int) -> MarketEvent:
    """Parse one raw server dict into the appropriate typed event.

    The server's ``timestamp`` field (string ms since epoch) is renamed to
    ``timestamp_ms`` before validation. ``recv_ts_ms`` is injected so every
    event carries both clocks.

    Raises
    ------
    UnknownEventTypeError
        If ``raw["event_type"]`` is missing or not in the registry.
    pydantic.ValidationError
        If a known event fails validation.
    """
    event_type = raw.get("event_type")
    if not isinstance(event_type, str) or event_type not in _REGISTRY:
        raise UnknownEventTypeError(f"unknown event_type: {event_type!r}")

    # Don't mutate the caller's dict — capture stores it verbatim.
    payload = dict(raw)
    if "timestamp" in payload and "timestamp_ms" not in payload:
        payload["timestamp_ms"] = payload.pop("timestamp")
    payload["recv_ts_ms"] = recv_ts_ms

    model_cls = _REGISTRY[event_type]
    return model_cls.model_validate(payload)


__all__ = [
    "BookEvent",
    "LastTradePriceEvent",
    "MarketEvent",
    "PriceChangeEvent",
    "PriceLevel",
    "PriceLevelChange",
    "Side",
    "TickSizeChangeEvent",
    "UnknownEventTypeError",
    "parse_event",
]
