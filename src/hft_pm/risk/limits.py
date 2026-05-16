"""Risk limits and kill switches (docs §12 "Kill switches non-negotiable").

A :class:`KillSwitch` is a stateful guard that the live runner consults
on every event tick. The four hard rules from CLAUDE.md Critical Rule
#9 are encoded here:

* Max drawdown from peak — halts ALL trading when breached
* Heartbeat timeout — halts when no event observed in ``heartbeat_timeout_s``
* Inventory cap exceeded — halts NEW orders on the violating side
* Daily realised loss — halts ALL trading until the next UTC day

When ``halted`` is True the runner must stop submitting orders; existing
resting orders should be cancelled by the runner (the switch reports
the state, it does not reach into the broker itself).

All thresholds are interpreted as **absolute dollars** for cash/PnL
fields and **absolute contracts** for inventory, matching the units
the engine reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class HaltReason(StrEnum):
    """Why the kill switch tripped. ``NONE`` = healthy."""

    NONE = "none"
    MAX_DRAWDOWN = "max_drawdown"
    HEARTBEAT_TIMEOUT = "heartbeat_timeout"
    DAILY_LOSS = "daily_loss"


@dataclass(frozen=True)
class RiskLimits:
    """Static configuration for the kill switch.

    Parameters
    ----------
    max_drawdown_pct:
        Trip if (peak_pnl - current_pnl) / max(peak_pnl, baseline_capital)
        exceeds this fraction. CLAUDE.md mandates 0.20 (20 %).
    max_inventory:
        Per-side inventory cap (absolute contracts). When breached on a
        side, :meth:`KillSwitch.can_open` returns False for that side
        while still allowing inventory-reducing orders.
    heartbeat_timeout_s:
        Halt if no event observed in this many seconds. The runner is
        responsible for calling :meth:`KillSwitch.tick` with the
        current wall-clock so the switch can detect a frozen feed.
        CLAUDE.md mandates 30 s for the WS heartbeat.
    daily_loss_limit:
        Trip if cumulative realised loss within a single UTC day
        exceeds this dollar value (positive number). ``None`` disables.
    baseline_capital:
        Reference capital used as the denominator for drawdown when
        peak_pnl is still below this value. Prevents 100 %-of-zero
        drawdowns at the start of a session.
    """

    max_drawdown_pct: float = 0.20
    max_inventory: float = 100.0
    heartbeat_timeout_s: float = 30.0
    daily_loss_limit: float | None = None
    baseline_capital: float = 100.0

    def __post_init__(self) -> None:
        if not 0 < self.max_drawdown_pct < 1:
            raise ValueError("max_drawdown_pct must be in (0, 1)")
        if self.max_inventory <= 0:
            raise ValueError("max_inventory must be positive")
        if self.heartbeat_timeout_s <= 0:
            raise ValueError("heartbeat_timeout_s must be positive")
        if self.daily_loss_limit is not None and self.daily_loss_limit <= 0:
            raise ValueError("daily_loss_limit must be positive when set")
        if self.baseline_capital <= 0:
            raise ValueError("baseline_capital must be positive")


class KillSwitch:
    """Stateful kill switch consulted by the live / paper runner.

    Usage::

        switch = KillSwitch(RiskLimits(max_drawdown_pct=0.20))
        switch.tick(now_s=time.time(), current_pnl=pnl, inventory=inv)
        if switch.halted:
            stop_trading(reason=switch.halt_reason)
        elif not switch.can_open("bid"):
            ...
    """

    def __init__(self, limits: RiskLimits) -> None:
        self.limits = limits
        self._peak_pnl: float = 0.0
        self._inventory: float = 0.0
        self._last_event_s: float | None = None
        self._daily_loss: float = 0.0
        self._daily_anchor_pnl: float = 0.0
        self._current_utc_day: int | None = None
        self.halted: bool = False
        self.halt_reason: HaltReason = HaltReason.NONE

    def tick(
        self,
        *,
        now_s: float,
        current_pnl: float,
        inventory: float,
    ) -> None:
        """Feed the latest state. Sets ``halted`` if any rule trips.

        ``current_pnl`` is the mark-to-mid PnL since session start
        (cash + inventory · mid). ``inventory`` is signed contracts.
        """
        self._inventory = inventory
        self._last_event_s = now_s
        self._peak_pnl = max(self._peak_pnl, current_pnl)

        utc_day = int(now_s // 86_400)
        if self._current_utc_day is None or utc_day != self._current_utc_day:
            self._current_utc_day = utc_day
            self._daily_anchor_pnl = current_pnl
            self._daily_loss = 0.0
        else:
            self._daily_loss = max(0.0, self._daily_anchor_pnl - current_pnl)

        denom = max(self._peak_pnl, self.limits.baseline_capital)
        drawdown_pct = (self._peak_pnl - current_pnl) / denom
        if drawdown_pct > self.limits.max_drawdown_pct:
            self._trip(HaltReason.MAX_DRAWDOWN)
            return
        if (
            self.limits.daily_loss_limit is not None
            and self._daily_loss > self.limits.daily_loss_limit
        ):
            self._trip(HaltReason.DAILY_LOSS)
            return

    def heartbeat_check(self, *, now_s: float) -> None:
        """Call from a watchdog timer to detect a frozen feed."""
        if self._last_event_s is None:
            return
        if now_s - self._last_event_s > self.limits.heartbeat_timeout_s:
            self._trip(HaltReason.HEARTBEAT_TIMEOUT)

    def can_open(self, side: str) -> bool:
        """Return False if a new order on ``side`` would breach the inventory cap.

        Inventory-reducing orders are always allowed (you must be able
        to flatten even at the cap).
        """
        if self.halted:
            return False
        cap = self.limits.max_inventory
        if side == "bid":
            return self._inventory < cap  # buying must not push above +cap
        if side == "ask":
            return self._inventory > -cap  # selling must not push below -cap
        raise ValueError(f"invalid side: {side!r}")

    def reset(self) -> None:
        """Clear the halt and zero stateful tracking — for tests / manual recovery."""
        self._peak_pnl = 0.0
        self._inventory = 0.0
        self._last_event_s = None
        self._daily_loss = 0.0
        self._daily_anchor_pnl = 0.0
        self._current_utc_day = None
        self.halted = False
        self.halt_reason = HaltReason.NONE

    def _trip(self, reason: HaltReason) -> None:
        self.halted = True
        self.halt_reason = reason


__all__ = ["HaltReason", "KillSwitch", "RiskLimits"]
