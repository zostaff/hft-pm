"""Unit tests for hft_pm.risk.limits."""

from __future__ import annotations

import pytest

from hft_pm.risk.limits import HaltReason, KillSwitch, RiskLimits

# ----------------------------------------------------------------------
# RiskLimits validation
# ----------------------------------------------------------------------


def test_limits_validates_inputs() -> None:
    with pytest.raises(ValueError):
        RiskLimits(max_drawdown_pct=0)
    with pytest.raises(ValueError):
        RiskLimits(max_drawdown_pct=1.5)
    with pytest.raises(ValueError):
        RiskLimits(max_inventory=0)
    with pytest.raises(ValueError):
        RiskLimits(heartbeat_timeout_s=0)
    with pytest.raises(ValueError):
        RiskLimits(daily_loss_limit=-10)
    with pytest.raises(ValueError):
        RiskLimits(baseline_capital=0)


# ----------------------------------------------------------------------
# Max drawdown
# ----------------------------------------------------------------------


def test_drawdown_uses_baseline_when_peak_below_baseline() -> None:
    """At session start with no PnL, drawdown is measured against baseline_capital."""
    sw = KillSwitch(RiskLimits(max_drawdown_pct=0.20, baseline_capital=100.0))
    sw.tick(now_s=0, current_pnl=0.0, inventory=0.0)
    sw.tick(now_s=1, current_pnl=-15.0, inventory=0.0)
    # -15 / 100 = 15 % DD < 20 % → still healthy
    assert sw.halted is False
    sw.tick(now_s=2, current_pnl=-25.0, inventory=0.0)
    # 25 % DD > 20 % → tripped
    assert sw.halted is True
    assert sw.halt_reason == HaltReason.MAX_DRAWDOWN


def test_drawdown_uses_peak_once_above_baseline() -> None:
    sw = KillSwitch(RiskLimits(max_drawdown_pct=0.20, baseline_capital=100.0))
    sw.tick(now_s=0, current_pnl=200.0, inventory=0.0)  # peak = 200
    sw.tick(now_s=1, current_pnl=170.0, inventory=0.0)  # 15 % DD
    assert sw.halted is False
    sw.tick(now_s=2, current_pnl=150.0, inventory=0.0)  # 25 % DD
    assert sw.halted is True


# ----------------------------------------------------------------------
# Heartbeat
# ----------------------------------------------------------------------


def test_heartbeat_trips_after_timeout() -> None:
    sw = KillSwitch(RiskLimits(heartbeat_timeout_s=5.0))
    sw.tick(now_s=1000, current_pnl=0, inventory=0)
    sw.heartbeat_check(now_s=1004)
    assert sw.halted is False
    sw.heartbeat_check(now_s=1006)
    assert sw.halted is True
    assert sw.halt_reason == HaltReason.HEARTBEAT_TIMEOUT


def test_heartbeat_noop_before_first_tick() -> None:
    sw = KillSwitch(RiskLimits())
    sw.heartbeat_check(now_s=10_000)
    assert sw.halted is False


def test_note_ws_message_keeps_heartbeat_fresh_without_pnl_tick() -> None:
    """`note_ws_message` is the WS-liveness ping that runs on every raw
    message including foreign-asset ones. It must defer the heartbeat
    halt without touching PnL/inventory state, so a low-volume token
    paired with a busy multi-token WS doesn't trip the kill switch."""
    sw = KillSwitch(RiskLimits(heartbeat_timeout_s=5.0))
    # Bootstrap _last_event_s with a tick (an own-asset event).
    sw.tick(now_s=1000, current_pnl=0, inventory=0)
    # Lots of foreign-asset messages over the next 10 s — far past the
    # 5 s heartbeat timeout — but `note_ws_message` keeps refreshing.
    for t in range(1001, 1011):
        sw.note_ws_message(now_s=float(t))
        sw.heartbeat_check(now_s=float(t))
        assert sw.halted is False, f"tripped at t={t}"
    # The moment WS goes silent, heartbeat starts counting again.
    sw.heartbeat_check(now_s=1016.0)  # 6 s since last note
    assert sw.halted is True
    assert sw.halt_reason == HaltReason.HEARTBEAT_TIMEOUT


# ----------------------------------------------------------------------
# Inventory cap
# ----------------------------------------------------------------------


def test_can_open_bid_blocks_at_positive_cap() -> None:
    sw = KillSwitch(RiskLimits(max_inventory=10.0))
    sw.tick(now_s=0, current_pnl=0, inventory=10.0)
    assert sw.can_open("bid") is False  # would push above +10
    assert sw.can_open("ask") is True  # selling reduces inventory


def test_can_open_ask_blocks_at_negative_cap() -> None:
    sw = KillSwitch(RiskLimits(max_inventory=10.0))
    sw.tick(now_s=0, current_pnl=0, inventory=-10.0)
    assert sw.can_open("ask") is False  # would push below -10
    assert sw.can_open("bid") is True


def test_can_open_rejects_invalid_side() -> None:
    sw = KillSwitch(RiskLimits())
    with pytest.raises(ValueError):
        sw.can_open("foo")


def test_can_open_returns_false_when_halted() -> None:
    sw = KillSwitch(RiskLimits(max_drawdown_pct=0.20))
    sw.tick(now_s=0, current_pnl=-100, inventory=0)  # huge DD vs baseline 100
    assert sw.halted is True
    assert sw.can_open("bid") is False
    assert sw.can_open("ask") is False


# ----------------------------------------------------------------------
# Daily loss limit
# ----------------------------------------------------------------------


def test_daily_loss_trips_within_one_day() -> None:
    sw = KillSwitch(RiskLimits(daily_loss_limit=50.0, max_drawdown_pct=0.99))
    sw.tick(now_s=0, current_pnl=100, inventory=0)  # anchor 100
    sw.tick(now_s=100, current_pnl=80, inventory=0)  # loss 20
    assert sw.halted is False
    sw.tick(now_s=200, current_pnl=45, inventory=0)  # loss 55 → trip
    assert sw.halted is True
    assert sw.halt_reason == HaltReason.DAILY_LOSS


def test_daily_loss_resets_at_utc_midnight() -> None:
    sw = KillSwitch(RiskLimits(daily_loss_limit=50.0, max_drawdown_pct=0.99))
    sw.tick(now_s=86_000, current_pnl=100, inventory=0)  # day 0, anchor 100
    sw.tick(now_s=86_100, current_pnl=60, inventory=0)  # loss 40, still under
    sw.tick(now_s=86_500, current_pnl=60, inventory=0)  # day 1: reset anchor to 60
    sw.tick(now_s=86_700, current_pnl=20, inventory=0)  # loss 40 within day 1
    assert sw.halted is False
    sw.tick(now_s=86_800, current_pnl=5, inventory=0)  # loss 55 within day 1 → trip
    assert sw.halted is True
    assert sw.halt_reason == HaltReason.DAILY_LOSS


# ----------------------------------------------------------------------
# Reset
# ----------------------------------------------------------------------


def test_reset_clears_halt_and_history() -> None:
    sw = KillSwitch(RiskLimits(max_drawdown_pct=0.10, baseline_capital=100.0))
    sw.tick(now_s=0, current_pnl=-50, inventory=0)
    assert sw.halted is True
    sw.reset()
    assert sw.halted is False
    assert sw.halt_reason == HaltReason.NONE
    # After reset, a small drawdown does NOT immediately retrip.
    sw.tick(now_s=10, current_pnl=-5, inventory=0)
    assert sw.halted is False
