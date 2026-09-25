"""Hand-computed cases for fsm.py's decision table + hysteresis."""
import config
from costmap import NONE, RED, YELLOW
from fsm import BehaviorFSM


def test_starts_at_cruise_with_no_risk():
    fsm = BehaviorFSM()
    assert fsm.step(min_ttc=None, footprint_now_risk=NONE, footprint_future_risk=NONE) == "CRUISE"


def test_low_ttc_forces_emergency_brake_unconditionally():
    fsm = BehaviorFSM()
    state = fsm.step(min_ttc=config.TTC_EMERGENCY_S, footprint_now_risk=NONE, footprint_future_risk=NONE)
    assert state == "EMERGENCY_BRAKE"


def test_emergency_brake_escalation_is_immediate_even_from_cruise():
    fsm = BehaviorFSM()
    fsm.step(min_ttc=None, footprint_now_risk=NONE, footprint_future_risk=NONE)
    assert fsm.state == "CRUISE"
    state = fsm.step(min_ttc=0.5, footprint_now_risk=NONE, footprint_future_risk=NONE)
    assert state == "EMERGENCY_BRAKE"


def test_red_footprint_now_forces_emergency_even_without_ttc():
    fsm = BehaviorFSM()
    state = fsm.step(min_ttc=None, footprint_now_risk=RED, footprint_future_risk=NONE)
    assert state == "EMERGENCY_BRAKE"


def test_future_yellow_triggers_yield_not_avoid():
    # A predicted (not current) conflict, per spec item E's mitigation.
    fsm = BehaviorFSM()
    state = fsm.step(min_ttc=None, footprint_now_risk=NONE, footprint_future_risk=YELLOW)
    assert state == "YIELD"


def test_moderate_ttc_triggers_avoid():
    fsm = BehaviorFSM()
    mid_ttc = (config.TTC_EMERGENCY_S + config.TTC_AVOID_S) / 2.0
    state = fsm.step(min_ttc=mid_ttc, footprint_now_risk=NONE, footprint_future_risk=NONE)
    assert state == "AVOID"


def test_long_ttc_triggers_follow():
    fsm = BehaviorFSM()
    mid_ttc = (config.TTC_AVOID_S + config.TTC_FOLLOW_S) / 2.0
    state = fsm.step(min_ttc=mid_ttc, footprint_now_risk=NONE, footprint_future_risk=NONE)
    assert state == "FOLLOW"


def test_hysteresis_holds_emergency_brake_just_past_the_raw_threshold():
    fsm = BehaviorFSM()
    fsm.step(min_ttc=1.0, footprint_now_risk=NONE, footprint_future_risk=NONE)
    assert fsm.state == "EMERGENCY_BRAKE"
    # TTC recovers to just barely above the raw threshold -- NOT enough
    # margin to leave EMERGENCY_BRAKE yet (this is the whole point of
    # hysteresis: prevent flapping right at the boundary).
    just_over = config.TTC_EMERGENCY_S + 0.01
    state = fsm.step(min_ttc=just_over, footprint_now_risk=NONE, footprint_future_risk=NONE)
    assert state == "EMERGENCY_BRAKE"


def test_hysteresis_releases_once_clearly_safe():
    fsm = BehaviorFSM()
    fsm.step(min_ttc=1.0, footprint_now_risk=NONE, footprint_future_risk=NONE)
    assert fsm.state == "EMERGENCY_BRAKE"
    clearly_safe_ttc = config.TTC_EMERGENCY_S + config.HYSTERESIS_MARGIN_S + 0.5
    state = fsm.step(min_ttc=clearly_safe_ttc, footprint_now_risk=NONE, footprint_future_risk=NONE)
    assert state != "EMERGENCY_BRAKE"


def test_sustained_safe_input_eventually_returns_to_cruise():
    fsm = BehaviorFSM()
    fsm.step(min_ttc=1.0, footprint_now_risk=NONE, footprint_future_risk=NONE)
    assert fsm.state == "EMERGENCY_BRAKE"
    for _ in range(10):
        state = fsm.step(min_ttc=None, footprint_now_risk=NONE, footprint_future_risk=NONE)
    assert state == "CRUISE"


def test_no_oscillation_right_at_the_avoid_boundary():
    # Stepping the same borderline TTC repeatedly must not flip-flop.
    fsm = BehaviorFSM()
    seen = set()
    borderline = config.TTC_AVOID_S
    for _ in range(5):
        seen.add(fsm.step(min_ttc=borderline, footprint_now_risk=NONE, footprint_future_risk=NONE))
    assert len(seen) == 1
