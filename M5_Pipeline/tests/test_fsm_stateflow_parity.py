"""Golden-vector parity check: the real MATLAB/Simulink Stateflow chart
(behavior_fsm.slx, driven via fsm_stateflow.StateflowFSM) against fsm.py's
BehaviorFSM reference -- every case in test_fsm.py, run through both, plus
the two live-incident regression scenarios' exact FSM-input triples
(extracted from M5Engine.last_diagnostics against the real
test_planner.py regression packets, not guessed) since those pin down the
AVOID-before-YIELD ordering fix specifically.

Requires a working MATLAB Engine + Simulink/Stateflow install -- if it's
not even installed, `pytest.importorskip` skips this module entirely, but
that's NOT enough on its own to keep day-to-day runs fast: matlabengine
genuinely IS installed in this venv, so importorskip alone would let this
whole (slow) module collect and run as part of a plain `pytest` invocation
(confirmed: this bumped the full-suite runtime from <1s to ~36s the first
time). Marked `matlab_parity` (see pytest.ini's `addopts = -m "not
matlab_parity"`) so it's excluded from the default run regardless of
whether the engine is installed -- invoke explicitly with
`pytest tests/test_fsm_stateflow_parity.py -m matlab_parity`.
matlab.engine.start_matlab() alone costs 10-30s, paid ONCE per test session
via the module-scoped fixture below, not per test case.
"""
import config
import pytest
from costmap import NONE, RED, YELLOW
from fsm import BehaviorFSM

pytestmark = pytest.mark.matlab_parity

matlab_engine = pytest.importorskip("matlab.engine", reason="MATLAB Engine for Python not installed")

from fsm_stateflow import StateflowFSM  # noqa: E402  (must follow importorskip)


@pytest.fixture(scope="module")
def _stateflow_session():
    try:
        fsm = StateflowFSM()
    except Exception as exc:  # pragma: no cover -- environment-dependent (no MATLAB/Simulink license, etc.)
        pytest.skip(f"MATLAB engine/model unavailable: {exc}")
    yield fsm
    fsm.close()


@pytest.fixture()
def stateflow_fsm(_stateflow_session):
    _stateflow_session.reset()
    return _stateflow_session


# ---- Every case in test_fsm.py, mirrored against the real chart ----

def test_starts_at_cruise_with_no_risk(stateflow_fsm):
    assert stateflow_fsm.step(min_ttc=None, footprint_now_risk=NONE, footprint_future_risk=NONE) == "CRUISE"


def test_low_ttc_forces_emergency_brake_unconditionally(stateflow_fsm):
    state = stateflow_fsm.step(min_ttc=config.TTC_EMERGENCY_S, footprint_now_risk=NONE, footprint_future_risk=NONE)
    assert state == "EMERGENCY_BRAKE"


def test_emergency_brake_escalation_is_immediate_even_from_cruise(stateflow_fsm):
    state = stateflow_fsm.step(min_ttc=None, footprint_now_risk=NONE, footprint_future_risk=NONE)
    assert state == "CRUISE"
    state = stateflow_fsm.step(min_ttc=0.5, footprint_now_risk=NONE, footprint_future_risk=NONE)
    assert state == "EMERGENCY_BRAKE"


def test_red_footprint_now_forces_emergency_even_without_ttc(stateflow_fsm):
    state = stateflow_fsm.step(min_ttc=None, footprint_now_risk=RED, footprint_future_risk=NONE)
    assert state == "EMERGENCY_BRAKE"


def test_future_yellow_triggers_yield_not_avoid(stateflow_fsm):
    # A predicted (not current) conflict, per spec item E's mitigation.
    state = stateflow_fsm.step(min_ttc=None, footprint_now_risk=NONE, footprint_future_risk=YELLOW)
    assert state == "YIELD"


def test_moderate_ttc_triggers_avoid(stateflow_fsm):
    mid_ttc = (config.TTC_EMERGENCY_S + config.TTC_AVOID_S) / 2.0
    state = stateflow_fsm.step(min_ttc=mid_ttc, footprint_now_risk=NONE, footprint_future_risk=NONE)
    assert state == "AVOID"


def test_long_ttc_triggers_follow(stateflow_fsm):
    mid_ttc = (config.TTC_AVOID_S + config.TTC_FOLLOW_S) / 2.0
    state = stateflow_fsm.step(min_ttc=mid_ttc, footprint_now_risk=NONE, footprint_future_risk=NONE)
    assert state == "FOLLOW"


def test_hysteresis_holds_emergency_brake_just_past_the_raw_threshold(stateflow_fsm):
    stateflow_fsm.step(min_ttc=1.0, footprint_now_risk=NONE, footprint_future_risk=NONE)
    just_over = config.TTC_EMERGENCY_S + 0.01
    state = stateflow_fsm.step(min_ttc=just_over, footprint_now_risk=NONE, footprint_future_risk=NONE)
    assert state == "EMERGENCY_BRAKE"


def test_hysteresis_releases_once_clearly_safe(stateflow_fsm):
    stateflow_fsm.step(min_ttc=1.0, footprint_now_risk=NONE, footprint_future_risk=NONE)
    clearly_safe_ttc = config.TTC_EMERGENCY_S + config.HYSTERESIS_MARGIN_S + 0.5
    state = stateflow_fsm.step(min_ttc=clearly_safe_ttc, footprint_now_risk=NONE, footprint_future_risk=NONE)
    assert state != "EMERGENCY_BRAKE"


def test_sustained_safe_input_eventually_returns_to_cruise(stateflow_fsm):
    stateflow_fsm.step(min_ttc=1.0, footprint_now_risk=NONE, footprint_future_risk=NONE)
    state = None
    for _ in range(10):
        state = stateflow_fsm.step(min_ttc=None, footprint_now_risk=NONE, footprint_future_risk=NONE)
    assert state == "CRUISE"


def test_no_oscillation_right_at_the_avoid_boundary(stateflow_fsm):
    seen = set()
    borderline = config.TTC_AVOID_S
    for _ in range(5):
        seen.add(stateflow_fsm.step(min_ttc=borderline, footprint_now_risk=NONE, footprint_future_risk=NONE))
    assert len(seen) == 1


# ---- The two live-incident regression triples (the AVOID-before-YIELD fix) ----
# Values extracted from M5Engine.last_diagnostics by actually running
# test_planner.py's real regression packets through the full engine, not
# guessed -- these are exactly what starved AVOID before the ordering fix.

def test_regression_static_ahead_yellow_risk_triple(stateflow_fsm):
    # test_regression_static_ahead_at_yellow_risk_triggers_avoid_not_yield's
    # underlying FSM inputs: a static object, so TTC is never computed
    # (compute_min_ttc only sees packet.obstacles, never packet.statics).
    state = stateflow_fsm.step(min_ttc=None, footprint_now_risk=YELLOW, footprint_future_risk=NONE)
    assert state == "AVOID"


def test_regression_near_stationary_dynamic_triple(stateflow_fsm):
    # test_regression_near_stationary_dynamic_obstacle_ahead_triggers_avoid_not_yield's
    # underlying FSM inputs: now_risk AND future_risk both YELLOW (a
    # near-zero-velocity obstacle's CV-predicted future tracks its
    # current position) -- this is exactly the case the old
    # future-before-AVOID ordering got wrong.
    state = stateflow_fsm.step(min_ttc=9.482758620689657, footprint_now_risk=YELLOW, footprint_future_risk=YELLOW)
    assert state == "AVOID"


# ---- Cross-check against the Python reference directly (belt and braces:
# confirms BOTH implementations agree, not just that each independently
# matches the expected label) ----

@pytest.mark.parametrize("min_ttc,now_risk,future_risk", [
    (None, NONE, NONE),
    (config.TTC_EMERGENCY_S, NONE, NONE),
    (None, RED, NONE),
    (None, NONE, YELLOW),
    ((config.TTC_EMERGENCY_S + config.TTC_AVOID_S) / 2.0, NONE, NONE),
    ((config.TTC_AVOID_S + config.TTC_FOLLOW_S) / 2.0, NONE, NONE),
    (None, YELLOW, NONE),
    (9.482758620689657, YELLOW, YELLOW),
])
def test_matches_python_reference_single_step(stateflow_fsm, min_ttc, now_risk, future_risk):
    python_state = BehaviorFSM().step(min_ttc, now_risk, future_risk)
    stateflow_state = stateflow_fsm.step(min_ttc, now_risk, future_risk)
    assert stateflow_state == python_state
