"""Behavior decision engine (spec section 3): CRUISE / FOLLOW / AVOID /
YIELD / EMERGENCY_BRAKE, driven by TTC and costmap/footprint risk flags.

Spec section 3 describes this as a MATLAB/Simulink Stateflow chart
(`behavior_fsm.slx`) reached via `matlabengine`. This machine checked: no
MATLAB/Simulink installation or license is available in this environment
(only CARLA + a Python toolchain), so this is a plain deterministic Python
state machine instead -- for exactly the reason M4's own README gives for
making the same call about its own math ("the whole pipeline is already
Python; a cross-language bridge would only add latency to the critical
path for no benefit"): a Stateflow chart's job here is a fixed decision
table over TTC/risk thresholds, which a Python function expresses exactly
as deterministically, with zero engine-startup latency and zero extra
runtime dependency. If a real Simulink chart is later required (e.g. for
certification/traceability reasons a Python function can't satisfy), this
module's `_raw_level` is the one place encoding the decision table that
chart would need to reproduce.

Escalating to a MORE cautious state is immediate (no hysteresis) --
delaying a safety escalation to avoid "flapping" is never the right
trade-off. De-escalating to a LESS cautious state requires the clearing
condition to hold with config.HYSTERESIS_MARGIN_S of margin (TTC-based
states) or a fully-clear risk reading (risk-based states) -- this is the
"missing transition arrows... lock the vehicle in safety states" behavior
spec section 3 describes.
"""
from __future__ import annotations

import config
from costmap import NONE, RED, YELLOW

STATE_NAMES = {0: "CRUISE", 1: "FOLLOW", 2: "AVOID", 3: "YIELD", 4: "EMERGENCY_BRAKE"}
LEVEL_BY_NAME = {v: k for k, v in STATE_NAMES.items()}

TARGET_SPEED_BY_STATE = {
    "CRUISE": config.CRUISE_SPEED_MPS,
    "FOLLOW": config.FOLLOW_SPEED_MPS,
    "AVOID": config.AVOID_SPEED_MPS,
    "YIELD": config.YIELD_SPEED_MPS,
    "EMERGENCY_BRAKE": config.EMERGENCY_SPEED_MPS,
}


def _raw_level(min_ttc: float | None, footprint_now_risk: str, footprint_future_risk: str) -> int:
    # 4: EMERGENCY_BRAKE -- imminent (TTC) or already-overlapping (RED right now) collision.
    if min_ttc is not None and min_ttc <= config.TTC_EMERGENCY_S:
        return 4
    if footprint_now_risk == RED:
        return 4
    # 2: AVOID -- a PRESENT, avoidable hazard (close now, or closing fast per
    # TTC) always gets the state that actually steers around it, checked
    # before YIELD's predicted-only branch below. Live 5-scenario validation
    # (dense_market) found the opposite ordering starved AVOID almost
    # entirely: for any obstacle that's roughly stationary relative to the
    # ego -- a parked bicycle, a roadside cone, anything M4 predicts holding
    # its current position -- footprint_future_risk tracks footprint_now_risk
    # almost exactly (nothing about the future is actually uncertain), so
    # the future check below fired first every time and locked in YIELD (a
    # straight-line path, planner.py's _plan_path) instead of AVOID (which
    # actually offsets laterally) -- the ego crept at full throttle straight
    # into the obstacle and got physically wedged, hundreds of
    # collision-sensor contacts across two live runs before this reorder.
    if min_ttc is not None and min_ttc <= config.TTC_AVOID_S:
        return 2
    if footprint_now_risk in (RED, YELLOW):
        return 2
    # 3: YIELD -- a PREDICTED-ONLY future conflict the raw TTC formula can
    # miss entirely for a crossing/offset path (spec open item E): nothing
    # avoidable right now (both AVOID checks above already passed), but the
    # ego's own straight-line projection meets the obstacle later because
    # THAT obstacle is still moving into the ego's path. Checked after AVOID
    # so a present hazard is never downgraded to a straight-line slowdown.
    if footprint_future_risk in (RED, YELLOW):
        return 3
    # 1: FOLLOW -- something ahead worth slowing for, no avoidance maneuver needed.
    if min_ttc is not None and min_ttc <= config.TTC_FOLLOW_S:
        return 1
    return 0


class BehaviorFSM:
    def __init__(self) -> None:
        self.level = 0

    @property
    def state(self) -> str:
        return STATE_NAMES[self.level]

    def step(self, min_ttc: float | None, footprint_now_risk: str, footprint_future_risk: str) -> str:
        raw = _raw_level(min_ttc, footprint_now_risk, footprint_future_risk)

        if raw >= self.level:
            self.level = raw  # escalate immediately, no hysteresis
            return self.state

        # De-escalating: require the condition that put us in the CURRENT
        # (more cautious) level to have cleared with margin, not just the
        # raw level to have dropped by one.
        cleared = self._current_condition_cleared(min_ttc, footprint_now_risk, footprint_future_risk)
        if cleared:
            self.level = raw
        return self.state

    def _current_condition_cleared(self, min_ttc: float | None, footprint_now_risk: str, footprint_future_risk: str) -> bool:
        margin = config.HYSTERESIS_MARGIN_S
        if self.level == 4:
            ttc_clear = min_ttc is None or min_ttc > config.TTC_EMERGENCY_S + margin
            return ttc_clear and footprint_now_risk != RED
        if self.level == 3:
            return footprint_future_risk == NONE
        if self.level == 2:
            ttc_clear = min_ttc is None or min_ttc > config.TTC_AVOID_S + margin
            return ttc_clear and footprint_now_risk in (NONE,)
        if self.level == 1:
            return min_ttc is None or min_ttc > config.TTC_FOLLOW_S + margin
        return True
