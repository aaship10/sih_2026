"""Python adapter for behavior_fsm.slx -- the real MATLAB/Simulink
Stateflow chart the problem statement requires (SPEC.md section 3), wired
in behind the exact same interface fsm.py's BehaviorFSM exposes
(`.step(min_ttc, now_risk, future_risk) -> str`) so planner.py's call site
(`M5Engine.step()`) doesn't need to know which one it's talking to.

Why a persistent, kept-loaded simulation (SimulationCommand 'start'/
'pause'/'step') instead of a fresh sim() per tick: benchmarked directly in
this environment -- a fresh-sim()-per-call prototype (even under Fast
Restart, even after removing per-call string-parsing and object-
reconstruction overhead) cost ~72-75ms median / ~106ms p95, already
exceeding the 100ms/10Hz tick budget on its own before the rest of
M5Engine.step()'s TTC/costmap/quintic work shares that same window.
Persistent step-mode (matlab/benchmark_stepmode.m) measured ~9ms median /
~16ms p95 -- comfortably inside budget with real headroom left over.
End-to-end M5Engine.step() latency with this adapter wired in
(matlab/benchmark_e2e.py): ~31ms median / ~58ms p95.

The chart itself (behavior_fsm.slx, built by matlab/build_behavior_fsm.m)
carries the decision-table state (`level`) as its OWN internal Stateflow
persistent local data, not an explicit input -- correct because a
continuously running/paused simulation naturally carries that forward
between manual 'step' calls with no extra plumbing (unlike a fresh sim()
per call, which is why an earlier abandoned prototype needed an explicit
level_in workaround instead). compute_raw_level and each state's
cleared() condition are byte-for-byte the same decision table as fsm.py's
_raw_level/_current_condition_cleared -- verified via matlab/test_stepmode.m
against the exact same cases (including the AVOID-before-YIELD
live-incident fix).
"""
from __future__ import annotations

import math
from pathlib import Path

from costmap import RISK_SEVERITY
from fsm import STATE_NAMES

MATLAB_DIR = Path(__file__).parent / "matlab"
MODEL_NAME = "behavior_fsm"


class StateflowFSM:
    """Drop-in replacement for fsm.py's BehaviorFSM, backed by the real
    Simulink Stateflow chart. Raises on construction if the MATLAB Engine
    or the model can't be started -- callers (M5Engine.__init__) are
    expected to catch this and fall back to BehaviorFSM with a loud
    logged warning, never silently."""

    def __init__(self) -> None:
        import matlab.engine  # local import: only needed on this path, keeps the Python-only fallback importable without matlabengine installed

        self._eng = matlab.engine.start_matlab()
        self._eng.addpath(str(MATLAB_DIR), nargout=0)
        model_path = str(MATLAB_DIR / f"{MODEL_NAME}.slx")
        if not (MATLAB_DIR / f"{MODEL_NAME}.slx").exists():
            self._eng.eval(f"build_behavior_fsm();", nargout=0)
        self._eng.load_system(model_path, nargout=0)
        self._eng.set_param(MODEL_NAME, "StopTime", "inf", nargout=0)
        self._eng.set_param(MODEL_NAME, "SimulationCommand", "start", nargout=0)
        self._eng.set_param(MODEL_NAME, "SimulationCommand", "pause", nargout=0)
        self._chart_path = f"{MODEL_NAME}/behavior_fsm_chart"
        self._closed = False

    def step(self, min_ttc: float | None, footprint_now_risk: str, footprint_future_risk: str) -> str:
        ttc_value = math.inf if min_ttc is None else min_ttc
        self._eng.set_param(f"{MODEL_NAME}/min_ttc", "Value", str(ttc_value), nargout=0)
        self._eng.set_param(f"{MODEL_NAME}/now_risk", "Value", str(RISK_SEVERITY[footprint_now_risk]), nargout=0)
        self._eng.set_param(f"{MODEL_NAME}/future_risk", "Value", str(RISK_SEVERITY[footprint_future_risk]), nargout=0)
        self._eng.set_param(MODEL_NAME, "SimulationCommand", "step", nargout=0)
        # The whole property-chain read happens MATLAB-side (a Python-side
        # handle to a Simulink.MSObject can't be indexed/dotted the way
        # MATLAB's own OutputPort(1).Data syntax needs) -- only a plain
        # double crosses the engine boundary.
        level = self._eng.eval(
            f"double(get_param('{self._chart_path}','RuntimeObject').OutputPort(1).Data)",
            nargout=1,
        )
        return STATE_NAMES[int(round(level))]

    def reset(self) -> None:
        """Restarts the paused simulation so the chart's persistent
        `level` goes back through its default transition to CRUISE (0) --
        lets test suites get a logically fresh instance per test case
        without paying MATLAB engine startup cost (10-30s) each time."""
        self._eng.set_param(MODEL_NAME, "SimulationCommand", "stop", nargout=0)
        self._eng.set_param(MODEL_NAME, "SimulationCommand", "start", nargout=0)
        self._eng.set_param(MODEL_NAME, "SimulationCommand", "pause", nargout=0)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._eng.set_param(MODEL_NAME, "SimulationCommand", "stop", nargout=0)
            self._eng.close_system(MODEL_NAME, 0, nargout=0)
        finally:
            self._eng.quit()

    def __del__(self) -> None:
        self.close()
