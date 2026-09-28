"""Test-suite-wide fixture: forces config.USE_STATEFLOW_FSM=False for
every test, regardless of config.py's actual checked-in default (True,
for real m5_server.py runs -- see that file's own comment). Keeps the
default `pytest tests/` run fast and MATLAB-free: test_planner.py builds
a fresh M5Engine 12 times, and each would otherwise try to start a MATLAB
engine (10-30s apiece) now that the project default is True.

tests/test_fsm_stateflow_parity.py is unaffected by this fixture on
purpose -- it exercises the real chart via fsm_stateflow.StateflowFSM
directly, never through M5Engine/this flag, gated behind its own
`matlab_parity` pytest marker instead.
"""
import pytest

import config as m5_config


@pytest.fixture(autouse=True)
def _force_python_fsm_by_default():
    original = m5_config.USE_STATEFLOW_FSM
    m5_config.USE_STATEFLOW_FSM = False
    yield
    m5_config.USE_STATEFLOW_FSM = original
