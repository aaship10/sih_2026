function test_stepmode()
addpath(fileparts(mfilename('fullpath')));
mdl = 'behavior_fsm';
load_system(fullfile(fileparts(mfilename('fullpath')), 'behavior_fsm.slx'));
set_param(mdl, 'StopTime', 'inf');

set_param(mdl, 'SimulationCommand', 'start');
set_param(mdl, 'SimulationCommand', 'pause');
chartPath = [mdl '/behavior_fsm_chart'];

function v = readLevel()
    rto = get_param(chartPath, 'RuntimeObject');
    v = rto.OutputPort(1).Data;
end

% Step 1: min_ttc=2.75 -> expect raw=2, fresh CRUISE -> should escalate to AVOID.
set_param([mdl '/min_ttc'], 'Value', '2.75');
set_param([mdl '/now_risk'], 'Value', '0');
set_param([mdl '/future_risk'], 'Value', '0');
set_param(mdl, 'SimulationCommand', 'step');
fprintf('step 1 (min_ttc=2.75, fresh): level=%g (expect 2)\n', readLevel());

% Step 2: same inputs -- expect STAYS at 2.
set_param(mdl, 'SimulationCommand', 'step');
fprintf('step 2 (same inputs, confirm STAYS at AVOID): level=%g (expect 2)\n', readLevel());

% Step 3: force EMERGENCY_BRAKE.
set_param([mdl '/min_ttc'], 'Value', '1.0');
set_param(mdl, 'SimulationCommand', 'step');
fprintf('step 3 (min_ttc=1.0): level=%g (expect 4)\n', readLevel());

% Step 4: raw=2 but EMERGENCY_BRAKE not cleared (min_ttc=1.8 < 2.0) -> stays 4.
set_param([mdl '/min_ttc'], 'Value', '1.8');
set_param(mdl, 'SimulationCommand', 'step');
fprintf('step 4 (min_ttc=1.8, not cleared): level=%g (expect 4)\n', readLevel());

% Step 5: cleared -> de-escalate to AVOID (raw=2).
set_param([mdl '/min_ttc'], 'Value', '2.75');
set_param(mdl, 'SimulationCommand', 'step');
fprintf('step 5 (min_ttc=2.75, cleared): level=%g (expect 2)\n', readLevel());

set_param(mdl, 'SimulationCommand', 'stop');
close_system(mdl, 0);
end
