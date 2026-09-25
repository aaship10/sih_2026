function build_behavior_fsm()
% Builds behavior_fsm.slx: the MATLAB/Simulink Stateflow chart SPEC.md
% section 3 requires as M5's behavior decision engine, ported line-for-line
% from M5_Pipeline/fsm.py's _raw_level (compute_raw_level below) and
% BehaviorFSM._current_condition_cleared (each state's cleared() guard) --
% see fsm.py's own docstring, which names _raw_level as exactly what a
% Stateflow chart would need to reproduce.
%
% Invocation model: designed for a PERSISTENT, kept-loaded simulation
% (SimulationCommand 'start'/'pause'/'step', driven from Python by
% fsm_stateflow.py's StateflowFSM adapter), not a fresh sim() per tick.
% This was a real, measured decision, not a style choice: a fresh-sim()
% prototype (even under Simulink's Fast Restart, even after removing
% per-call string-parsing/object-reconstruction overhead) cost ~72-75ms
% median / ~106ms p95 per FSM call alone -- already over the 100ms/10Hz
% tick budget before the rest of M5Engine.step()'s TTC/costmap/quintic
% work shares that same window. Persistent step-mode measures ~9ms
% median / ~16ms p95 (matlab/benchmark_stepmode.m) -- comfortably inside
% budget with headroom for the rest of the tick.
%
% Two consequences of the persistent-step-mode invocation model:
% - Inputs are Constant blocks (not root In1 ports) whose Value gets
%   poked via set_param between steps -- root-level ExternalInput
%   strings/SimulationInput objects are for sim()-style runs, not a
%   continuously running/paused model.
% - Output is read live via the chart block's own RuntimeObject.
%   OutputPort(1).Data while paused between steps (a "To Workspace"
%   block was tried first -- it only flushes its buffer when the
%   simulation fully STOPS, not on pause/step, so it can't read
%   intermediate values; a plain Terminator sinks the output instead).
% - `level` is the chart's OWN internal persistent Local data (not an
%   explicit input) -- a continuously-running/paused simulation
%   naturally carries Stateflow local data forward between manual
%   'step' calls with no extra plumbing needed.

modelName = 'behavior_fsm';
if bdIsLoaded(modelName)
    close_system(modelName, 0);
end
new_system(modelName);
open_system(modelName);
set_param(modelName, 'Solver', 'FixedStepDiscrete', 'FixedStep', '0.1');

% ---- Inputs: Constant blocks, value poked via set_param between steps ----
% Default Values matter even though the caller overwrites them every real
% step() call: StateflowFSM.__init__'s own 'start' then 'pause' sequence
% (needed to get the chart into a pauseable, steppable state at all)
% evaluates the chart ONCE at t=0 using whatever these defaults are,
% BEFORE the caller ever gets a chance to set_param a real reading in --
% confirmed live by reading RuntimeObject.OutputPort(1).Data immediately
% after __init__, with no step() call at all. min_ttc defaulting to '0'
% (0 <= TTC_EMERGENCY_S) made that spurious first evaluation escalate
% straight to EMERGENCY_BRAKE(4) on every single engine construction,
% before any real sensor data existed -- and because de-escalation out of
% EMERGENCY_BRAKE requires min_ttc_c > 1.5+0.5 strictly, a plausible
% first-real-tick TTC anywhere at or below 2.0s (not otherwise unusual)
% left it wrongly stuck there instead of reflecting the real, lower-risk
% state. A large default (999, meaning "no obstacle yet") makes this
% spurious pre-step evaluation land on raw=0 (CRUISE) instead -- exactly
% what a real absence of input should mean, matching min_ttc=None's own
% None->math.inf mapping in StateflowFSM.step. now_risk/future_risk's '0'
% (NONE) was already the correct safe default and is unchanged.
add_block('simulink/Sources/Constant', [modelName '/min_ttc']);
add_block('simulink/Sources/Constant', [modelName '/now_risk']);
add_block('simulink/Sources/Constant', [modelName '/future_risk']);
set_param([modelName '/min_ttc'], 'Value', '999', 'Position', [30 40 60 60]);
set_param([modelName '/now_risk'], 'Value', '0', 'Position', [30 100 60 120]);
set_param([modelName '/future_risk'], 'Value', '0', 'Position', [30 160 60 180]);

% ---- Output: Terminator -- read live via the chart block's own
% RuntimeObject.OutputPort(1).Data during pauses instead (To Workspace
% was tried first but only flushes its buffer when the simulation fully
% STOPS, not on pause/step -- useless for reading intermediate values).
add_block('simulink/Sinks/Terminator', [modelName '/level_out']);
set_param([modelName '/level_out'], 'Position', [520 100 550 120]);

% ---- MATLAB Function block: raw = compute_raw_level(...) ----
rawBlock = [modelName '/compute_raw_level'];
add_block('simulink/User-Defined Functions/MATLAB Function', rawBlock);
set_param(rawBlock, 'Position', [150 40 320 180]);
rt = sfroot;
rawChart = rt.find('-isa', 'Stateflow.EMChart', '-and', 'Path', ...
    [modelName '/compute_raw_level']);
if isempty(rawChart)
    rawChart = rt.find('-isa', 'Stateflow.EMChart');
    rawChart = rawChart(1);
end
rawScript = [ ...
    "function raw = compute_raw_level(min_ttc, now_risk, future_risk)" newline ...
    "% Direct transliteration of M5_Pipeline/fsm.py's _raw_level -- branch" newline ...
    "% ORDER matters (AVOID-before-YIELD live-incident fix, see fsm.py's" newline ...
    "% own docstring/README for why). RED=3, YELLOW=2, GREEN=1, NONE=0." newline ...
    "TTC_EMERGENCY_S = 1.5;" newline ...
    "TTC_AVOID_S = 4.0;" newline ...
    "TTC_FOLLOW_S = 8.0;" newline ...
    "RED = 3; YELLOW = 2;" newline ...
    "" newline ...
    "if min_ttc <= TTC_EMERGENCY_S" newline ...
    "    raw = 4; return;" newline ...
    "end" newline ...
    "if now_risk == RED" newline ...
    "    raw = 4; return;" newline ...
    "end" newline ...
    "if min_ttc <= TTC_AVOID_S" newline ...
    "    raw = 2; return;" newline ...
    "end" newline ...
    "if now_risk == RED || now_risk == YELLOW" newline ...
    "    raw = 2; return;" newline ...
    "end" newline ...
    "if future_risk == RED || future_risk == YELLOW" newline ...
    "    raw = 3; return;" newline ...
    "end" newline ...
    "if min_ttc <= TTC_FOLLOW_S" newline ...
    "    raw = 1; return;" newline ...
    "end" newline ...
    "raw = 0;" newline ...
    "end" newline ...
];
rawChart.Script = char(rawScript);

% ---- Stateflow chart: state transition logic, own persistent `level` ----
chartBlock = [modelName '/behavior_fsm_chart'];
add_block('sflib/Chart', chartBlock);
set_param(chartBlock, 'Position', [370 40 500 220]);
chart = rt.find('-isa', 'Stateflow.Chart', '-and', 'Path', chartBlock);
if isempty(chart)
    allCharts = rt.find('-isa', 'Stateflow.Chart');
    for i = 1:numel(allCharts)
        if ~isempty(strfind(allCharts(i).Path, 'behavior_fsm_chart'))
            chart = allCharts(i);
            break;
        end
    end
end
chart.ActionLanguage = 'MATLAB';
chart.ChartUpdate = 'INHERITED';

dRaw = Stateflow.Data(chart);
dRaw.Name = 'raw'; dRaw.Scope = 'Input'; dRaw.DataType = 'double';

dNow = Stateflow.Data(chart);
dNow.Name = 'now_risk_c'; dNow.Scope = 'Input'; dNow.DataType = 'double';

dFuture = Stateflow.Data(chart);
dFuture.Name = 'future_risk_c'; dFuture.Scope = 'Input'; dFuture.DataType = 'double';

dTtc = Stateflow.Data(chart);
dTtc.Name = 'min_ttc_c'; dTtc.Scope = 'Input'; dTtc.DataType = 'double';

dOut = Stateflow.Data(chart);
dOut.Name = 'level_out'; dOut.Scope = 'Output'; dOut.DataType = 'double';

names = {'CRUISE', 'FOLLOW', 'AVOID', 'YIELD', 'EMERGENCY_BRAKE'};
positions = [40 40 120 60; 200 40 120 60; 40 140 120 60; 200 140 120 60; 120 240 160 60];
states = cell(1, 5);
for i = 1:5
    s = Stateflow.State(chart);
    s.Position = positions(i, :);
    s.LabelString = sprintf('%s\nentry: level_out = %d;', names{i}, i - 1);
    states{i} = s;
end

% Plain unconditional default transition into CRUISE -- fires once at
% simulation start (t=0), same as fsm.py's BehaviorFSM.__init__ starting
% at level 0. Every subsequent manual 'step' evaluates the ALREADY-active
% state's own outgoing transitions -- no per-call state restoration needed.
defaultT = Stateflow.Transition(chart);
defaultT.Destination = states{1};
defaultT.DestinationOClock = 0;

clearedExprs = { ...
    '', ...
    '(min_ttc_c > 8.0 + 0.5)', ...
    '(min_ttc_c > 4.0 + 0.5) && (now_risk_c == 0)', ...
    '(future_risk_c == 0)', ...
    '(min_ttc_c > 1.5 + 0.5) && (now_risk_c ~= 3)' ...
};

for i = 1:5
    myLevel = i - 1;
    for j = 1:5
        targetLevel = j - 1;
        if targetLevel > myLevel
            t = Stateflow.Transition(chart);
            t.Source = states{i};
            t.Destination = states{j};
            t.LabelString = sprintf('[raw == %d]', targetLevel);
        end
    end
    if ~isempty(clearedExprs{i})
        for j = 1:5
            targetLevel = j - 1;
            if targetLevel < myLevel
                t = Stateflow.Transition(chart);
                t.Source = states{i};
                t.Destination = states{j};
                t.LabelString = sprintf('[(raw == %d) && %s]', targetLevel, clearedExprs{i});
            end
        end
    end
end

add_line(modelName, 'min_ttc/1', 'compute_raw_level/1', 'autorouting', 'on');
add_line(modelName, 'now_risk/1', 'compute_raw_level/2', 'autorouting', 'on');
add_line(modelName, 'future_risk/1', 'compute_raw_level/3', 'autorouting', 'on');

add_line(modelName, 'min_ttc/1', 'behavior_fsm_chart/4', 'autorouting', 'on');
add_line(modelName, 'now_risk/1', 'behavior_fsm_chart/2', 'autorouting', 'on');
add_line(modelName, 'future_risk/1', 'behavior_fsm_chart/3', 'autorouting', 'on');
add_line(modelName, 'compute_raw_level/1', 'behavior_fsm_chart/1', 'autorouting', 'on');
add_line(modelName, 'behavior_fsm_chart/1', 'level_out/1', 'autorouting', 'on');

save_system(modelName, fullfile(fileparts(mfilename('fullpath')), 'behavior_fsm.slx'));
close_system(modelName, 0);
fprintf('behavior_fsm.slx built successfully.\n');
end
