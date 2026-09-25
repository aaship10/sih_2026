function benchmark_stepmode(n)
addpath(fileparts(mfilename('fullpath')));
mdl = 'behavior_fsm';
load_system(fullfile(fileparts(mfilename('fullpath')), 'behavior_fsm.slx'));
set_param(mdl, 'StopTime', 'inf');

set_param(mdl, 'SimulationCommand', 'start');
set_param(mdl, 'SimulationCommand', 'pause');
chartPath = [mdl '/behavior_fsm_chart'];

times = zeros(1, n);
minTtcBlk = [mdl '/min_ttc'];
nowRiskBlk = [mdl '/now_risk'];
futureRiskBlk = [mdl '/future_risk'];

for i = 1:n
    tic;
    set_param(minTtcBlk, 'Value', num2str(mod(i, 10)));
    set_param(nowRiskBlk, 'Value', num2str(mod(i, 4)));
    set_param(futureRiskBlk, 'Value', num2str(mod(i, 4)));
    set_param(mdl, 'SimulationCommand', 'step');
    rto = get_param(chartPath, 'RuntimeObject');
    level = rto.OutputPort(1).Data; %#ok<NASGU>
    times(i) = toc;
end

set_param(mdl, 'SimulationCommand', 'stop');
times_ms = times * 1000;
fprintf('n=%d median=%.3fms p95=%.3fms p99=%.3fms max=%.3fms first=%.3fms\n', ...
    n, median(times_ms), prctile(times_ms, 95), prctile(times_ms, 99), max(times_ms), times_ms(1));
close_system(mdl, 0);
end
