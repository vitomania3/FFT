clear;
clc;

f = figure;
% plotFT1(f, @func2, @analytical2, 1, [-50 300], [-10 10]);

% plotFT1(f, @func1, @analytical1, 0.0001, [-400 1000], [-3 3]);

plotFT1(f, @func4, [], 0.01, [-100 100], [-15 15]);

% print('Graph', '-dpng')

step = 1;

shift = pi / step;

% plotFT_shift1(shift, f, @func2, @analytical2, step, [-20 50], [-7 7]);