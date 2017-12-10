% Laboratory Work #3
% Numerical Methods
% Option #2

%% Task #1

clear;
clc;

newplot;

fig = gcf;

% ftfunc1 = @(l) - 1 ./ 2 .* (1i .* l) .* sqrt(pi) .* exp((- l .^ 2) ./ 4);

func2 = @(t) feval(func2(t));
ftfunc2 = @(l) - 1i .* (pi .* sign(l - 1) + pi .* sign(l + 1) - 4 .* atan(l)) ./ 2;

% func3 = @(t) exp(- 2 .* abs(t)) ./ (1 + cos(t) .^ 2);

% func4 = @(t) 2 ./ (3 + 4 .* (t .^ 4));

% func5 = @(t) exp(-t .^ 2 ./ 2);
% ftfunc5 = @(t) sqrt(2 * pi) * func5(t);

step = 0.0001;

left_point_in = -500;
right_point_in = 1000;

inpLimVec = [left_point_in, right_point_in];

left_point_out = -2;
right_point_out = 2;

outLimVec = [left_point_out, right_point_out];

% plotFT(fig, @func4, [], step, inpLimVec, outLimVec);

shift = 2*pi/step;
% 
plotFT_shift(shift, fig, func2, ftfunc2, step, inpLimVec, outLimVec);

%% Task #2

% alive

%% Task #3

clear;
clc;

n_max = 100;
N_splitting = 1 : n_max;
u_n = @(n) ((-1) .^ (n - 1)) ./ n;
S_n = @(n) cumsum(u_n(n));
S_ns = S_n(N_splitting);
S = log(2);
plot(N_splitting, S_ns - S);
hold on;
grid on;
psi_apriori = @(n) 1 ./ (n + 1);
psi_real = @(n) abs(S_ns - S);
plot(N_splitting, psi_apriori(N_splitting), 'k', N_splitting, psi_real(N_splitting));          % approximation plot
hleg = legend('Sn - S', 'Psi_n(x) (a priori error estimate)', '|Sn - S| (real error)');
set(hleg, 'Interpreter', 'none');
xlabel('x');
ylabel('S');

%% Task #4

clear;
clc;

left_point = - 1;
right_point = 2;

% real root equals 0.95193

step = 0.00001;

splitting = left_point : step : right_point;

func_left = @(x) sin(x);
values_left = func_left(splitting);

func_right = @(x) x .^ 3 + x - 1;
values_right = func_right(splitting);

func = @(x) func_left(x) - func_right(x);

plot(splitting, values_left, 'r-', splitting, values_right, 'b-');

grid on;

number_iters = 2;

[x0, ~] = ginput(number_iters);

for i = 1 : number_iters
    [roots, val] = fzero(func, x0(i));

    disp(['Root = ', num2str(roots)]);
    disp(['Difference between left and right parts of equation = ', num2str(val)]);
end

%% Task #5

clear;
clc;

func = @(x) x .* cos(log(abs(x)));

left_point = -1;
right_point = 1;

step = 0.001;
splitting = left_point : step : right_point;
len = numel(splitting);

zero_arg = find(splitting == 0);

values = func(splitting);

values(zero_arg) = 0;

plot(splitting, values, 'r-');
grid on; 
xlabel('x');
ylabel('y');
title('f(x)');
figure;

estimation = splitting;
%eps1 = 0.01;
eps2 = 0.0001 / 2;
roots = zeros(1, len);
for i = 1 : len
    if i == zero_arg
        func = @(x) 0 .* x;
        %eps_func = @(x) 0 .* x;
    else
        func = @(x) x .* cos(log(abs(x)));
        %eps_func = @(x) (x + estimation(i) - eps1) .* cos(log(abs(x + estimation(i) - eps1)));
    end
    %func(x + estimation(i) - eps1);
%     try
%         roots(i) = fzero(func, [estimation(i) - eps2, estimation(i) + eps2]);
%     catch
%         roots(i) = fzero(func, estimation(i));
%     end
    %options = optimset('TolX', eps);
    
    %opts = optimset('Display','iter');%, 'TolX', 1);
    
    roots(i) = fzero(func, estimation(i));
%     
    %roots(i) = fzero(eps_func, eps1) + estimation(i) - eps1;%, opts);
    %disp([num2str(roots(i)), ' ', num2str(estimation(i))]);
    
    if ((sign(roots(i)) * sign(estimation(i)) ~= 1) || (abs(roots(i)) < eps2))
        roots(i) = roots(i - 1);
    end

%     old_root = roots(i);
%     dist = abs(estimation(i) - roots(i));
%     try
%         roots(i) = fzero(func, [estimation(i) - dist + eps2, estimation(i) + dist - eps2]);
%     catch
%         roots(i) = old_root;
%     end
end
stairs(estimation, roots, 'b');
grid on;

xlabel('Initial estimation');
ylabel('Root');

title('Roots of f(x)');

%% Task #6

%% example #1 (k = 2)

clear;
clc;

f = @(x) norm(x(:) - [1; 1]) * norm(x(:) - [-3; 3]) * norm(x(:) - [-3; -2]);
%f = @(x) cos(x(1)) + sin(x(2));
k = 2;
n = 18;
xmin = -5;
xmax = 12;
allRoots(f, n, k, xmin, xmax);

%% example #2 (k = 2)

clear;
clc;

f = @(x) (x(1) + 2) .^ 2 + (x(2) - 3) .^ 2;
k = 2;
n = 2;
xmin = -2;
xmax = 3;
allRoots(f, n, k, xmin, xmax);

%% example #3 (k = 3)

clear;
clc;

f = @(x) (x(1) - 2) .^ 2 + (x(2) - 3) .^ 2 + sin(x(3));
k = 3;
n = 80;
xmin = 2;
xmax = 7;
allRoots(f, n, k, xmin, xmax);

%% example #4 (k = 4)

clear;
clc;

f = @(x) x(1) + x(2) + x(3) + x(4);
k = 4;
n = 1600;
xmin = -7;
xmax = 7;
allRoots(f, n, k, xmin, xmax);

%% example #5 (k = 5)

clear;
clc;

f = @(x) (x(1) - 1) .* x(2) .* (x(3) + 2) .* (x(4) - 5) .* x(5);
k = 5;
n = 50;
xmin = -4;
xmax = 4;
allRoots(f, n, k, xmin, xmax);

%% example #6 (k = 1)

clear;
clc;

f = @(x) sin(1 ./ abs(x));
k = 1;
n = 10;
xmin = -1;
xmax = 1;
allRoots(f, n, k, xmin, xmax);

%% Task #7

clear;
clc;

x_min = -0.5;
x_max = 0.3;

y_min = -1;
y_max = 1.7;

t_left = 0;
t_right = 15;

trimmer = 0.5;

x0 = [-0.4; 0];
v0 = [-1; -2];
y0 = [x0(1); v0(1); x0(2); v0(2)];

times = t_left;
trajectory = y0;

step = 0.01;

t_splitting = t_left : step : t_right;

old_Y = y0';

for i = 1 : (numel(t_splitting) - 1)
    [T, Y] = ode45(@odefun, [t_splitting(i), t_splitting(i + 1)], y0);
    
    times = [times; T];
    trajectory = [trajectory, Y'];
    
    if (Y(end, 1) > x_max)
        trajectory(1, end) = x_max;
        y0(1) = x_max;
        y0(2) = - Y(end, 2);
    else
    if (Y(end, 1) < x_min)
        trajectory(1, end) = x_min;
        y0(1) = x_min;
        y0(2) = - Y(end, 2);
    else
        y0(1) = Y(end, 1);
        y0(2) = Y(end, 2);
    end
    end
    if (Y(end, 3) > y_max)
        trajectory(3, end) = y_max;
        y0(3) = y_max;
        y0(4) = - Y(end, 4);
    else
    if (Y(end, 3) < y_min)
        trajectory(3, end) = y_min;
        y0(3) = y_min;
        y0(4) = - Y(end, 4);
    else
        y0(3) = Y(end, 3);
        y0(4) = Y(end, 4);
    end
    end
    new_Y = cat(1, old_Y, Y);
    plot(new_Y(:, 1), new_Y(:, 3), 'r');
    %plot(trajectory(1, :), trajectory(3, :), 'r');
    hold on;
    plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min], 'b-');
    %legend('trajection', 'room');
    axis([x_min - trimmer, x_max + trimmer, y_min - trimmer, y_max + trimmer]);
    grid on;
    pause(0.01);
    old_Y = Y;
end
hold off;
%plot(trajectory(1, :), trajectory(3, :), 'r');

%% Task #7 modified

clear;
clc;

x_min = -0.5;
x_max = 0.3;

y_min = -1;
y_max = 1.7;

t_left = 0;
t_right = 1000;

trimmer = 0.5;

options = odeset('Events', @event);

x0 = [-0.4; 0];
v0 = [-1; -2];
y0 = [x0(1); v0(1); x0(2); v0(2)];

times = t_left;
trajectory = y0;

%step = 0.01;

%t_splitting = t_left : step : t_right;

old_Y = y0';

while 1
    [T, Y] = ode45(@odefun, [t_left, t_right], y0, options);
    
    disp(T(end));
    
    t_left = T(end);
    
    times = [times; T];
    trajectory = [trajectory, Y'];
    
    if (Y(end, 1) >= x_max)
        trajectory(1, end) = x_max;
        y0(1) = x_max;
        y0(2) = - Y(end, 2);
    else
    if (Y(end, 1) <= x_min)
        trajectory(1, end) = x_min;
        y0(1) = x_min;
        y0(2) = - Y(end, 2);
    else
        y0(1) = Y(end, 1);
        y0(2) = Y(end, 2);
    end
    end
    if (Y(end, 3) >= y_max)
        trajectory(3, end) = y_max;
        y0(3) = y_max;
        y0(4) = - Y(end, 4);
    else
    if (Y(end, 3) <= y_min)
        trajectory(3, end) = y_min;
        y0(3) = y_min;
        y0(4) = - Y(end, 4);
    else
        y0(3) = Y(end, 3);
        y0(4) = Y(end, 4);
    end
    end
    new_Y = cat(1, old_Y, Y);
    plot(new_Y(:, 1), new_Y(:, 3), 'r');
    %plot(trajectory(1, :), trajectory(3, :), 'r');
    hold on;
    plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min], 'b-');
    %legend('trajection', 'room');
    axis([x_min - trimmer, x_max + trimmer, y_min - trimmer, y_max + trimmer]);
    grid on;
    pause(0.001);
    old_Y = Y;
    if (T(end) >= t_right)
        break
    end
end
hold off;

%% Task #8

%% example 1

clear;
clc; 

%G = 6.67 * (10 ^ (-11)); 
G = 1;

m1 = 10; 
m2 = 10; 

x0 = [0, 0.1, 0, -0.1, 1, 0, -1, 0];

%% example 2

clear;
clc; 

%G = 6.67 * (10 ^ (-11)); 
G = 1;

m1 = 10; 
m2 = 10; 

x0 = [0, 2.5, 0, -2.5, 1, 0, -1, 0];

%% computing

interval = [0, 25];

[t, y] = ode45(@(t, x) odefun3(t, x, G, m1, m2), interval, x0); 

figure; 
cla;
hold on;
title('trajectors'); 
n = min(size(t, 1), 500); 
%n = size(t, 1);
for i = 1 : n 
plot(y(1 : i, 1), y(1 : i, 2), 'r', y(1 : i, 3), y(1 : i, 4), 'b');
grid on;
pause(0.01); 
end 

hold off;

%% Task #9

%% Stable node (0,0)

clear;
clc;

figure;
a = 2;
axis([-a, a, -a, a]);
axis equal;
hold on;
left_point = - pi;
right_point = pi;
number_of_lines = 40;
step = 2 * pi / number_of_lines;
interval = [0, 5];
line_width = 0.2;
for theta = left_point : step : right_point
    x0 = [a * cos(theta); a * sin(theta)];
    [t, x] = ode45(@StableNode, interval, x0);
    p1 = plot(x(:, 1), x(:, 2), 'r-', 'LineWidth', line_width);
    quiver(x(:, 1), x(:, 2), -x(:, 1), 2 * x(:, 1) - 2 * x(:, 2), 'r');
    grid on;
    xlabel('x');
    ylabel('y');
    title('Stable node');
end

x = linspace(-a, a, 10);
p2 = plot(0 * x, x, 'b--');
plot(x, 2 * x, 'b--');
hold off;
legend([p1, p2], 'Trajectory', 'Eigenvectors', 'Location', 'northwest');

%%  Dicritical node (0,0)

clear;
clc;

figure;
a = 3;
axis([-a, a, -a, a])
axis equal;
hold on;

left_point = - pi;
right_point = pi;
number_of_lines = 20;
step = 2 * pi / number_of_lines;
line_width = 0.2;
interval = [0, 5];
for theta = left_point : step : right_point
    x0 = [a * cos(theta); a * sin(theta)];
    [t, x] = ode45(@DicriticalNode, interval, x0);
    plot(x(:, 1), x(:, 2), 'r', 'LineWidth', line_width);
    quiver(x(:, 1), x(:, 2), -x(:, 1), -x(:, 2), 0.4, 'r');
    grid on;
    xlabel('x');
    ylabel('y');
    title('Stable Degenerate node');
end

hold off;

%% Stable Degenerate node (0,0)

clear;
clc;

figure;
a = 3;
axis([-a, a, -a, a]);
axis equal;
hold on;

left_point = - pi;
right_point = pi;
number_of_lines = 10;
step = 2 * pi / number_of_lines;
line_width = 0.2;
interval = [0, 5];
for theta = left_point : step : right_point
    x0 = [cos(theta); sin(theta)];
    [t, x] = ode45(@StableDegenerateNode, interval, x0);
    p1 = plot(x(:, 1), x(:, 2), 'r', 'LineWidth', line_width);
    quiver(x(:, 1), x(:, 2), -3 * x(:, 1) + 2 * x(:, 2), -2 * x(:, 1) + x(:, 2), 0.4, 'r');
    grid on;
    xlabel('x');
    ylabel('y');
    title('Stable Degenerate node');
end
x = linspace(-a, a, 10);
p2 = plot(x, x, 'b--');
hold off;
legend([p1, p2], 'Trajectory', 'Eigenvector', 'Location', 'northwest');

%% Saddle (0,0)

clear;
clc;

figure;
hold on;
a = 10;
interval = [0, 2];

for theta = (-a : a) / 2
    x0 = [theta, 2];
    [~, x] = ode45(@Saddle, interval, x0);
    p1 = plot(x(:, 1), x(:, 2), 'r');
    quiver(x(:, 1), x(:, 2), 4 * x(:, 1) - 2 * x(:, 2), -3 * x(:, 2), 0.4, 'r');
    x0 = [theta, -2];
    [t, x] = ode45(@Saddle, interval, x0);
    plot(x(:, 1), x(:, 2), 'r');
    quiver(x(:, 1), x(:, 2), 4 * x(:, 1) - 2 * x(:, 2), -3 * x(:, 2), 0.4, 'r');
    grid on;
    xlabel('x');
    ylabel('y');
    title('Saddle');
end

axis([-a, a, -2, 2]);
x = linspace(-a, a, 10);
p2 = plot(2 * x, 7 * x, 'b--');
plot(x, 0 * x, 'b--');
hold off;
legend([p1, p2], 'Trajectory', 'Eigenvectors', 'Location', 'northwest');

%% Centre (0, 0)

clear;
clc;

figure;
hold on;
quiver_step = 5;

left_point = - pi;
right_point = pi;
number_of_lines = 30;
step = 2 * pi / number_of_lines;
line_width = 0.2;
interval = [0, 4];
for theta = left_point : step : right_point
    x0 = [cos(theta); sin(theta)];
    [t, x]=ode45(@Centre, interval, x0);
    plot(x(:, 1), x(:, 2), 'r', 'LineWidth', line_width);
    quiver(x(1 : quiver_step : end, 1), x(1 : quiver_step : end, 2), x(1 : quiver_step : end, 1) - x(1 : quiver_step : end, 2), 2 * x(1 : quiver_step : end, 1) - x(1 : quiver_step : end, 2), 0.15, 'r') 
    grid on; 
    xlabel('x');
    ylabel('y');
    title('Centre');
end

%% Stable Focus (0, 0)

clear;
clc;

figure;
hold on;
a = 1;

left_point = - pi;
right_point = pi;
number_of_lines = 10;
step = 2 * pi / number_of_lines;
line_width = 0.2;
interval = [0, 10];
for theta = left_point : step : right_point
    x0 = [cos(theta); sin(theta)];
    [t, x] = ode45(@StableFocus, interval, x0);
    plot(x(:, 1), x(:, 2), 'r', 'LineWidth', line_width);
    quiver(x(:, 1), x(:, 2), -2 * x(:, 2), x(:, 1) - x(:, 2), 0.5, 'r');
    grid on;
    xlabel('x'); 
    ylabel('y');
    title('Stable Focus');
end

axis([-a, a, -a, a]);

%% unstable focus (0, 0)

clear;
clc;

figure;
hold on;
a = 1;

left_point = - pi;
right_point = pi;
number_of_lines = 10;
step = 2 * pi / number_of_lines;
line_width = 0.2;
interval = [0, 10];
for theta = left_point : step : right_point
    x0 = [cos(theta); sin(theta)];
    [t, x] = ode45(@StableFocus, interval, x0);
    plot(x(:, 1), x(:, 2), 'r', 'LineWidth', line_width);
    quiver(x(:, 1), x(:, 2), -(-2*x(:, 2)), -(x(:, 1) - x(:, 2)), 0.5, 'r');
    grid on;
    xlabel('x');
    ylabel('y');
    title('Unstable Focus');
end

axis([-a, a, -a, a]);

%% Task #10

%% First system

clear;
clc;

v1 = @ (x, y) x .^ 2 + y .^ 2;

t0 = -10;
t1 = 10;
number_of_points = 20;
left_point = 0;
right_point = 2 * pi;
tetta = linspace(left_point, right_point, number_of_points);
radius = 3;
max = 15;

figure(1);
cla;
axis([-max, max, -max, max]);
hold on;
step = 0.02;
[X, Y] = meshgrid(- max : step : max, - max : step : max);
Z = v1(X, Y);
colormap cool;
number_level_lines = number_of_points;
contourf(X, Y, Z, number_level_lines);
colorbar;

for i = 1 : number_of_points
    y0 = [radius * cos(tetta(i)), radius * sin(tetta(i))];
    [~, F] = ode45(@system1, [t0, t1], y0);
    
    %disp(size(F, 1));
    j1 = 0;
    for j = 1 : size(F, 1)
        j1 = j1 + 1;
        if (abs(F(j1, 1)) > max) || (abs(F(j1, 2)) > max)
            %disp("OK!");
            F(j1, :) = [];
            j1 = j1 - 1;
        end
    end
    
    plot(F(:, 1), F(:, 2), 'k', 'LineSmoothing', 'on');
    U = F(:, 1) .^ 3 - F(:, 2);
    V = F(:, 1) + F(:, 2) .^ 3;
    %q = 
    quiver(F(:, 1), F(:, 2), U, V, 'r');
    %q.AutoScaleFactor = 16.402;
end

title('first system');
legend('system trajectory', 'level lines');
hold off;

%% Second system

clear;
clc;

v2 = @(x, y) x .^ 2 + y .^ 4;

t0 = 5;
t1 = 100;
number_of_points = 40;
left_point = 0;
right_point = 2 * pi;
tetta = linspace(left_point, right_point, number_of_points);
radius = 5;
max = 7;

figure(2);
cla;
axis([- max, max, - max, max]);
hold on;
step = 0.02;
[X, Y] = meshgrid(- max : step : max, - max : step : max);
Z = v2(X, Y);
colormap cool;
number_level_lines = 20;
contourf(X, Y, Z, number_level_lines);
colorbar;

for i = 1 : number_of_points
    y0 = [radius * cos(tetta(i)), radius * sin(tetta(i))];
    [~, F] = ode45(@system2, [t0, t1], y0);
    j1 = 0;
    %F((abs(F(:, 1)) > max) || (abs(F(:, 2) > max))) = [];
    for j = 1 : size(F, 1)
        j1 = j1 + 1;
        if (abs(F(j1, 1)) > max) || (abs(F(j1, 2)) > max)
            F(j1, :) = [];
            j1 = j1 - 1;
        end
    end
    plot(F(:, 1), F(:, 2), 'k', 'LineSmoothing', 'on');
    U = 2 .* F(:, 2) .^ 3 - F(:, 1) .^ 5;
    V = - F(:, 1) - F(:, 2) .^ 3 + F(:, 2) .^ 5;
    quiver(F(:, 1), F(:, 2), U, V, 'r');
end

title('second system');
legend('system trajectory', 'level lines');
hold off;

%% Task #11

clear;
clc;

solinit = bvpinit(linspace(0, pi, 5), [0, 0]);
sol = bvp4c(@odefun2, @bcfun, solinit);

x = linspace(0, pi);
y = deval(sol, x);
values = y(1, :);
analytic_values = 0.42 * sin(x) + pi * cos(x) - pi + 2 .* x;
values_diff = abs(values - analytic_values);
plot(x, values, '.', x, analytic_values);

legend('numeric result', 'analytic result');
xlabel('x');
ylabel('y');
disp(['Difference in L2-norm: ', num2str(trapz(x, (values_diff .^ 2)) .^ (1 / 2))]);
max_diff = max(values_diff);
disp(['Difference in C-norm: ', num2str(max_diff)]);

%% Task #12

%% first function

clear;
clc;

fun = @(x) exp(x(1) .^ 2 + x(2) .^ 2);

diffs = cell(2);

diffs(1) = {@(x) 2 .* x(1) .* exp(x(1) .^ 2 + x(2) .^ 2)};
diffs(2) = {@(x) 2 .* x(2) .* exp(x(1) .^ 2 + x(2) .^ 2)};

[f_min, x_vect, x_array] = GaussSeidelMin(fun, diffs, [0.1, 0.1], 0.0001);

%% second function 

clear;
clc;

fun = @(x) sqrt(x(1) .^ 2 + x(2) .^ 2 + 1);

diffs = {@(x) x(1) ./ (sqrt(x(1) .^ 2 + x(2) .^ 2 + 1)), @(x) x(2) ./ (sqrt(x(1) .^ 2 + x(2) .^ 2 + 1))};

[f_min, x_vect, x_array] = GaussSeidelMin(fun, diffs, [0.1, 0.1], 0.00000001);

%% third function

clear;
clc;

eps = 0.0001;

interval = [0.5, 0.5];

fun = @(x) x(1) .^ 4 + x(2) .^ 4;
diffs = {@(x) 4 .* x(1) .^ 3, @(x) 4 .* x(2) .^ 3};

[f_min, x_vect, x_array] = GaussSeidelMin(fun, diffs, interval, eps, 0.05);

[X, Y] = meshgrid(x_array(1, :), x_array(2, :));
Z = X .^ 4 + Y .^ 4;
z = x_array(1, :) .^ 4 + x_array(2, :) .^ 4;
figure;
surf(X, Y, Z);
xlabel('x');
ylabel('y');
figure;
contour(X, Y, Z, z);
hold on;
plot(x_array(1, :), x_array(2, :), '*');
hold off;

y = 3;

options = optimset('TolX', eps);

[x, ~] = fminbnd(@(x) x .^ 4 + y .^ 4, -interval(1), interval(1), options);
[y, ~] = fminbnd(@(y) x .^ 4 + y .^ 4, -interval(2), interval(2), options);
disp('fminbnd output: ');
disp([x, y]);