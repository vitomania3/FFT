%%                                          Task 3

clear;
clc;

n = input('Enter n: ');
psi = @ (n) 1 ./ ((n + 1) .^ 2);
Sn = @ (x) sum(1 ./ factorial(x));


S = exp(1);

range = 0 : n;
plot(range, psi(range), 'b');
hold on;
for i = 0 : n
   ans(i + 1) = Sn(0 : i) ;
end

plot(range, ans - S * ones(1, n + 1), 'r');
% plot(range, abs(psi(range) - S * ones(1, n + 1)), 'g');
legend('psi(n)', 'Sn - S');
grid on;    
abs(psi(range) - S * ones(1, n + 1))
%%                                          Task 4

clear;
clc;

x = linspace(0, 5 * pi, 100);
func = @ (x) tan(x) - sqrt(x);
plot(x, sqrt(x), 'b', x, tan(x), 'r', x, func(x), 'c');
grid on;
axis([x(1) - 1 x(end) + 1 min(func(x)) - 1 max(func(x)) + 1]);
legend('sqrt(x)', 'tg(x)');

[coordX, coordY] = ginput(2)

% if coordX(1) < 0 || coordX(2) < 0 % Domain sqrt(x)
%    warning('x must be positive');
%    return;
% end

% coordX ~= pi / 2 + pi * k
% if mod(coordX(1), pi / 2) == 0 || mod(coordX(1), 3 * pi / 2) == 0 || ...
%    mod(coordX(2), pi / 2) == 0 || mod(coordX(2), 3 * pi / 2) == 0 % Break point
%     warning('tg(x) must be finite');
%     return;
% end

% if (func(coordX(1)) > 0 && func(coordX(2)) > 0) || (func(coordX(1)) < 0 && func(coordX(2)) < 0)
%     warning('Function must differ in sign');
%     return;
% end

x0 = [coordX(1)];
res = fzero(func, x0);
fprintf("Answer: %f\n", res);

%%                                          Task 5

clear;
clc;

lBorder = -1;
rBorder = 1;
nPoints = 101;

x = linspace(lBorder, rBorder, nPoints);
% fun = @ (x) sqrt(abs(x)) .* sin(1 ./ (x .^ 2));

for i = 1 : numel(x)
%     if x(i) == 0
%         res(i) = fzero(@fun, eps);
%     else
        res(i) = fzero(@fun5, x(i));
%     end
end

plot(x, res, 'b-');
grid on;

%%                                              Task 6

clear;
clc;

dim = 2;
f = @ (x) sin(x);
% f = @ (x) 1;
% f = @ (x) (x - 0.00001) * (x - 0.00002);
% f = @ (x) sin(x);
% f = @ (x) (x - 1) .* (x - 2) .* (x - 3);
% f = @ (x) (norm(x) - 1) .* (norm(x) - 2) .* (norm(x) - 3);
n = 1;
% xmin = -3;
% xmax = 3;
xmin = [-2 -2];
xmax = [3 3];
% % xmin = -pi * ones(1, dim);
% xmax = 2 * pi * ones(1, dim);
accuracy = 0.00001;
roots = allRoots(f, n, xmin, xmax, accuracy)

%%                                              Task 9

clear;
clc;

% nodeName = 'system_1';
% matrix = [-1 2; 1 -2];
% nodeName = 'zzz';
% [nodeName, matrix] = Nodes('centerNode'); % saddleNode focusNode decriticalNode centerNode stableNode
n = 20; % number of points

% opt = odeset('OutputSel', [1 2], 'OutputFcn', 'odephas2');
% [T, X] = ode45(nodeName, [0 n], [0 1], opt);

grid on;
hold on;
for k = 1 : n
    split = linspace(-n, n, n);
    [T, X] = ode45(nodeName, [0 n], [split(k) n]);
    drowArrow(X, matrix, n);
    [T, X] = ode45(nodeName, [0 n], [split(k) -n]);
    drowArrow(X, matrix, n);
    [T, X] = ode45(nodeName, [0 n], [-n split(k)]);
    drowArrow(X, matrix, n);
    [T, X] = ode45(nodeName, [0 n], [n split(k)]);
    drowArrow(X, matrix, n);
%     [T, X] = ode45(nodeName, [0 n], [i j]);
%     drowArrow(X);
%     [T, X] = ode45(nodeName, [0 n], [-i j]);
%     drowArrow(X);
%     [T, X] = ode45(nodeName, [0 n], [-i -j]);
%     drowArrow(X);
end

%%                                          Task 10

%% 1 system

clear;
clc;

w = @ (x, y) (x + y) .^ 2 + y .^ 4 ./ 2; % Lyapunov function

n = 30;
[x, y] = meshgrid(-5 : 0.1 : 5);
z = w(x, y);
contour(x, y, z, 30, 'k');

hold on;

r = 4;
tetta = linspace(0, 2 * pi, n);

for i = 1 : n
    y0 = [r * cos(tetta(i)), r * sin(tetta(i))];
    [T, X] = ode45(@system_1, [0 n], y0);  
    [szX, szY] = size(X);
    x = X(1:szX, 1)';
    y = X(1:szX, 2)';
    l = 1 : szX - 1;
    dx = x(l + 1) - x(l);
    dx(end + 1) = dx(end);
    dy = y(l + 1) - y(l);
    dy(end + 1) = dy(end);
    
    d = 20;
    xx1 = x(1 : d);
    yy1 = y(1 : d);
    xx2 = x(d : end);
    yy2 = y(d : end);
    
    dxx1 = dx(1 : d);
    dyy1 = dy(1 : d);
    dxx2 = dx(d : end);
    dyy2 = dy(d : end);
    
    q = quiver(xx1, yy1, dxx1, dyy1, 0);
    q.Color = [0 0 1];
    q = quiver(xx2, yy2, dxx2, dyy2, 0);
    q.Color = [1 0 0];
    hold on;
    axis([-r r -r r]);
end

%% 2 system

clear;
clc;

w = @ (x, y) x .* y; % Lyapunov function

[x, y] = meshgrid(-5 : 0.1 : 5);
z = w(x, y);
contour(x, y, z, 70, 'k');

hold on;
n = 30;

r = 4;
tetta = linspace(0, 2 * pi, n);

for i = 1 : n
    y0 = [r * cos(tetta(i)), r * sin(tetta(i))];
    [T, X] = ode45(@system_2, [0 n], y0);  
    [szX, szY] = size(X);
    x = X(1:szX, 1)';
    y = X(1:szX, 2)';
    l = 1 : szX - 1;
    dx = x(l + 1) - x(l);
    dx(end + 1) = dx(end);
    dy = y(l + 1) - y(l);
    dy(end + 1) = dy(end);
    
%     d = fix(numel(x) / 3);
    d = 20;
    xx1 = x(1 : d);
    yy1 = y(1 : d);
    xx2 = x(d : end);
    yy2 = y(d : end);
    
    dxx1 = dx(1 : d);
    dyy1 = dy(1 : d);
    dxx2 = dx(d : end);
    dyy2 = dy(d : end);
    
    q = quiver(xx1, yy1, dxx1, dyy1, 0);
    q.Color = [0 0 1];
    q = quiver(xx2, yy2, dxx2, dyy2, 0);
    q.Color = [1 0 0];
    hold on;
    axis([-r r -r r]);
end

%%                                                  Task 11

clear;
clc;

N = 100;
dx = linspace(0, 1, N);
handle1 = @ (x) [exp(x) - 2; exp(x)];
solinit = bvpinit(dx, handle1);
ode2 = @ (x, y) [y(2); y(2)];

bc2 = @ (ya, yb) [ya(1) + 1, yb(2) - yb(1) - 2];

sol = bvp4c(ode2, bc2, solinit)

% CТРУКТУРА sol:  
% sol.x – ВЕКТОР, В КОТОРМ СОДЕРЖАТЬСЯ КООРДИНАТЫ УЗЛОВ СЕТКИ
% sol.y – МАТРИЦА, СОСТОЯЩАЯ ИЗ ДВУХ СТРОК,
% sol.y(1, :) – СООТВЕТСТВУЕТ ЗНАЧЕНИЯМ ФУНКЦИИ y1 
% sol.y(2, :) - СООТВЕТСТВУЕТ ЗНАЧЕНИЯМ ФУНКЦИИ y2
plot(sol.x, sol.y(1, : ), 'LineWidth', 2, 'Color', 'b')
hold on;
grid on
func = @ (x) exp(x) - 2;
t = linspace(0, 1, N);
plot(t, func(t), 'r')
legend('apposterior', 'apriori')
xlabel('t')

norma_C = max(abs(func(t) - sol.y(1, : )))
norma_Lp = sqrt(trapz(t, (func(t) - sol.y(1, : )) .^ 2))

%%                                          Task 12

%% Example 1

clear;
clc;

fun = @ (x, y) exp(x .^ 2 + y .^ 2);

Hessian = @ (x) [2 * (2 * x(1) .^ 2 + 1) .* fun(x(1), x(2)), 4 * x(1) .* x(2) .* fun(x(1), x(2));
                4 * x(1) .* x(2) .* fun(x(1), x(2)), 2 * (2 * x(2) .^ 2 + 1) .* fun(x(1), x(2))];
gradFun = @ (x) [2 * x(1) .* fun(x(1), x(2)); 
                 2 * x(2) .* fun(x(1), x(2))];

x0 = [1, 2]';

%% Example 2

clear;
clc;

fun = @ (x, y) x .^ 2 + 2 * y .^ 2 + x .^ 2 .* y .^ 2 + exp(y .^ 2) - y; 

Hessian = @ (x) [2 + 2 * x(2) .^ 2, 4 .* x(1) .* x(2); 
                 4 .* x(1) .* x(2), 4 + 2 * x(1) .^ 2 + 2 * exp(x(2) .^ 2) + 4 * x(2) .^ 2 .* exp(x(2) .^ 2)];
gradFun = @ (x) [2 * x(1) + 2 * x(1) .^ 2 .* x(2);
                 4 * x(2) + 2 * x(1) .^ 2 .* x(2) + 2 * x(2) .* exp(x(2) .^ 2) - 1];

x0 = [1 2]';

%%

x(:, 1) = x0;

for i = 1 : 15
    x(:, i + 1) = x(:, i) - inv(Hessian(x(:, i))) * gradFun(x(:, i))
end
[X, Y] = meshgrid(-2 : 0.1 : 2);
Z = fun(X, Y);

surf(X, Y, Z);
hold on;
for i = 1 : 10
   plot3(x(1, i), x(2, i), fun(x(1, i), x(2, i)), 'rX', 'LineWidth', 5); 
end

figure;
for i = 2 : 10
    c = contour(X, Y, Z, i);
    hold on;
    plot(c(1, i), c(2, i), 'rX'); 
end

legend('f(x, y)', 'minimum')
xlabel('x');
ylabel('y');
zlabel('f(x, y)');

%% Example 3

clear;
clc;

fun = @ (x) (x + 1) .^ 2;
Hessian = @ (x) [2];
gradFun = @ (x) [2 * (x(1) + 1)];

x0 = [10];

x = x0;

y = x' - inv(Hessian(x)) * gradFun(x)';

a = -4;
step = 0.1;
b = 4;
split = [a : step : b];
min_fminbnd = fminbnd(fun, a, b);

a = -4;
step = 0.1;
b = 4;
split = [a : step : b];
plot(split, fun(split), 'b');
hold on;
grid on;
plot(y(1), fun(y(1)), 'ro', 'LineWidth', 3)
plot(min_fminbnd, fun(min_fminbnd), 'gX', 'LineWidth', 3)
legend('f(x, y)', 'My min', 'fminbnd min')
xlabel('x');
ylabel('y');
zlabel('f(x, y)');

%%                                              Task 7

clear;
clc;

global alpha;
x0 = [0 2];
step = 0.01;
alpha = 1;
timeout = 20;

tt = [];
yy = [];
T = [0];
for t = 1 : timeout
    pause(0.01)
    tspan = [0 : step : t];
    
%     [T, Y] = ode45(@fun7, tspan, x0);
%     plot(T, abs(Y(:, 1)), 'b')

%     options = odeset('Events', @events);
    [T, Y] = ode45(@fun7, tspan, x0);
%     tt = cat(2, tt, [T(end) numel(tspan)]);
%     yy = cat(2, yy, [Y()]);
    plot(T, abs(Y(:, 1)), 'r')
    hold on;
%     options1 = odeset('Events', @events1);
%     [T, Y] = ode45(@func71, tspan, x0, options1);
%     tt = cat(2, tt, [T(end) : step : t + step]);
%     plot([T(end) : step : t + step], Y(:numel(T), 1), 'g')
%     options1 = odeset('Events', @events1);
%     [T, Y] = ode45(@func71, tspan, [2 0]);

%     [T, Y] = ode45(@fun7, tspan, x0);
%     t = cat(2, t, T');
%     y = cat(2, y, Y(:, 1)');
%     [T, Y] = ode45(@func71, tspan, x0);
%     plot(T, Y(:, 1), 'r')
    legend('x'''' = -ax'' - sin(x)')
    axis([0 timeout + 1 -5 5]);
    grid on;
    hold on;
end

%%                                              Task 8

%% Example 1

clear;
clc;

global G m1 m2;
G = 6.67 * 10 ^(-3);
m1 = 100000000;
m2 = 100;

x10 = [0; 0];
x20 = [0; 25];
dx1_dt0 = [5; 5];
dx2_dt0 = [10; 0];
y0 = [x10; x20; dx1_dt0; dx2_dt0];
    
t0 = 0;
t1 = 15;

%% Example 2

clear;
clc;

global G m1 m2;
G = 6.67 * 10 ^(-3);
m1 = 400000;
m2 = 100;

x10 = [0; 0];
x20 = [0; 25];
dx1_dt0 = [5; 5];
dx2_dt0 = [10; 0];
y0 = [x10; x20; dx1_dt0; dx2_dt0];
    
t0 = 0;
t1 = 15;

%% Example 3

clear;
clc;

global G m1 m2;
G = 6.67 * 10 ^(-3);
m1 = 400000;
m2 = 100;

x10 = [0; 0];
x20 = [0; 0.4];
dx1_dt0 = [5; 5];
dx2_dt0 = [-5; -5];
y0 = [x10; x20; dx1_dt0; dx2_dt0];
    
t0 = 0;
t1 = 10;

%% Example 4
clear;
clc;

global G m1 m2;
G = 6.67 * 10 ^(-1);
m1 = 20;
m2 = 100;

x10 = [0; 0];
x20 = [0; 0.4];
dx1_dt0 = [5; 5];
dx2_dt0 = [-5; -5];
y0 = [x10; x20; dx1_dt0; dx2_dt0];
    
t0 = 0;
t1 = 10;

%%    
[t, y] = ode45(@fun18, [t0, t1], y0);
        
figure(1);
cla;
hold on;
title(['G = ', num2str(G), ' m1 = ', num2str(m1), ' m2 = ', num2str(m2)]);
n = min(size(t, 1), 500);
% szPlane = 10;
% szStep = 1;

szPlane = 200;
szStep = 20;


a=-2;
b=3;
c=1;
[X,Y] = meshgrid(-szPlane : szStep : szPlane, -szPlane : szStep : szPlane);

% Z=c*(1-X-Y);
ss = 500;
Z = meshgrid(-ss : 50 : ss);
X = zeros(size(X, 1), size(X, 1));
% figure('Color','w')
hold on;
hS=mesh(X, Y, Z);
xlabel('x');
ylabel('y');
zlabel('z')

for i = 1 : n
%     t = linspace(1, 100, numel(y(:, 1)));
    az = 60;
    el = 45;
    view(az, el);
    
    s = numel(y(1:i, 1));
    plot3(zeros(1, s), y(1 : i, 2), [1:s], 'r');

%     plot3(y(1 : i, 1), y(1 : i, 2), c * (1 - y(1 : i, 1) - y(1 : i, 2)), 'r');
%     plot3(y(1 : i, 1), y(1 : i, 2), t(1 : i), 'r');
    hold on;
    grid on;
    
    plot3(zeros(1, s), y(1 : i, 4), [1:s], 'b');
%     plot3(y(1 : i, 3), y(1 : i, 4), c * (1 - y(1 : i, 3) - y(1 : i, 4)), 'b');
%     plot3(y(1 : i, 3), y(1 : i, 4), t(1 : i), 'b');
    pause(0.01);
end;
hold off