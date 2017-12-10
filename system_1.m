function f = system_1(t, x)
    f(1) = - x(1) + 2 * x(2) - x(2) .^ 3;
    f(2) = x(1) - 2 * x(2);
    f = f';
end

