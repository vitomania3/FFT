function f = system_2(t, x)
    f(1) = x(1) * x(2) - x(1) .^ 3 + x(2) .^ 3;
    f(2) = x(1) .^ 2 - x(2) .^ 3;
    f = f';
end

