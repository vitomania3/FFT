function f = func71(t, x)  
    global alpha;
    f(1) = -x(2);
    f(2) = - alpha * x(2) + sin(x(1));
    %     f(2) = - alpha^2 * x(2) - sin(x(1));
    f = f';
end