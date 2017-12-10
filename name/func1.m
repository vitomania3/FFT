function res = func1(t)
    res = (1 - cos(2 * t)) ./ (2 * t);
    res(abs(t) < eps) = 0;
    
end

