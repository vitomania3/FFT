function res = func4(t)
    res = exp(-5 * abs(t)) .* log(3 + t .^ 4);
end

