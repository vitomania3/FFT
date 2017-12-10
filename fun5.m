function res = fun(x)
    if isempty(find(x == 0))
        res = sqrt(abs(x)) .* sin(1 ./ (x .^ 2));
    else
        res = zeros(1, numel(x));
    end
end