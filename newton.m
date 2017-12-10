function [outputArg1,outputArg2] = newton(x0, eps)
    x1 = x0 - F(x0) / F1(x0);
    while abs(F(x1)) > eps || abs(x0 - x1) > eps
        x0 = x1;
        x1 = x0 - F(x0) / F1(x0);
    end
    fprintf('root=%e',x1); % вывод найденного корня на экран
end

function y = F(x)
%     y = x - log(x) - 2;
    y = x .^ 2;
end

function y = F1(x)
%     y = 1 - (1 ./ x);
    y = 2 * x;
end

