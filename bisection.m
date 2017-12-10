function [res, err] = bisection1(fun, left, right, tol)
   if fun(left) .* fun(right) > 0
      error('Значения функции на концах интервала должны быть разных знаков');
   end
  %Деление отрезка пополам
   middle = 0.5 * (left + right);
  %Итерационный цикл
    while norm(fun(middle)) > tol
        if norm(fun(left)) < tol
           res = left;
           err = norm(fun(middle));
           return;
        end
        if norm(fun(right)) < tol
           res = right;
           err = norm(fun(middle));
           return;
        end
     %Нахождение нового интервала
        left = left .* (fun(left) .* fun(middle) < 0) + middle .* (fun(left) .* fun(middle) > 0);
        right = right .* (fun(right) .* fun(middle) < 0) + middle .* (fun(right) .* fun(middle) > 0);
     %Деление нового интервала пополам
        middle = 0.5 * (left + right);
    end
   res = middle;
   err = norm(fun(middle));
end