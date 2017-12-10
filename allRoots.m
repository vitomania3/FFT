function roots = allRoots1(f, n, xmin, xmax, accuracy)
    for i = 1 : n
        root = findRoot(f, xmin, xmax, accuracy);
        if isnan(root)
           disp('The roots are very close or roots absent');
           roots = [];
           return;
        end
        roots(i, :) = root;
        f = @ (x) f(x) ./ (x - root * ones(1, numel(x)));
    end
    
end

function root = findRoot(f, left, right, accuracy)
    shift = (right - left) / 1000;
    
    while norm(right) > accuracy
        right 
        left
        if f(left) .* f(right) <= 0
            [root, ~] = bisection(f, left, right, accuracy);
            return;
        end;
        right = right - shift;% * ones(1, numel(right));
    end
    
    root = NaN;
end
