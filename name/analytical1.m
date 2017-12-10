function res = analyitical1(lamda)
    res = zeros(1, numel(lamda));
    for j = 1 : numel(lamda)
        if lamda(j) < 2 && lamda(j) > 0
            res(j) = -pi * i / 2;
        elseif lamda(j) > -2 && lamda(j) < 0
            res(j) = pi * i / 2;
        elseif lamda(j) == -2
            res(j) = pi * i / 4;
        elseif lamda(j) == 2
            res(j) = -pi * i / 4;
        end
    end
end

