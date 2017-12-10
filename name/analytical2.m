function res = analyitical2(lamda)
    res = -i * sqrt(pi) / (4 * sqrt(2)) * lamda .* exp(-lamda .^ 2 / 8);
end

