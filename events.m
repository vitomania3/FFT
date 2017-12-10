function [value, isterminal, direction] = events(t, h)
    value = h(1); % для слежения за обращением в 0 величины у(1)
    isterminal = 1 ; % прекратить при value = 0
    direction = -1; % при условии убывания value
end

