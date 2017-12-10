function [systemFunc, matrix] = Nodes(nodeName)
    if strcmp(nodeName, 'saddleNode')
        systemFunc = @ saddleNode;
        matrix = [2 1; 1 -3];
    elseif strcmp(nodeName, 'focusNode')
        systemFunc = @ focusNode;
        matrix = [0 2; -3 -1];
    elseif strcmp(nodeName, 'decriticalNode')
        systemFunc = @ decriticalNode;
        matrix = [-1 0; 0 -1];
    elseif strcmp(nodeName, 'centerNode')
        systemFunc = @ centerNode;
        matrix = [1 -4; 2 -1];
    elseif strcmp(nodeName, 'stableNode')
        systemFunc = @ stableNode;
        matrix = [-4 -4; 1.5 1];
    end
end

function f = saddleNode(t, x)
    f(1) = 2 * x(1) + x(2);
    f(2) = x(1) - 3 * x(2);
    f = f';
end

function f = stableNode(t, x)  
    f(1) = -4 * x(1) - 4 * x(2);
    f(2) = 1.5 * x(1) + x(2);
    f = f';
end

function f = focusNode(t, x)
    f(1) = 2 * x(2);
    f(2) = -3 * x(1) - x(2);
    f = f';
end

function f = decriticalNode(t, x)
    f(1) = -x(1);
    f(2) = -x(2);
    f = f';
end

function f = centerNode(t, x)
    f(1) = x(1) - 4 * x(2);
    f(2) = 2 * x(1) - x(2);
    f = f';
end


