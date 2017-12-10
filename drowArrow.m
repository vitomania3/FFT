function drowArrow(X, matrix, n)
    [szX, szY] = size(X);
    x = X(1:szX, 1)';
    y = X(1:szX, 2)';
    l = 1 : szX - 1;
    dx = x(l + 1) - x(l);
    dx(end + 1) = dx(end);
    dy = y(l + 1) - y(l);
    dy(end + 1) = dy(end);
    
    [V, D, W] = eig(matrix);
    
%     cmap = hsv(szX);
    quiver(x, y, dx, dy, 0);

    hold on;
    
    if isreal(D)
        V = real(V);
        W = real(W);
        D = real(D);
    
        t = linspace(-n, n, n);
        if -W(1, 1) / W(2, 1) < 0
%             l = 1 : szX - 1;
%             y = -W(1, 1) / W(2, 1) * x;
%             dy = y(l + 1) - y(l);
%             dy(end + 1) = dy(end);
%             dx = x(l + 1) - x(l);
%             dx(end + 1) = dx(end);
%             q = quiver(x, y, dx, dy, 0);
%             q.Color = 'red';
            
            plot(t, -W(1, 1) / W(2, 1) * t, '-r', 'LineWidth', 2)
            plot(t, -W(1, 2) / W(2, 2) * t, '-r', 'LineWidth', 2)
        else
            plot(t, -V(1, 1) / V(2, 1) * t, '-r', 'LineWidth', 2)
            plot(t, -V(1, 2) / V(2, 2) * t, '-r', 'LineWidth', 2)
        end
    end
    axis([-n n -n n])
    end

