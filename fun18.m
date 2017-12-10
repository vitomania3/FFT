function dxdt = fun18(t, x)
    
    global G m1 m2;
    
    x11 = x(1);
    x12 = x(2);
    x21 = x(3);
    x22 = x(4);
    dx11_dt = x(5);
    dx12_dt = x(6);
    dx21_dt = x(7);
    dx22_dt = x(8);
    
%     norm = @(x) sqrt(x(1)^2 + x(2)^2);
    n12_3 = norm([x11 - x21; x12 - x22])^3;
    
    d2x11_dt2 = G * m2 * (x21 - x11) / n12_3;
    d2x12_dt2 = G * m2 * (x22 - x12) / n12_3;
    d2x21_dt2 = G * m1 * (x11 - x21) / n12_3;
    d2x22_dt2 = G * m1 * (x12 - x22) / n12_3;
    
    dxdt = [dx11_dt;    dx12_dt;    dx21_dt;    dx22_dt; ...
            d2x11_dt2;  d2x12_dt2;  d2x21_dt2;   d2x22_dt2];
end