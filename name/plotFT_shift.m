function res_struct = plotFT_shift(shift, hFigure, fHandle, fFTHandle, step, inpLimVec, outLimVec)
res_struct = struct('nPoints', 0, 'step', 0, 'inpLimVec', inpLimVec, 'outLimVec', outLimVec);
nPoints = floor((inpLimVec(2) - inpLimVec(1)) ./ step + 1);
new_step = (inpLimVec(2) - inpLimVec(1)) / (nPoints - 1);
res_struct.nPoints = nPoints;
%disp(nPoints);
res_struct.step = new_step;

t_splitting = inpLimVec(1) : new_step : inpLimVec(2);
values = fHandle(t_splitting);

%values(isinf(values)) = 0;
%values(isnan(values)) = 0;

inverted_fourier_transform = fft(values);
fourier_transform = fftshift(inverted_fourier_transform);

fourier_values = new_step .* fourier_transform;

l_step = 2 * pi / new_step;

l_splitting_helper = linspace(0, l_step, nPoints);

middle_point = l_splitting_helper(floor(nPoints / 2) + 1);

l_splitting = l_splitting_helper - middle_point;

fourier_values = fourier_values .* exp(- 1i .* l_splitting .* inpLimVec(1));

cond_const = 0.0001;

SPlotInfo = get(hFigure, 'UserData');
if (isempty(SPlotInfo)) 
    if (isempty(outLimVec))
        left_point_splitting = 1;
        right_point_splitting = numel(l_splitting);
        for k = 2 : nPoints
            if (abs(fourier_values) > cond_const)
                if left_point_splitting == 1
                    left_point_splitting = k;
                end
                right_point_splitting = k;
            end
        end
        outLimVec = [l_splitting(left_point_splitting), l_splitting(right_point_splitting)];
    end
    clf(hFigure);
    real_fourier_plot = subplot(2, 1, 1);
    imag_fourier_plot = subplot(2, 1, 2);
    
    set(real_fourier_plot, 'XLim', [outLimVec(1) - shift, outLimVec(2) + shift]);
    set(imag_fourier_plot, 'XLim', [outLimVec(1) - shift, outLimVec(2) + shift]);
    
    real_fourier_plot.Title.String = 'Real Fourier';
    imag_fourier_plot.Title.String = 'Image Fourier';
    
    real_fourier_plot.XLabel.String = '\lambda';
    real_fourier_plot.YLabel.String = 'Fourier';
    
    imag_fourier_plot.XLabel.String = '\lambda';
    imag_fourier_plot.YLabel.String = 'Fourier';
    
    SPlotInfo = struct('real', real_fourier_plot, 'imag', imag_fourier_plot);
end 
if (isempty(outLimVec))
    real_fourier_plot = SPlotInfo.real;
    outLimVec = get(real_fourier_plot, 'xLim');
else
    set(SPlotInfo.real, 'XLim', [outLimVec(1) - shift, outLimVec(2) + shift]);
    set(SPlotInfo.imag, 'XLim', [outLimVec(1) - shift, outLimVec(2) + shift]);
end
set(hFigure, 'UserData', SPlotInfo);
res_struct.outLimVec = outLimVec;

final_l_splitting = l_splitting - outLimVec(1) + shift;
final_l_splitting = final_l_splitting(2 : end) .* final_l_splitting(1 : (end - 1));
minimum_number = find(final_l_splitting <= 0) + 1;
if (isempty(minimum_number))
    minimum_number = 1;
end

final_l_splitting = l_splitting - outLimVec(2) - shift;
final_l_splitting = final_l_splitting(2 : end) .* final_l_splitting(1 : (end - 1));
maximum_number = find(final_l_splitting <= 0) - 1;
if (isempty(maximum_number))
    maximum_number = numel(l_splitting);
end

final_l_splitting = l_splitting(minimum_number : maximum_number);

fourier_values = fourier_values(minimum_number : maximum_number);

hFigure.CurrentAxes = SPlotInfo.real;
hFigure.CurrentAxes.NextPlot = 'replacechildren';

plot(final_l_splitting, real(fourier_values), 'b');;

legend('real fourier');
if (~isempty(fFTHandle))
    hFigure.CurrentAxes.NextPlot = 'add';
    plot(final_l_splitting, real(fFTHandle(final_l_splitting)), 'g');
    
    plot(final_l_splitting + shift, real(fFTHandle(final_l_splitting)), 'r');
    plot(final_l_splitting - shift, real(fFTHandle(final_l_splitting)), 'r');
    xlim([outLimVec(1), outLimVec(2)]);

    lgnd = legend('real fourier', 'real analytic fourier', 'shifted real analytic fourier');
    set(lgnd, 'color', 'none', 'box', 'off');
end
hFigure.CurrentAxes = SPlotInfo.imag;
axis tight;

hFigure.CurrentAxes.NextPlot = 'replacechildren';
plot(final_l_splitting, imag(fourier_values), 'b');
legend('image fourier');

if (~isempty(fFTHandle))
    hFigure.CurrentAxes.NextPlot = 'add';
    plot(final_l_splitting, imag(fFTHandle(final_l_splitting)), 'g');
    
    plot([final_l_splitting, final_l_splitting + shift], imag(fFTHandle([final_l_splitting - shift, final_l_splitting])), 'r');
    plot([final_l_splitting - shift, final_l_splitting], imag(fFTHandle([final_l_splitting, final_l_splitting + shift])), 'r');
    xlim([outLimVec(1), outLimVec(2)]);
    
    lgnd = legend('image fourier', 'image analytic fourier', 'shifted image analytic fourier');
    set(lgnd, 'color', 'none', 'box', 'off');
end
end