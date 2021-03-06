function [ res ] = plotFT( hFigure, fHandle, fFTHandle, step, inpLimVec, outLimVec)
    res = struct('nPoints', [], 'Step', []);
    res.inpLimVec = inpLimVec;
    
    eps = .0001;
    a = inpLimVec(1);
    b = inpLimVec(2);
    n = floor((b - a) ./ step) + 1;
    step = (b - a) ./ (n - 1);
    res.nPoints = n;
    res.Step = step;
    
    lsp = linspace(inpLimVec(1), inpLimVec(2), n);

    func = fHandle(lsp);
    
    fourier = step .* fftshift(fft(func));
    lsp = linspace(0, 2 * pi ./ step, n);
    
    lsp = lsp - lsp(floor(n ./ 2 + 1));  %symmetrical partition
    fourier = fourier .* exp(-1i .* lsp .* a) ; %shifting the fourier transform
    
    SPlotInfo = get(hFigure, 'UserData');
    
    if isempty(SPlotInfo)
        if isempty(outLimVec)
            limits = [0 0];
            for i = 1:n
                if abs(fourier(i)) > eps
                    if limits(1) == 0
                        limits(1) = i;
                    end
                    limits(2) = i;
                end
            end
            outLimVec = [lsp(limits(1)), lsp(limits(2))];
            res.outLimVec = outLimVec;
        end
        
        clf(hFigure); %clear figure window
        
        axRe = subplot(2, 1, 1);
        set(axRe, 'XLim', outLimVec);
        axRe.Title.String = 'Real part of fft';
        axRe.XLabel.String = '\Lambda';
        axRe.YLabel.String = 'Re(fft)';
        
        axIm = subplot(2, 1, 2);
        set(axIm, 'XLim', outLimVec);
        axIm.Title.String = 'Imaginary part of fft';
        axIm.XLabel.String = '\Lambda';
        axIm.YLabel.String = 'Im(fft)';
       
        SPlotInfo = struct('axRe', axRe, 'axIm', axIm);
    end
    
    if isempty(outLimVec)
        outLimVec = get(SPlotInfo.axRe, 'xLim');
    else
        set(SPlotInfo.axRe, 'XLim', outLimVec);
        set(SPlotInfo.axIm, 'XLim', outLimVec);
    end
    set(hFigure, 'UserData', SPlotInfo);

%     t = lsp - outLimVec(1);
%     t = t(2:end).*t(1:end-1);
%     minInd = find(t<=0);
%     t = lsp - outLimVec(2);
%     t = t(2:end).*t(1:end-1);
%     maxInd = find(t<=0);
    
%     lsp = lsp(minInd:maxInd);
%     fourier = fourier(minInd:maxInd);
    
    % drawing graphs
    max(abs(real(fourier)));
    hFigure.CurrentAxes = SPlotInfo.axRe;
    hFigure.CurrentAxes.NextPlot = 'replacechildren';
    plot(lsp, real(fourier), 'Color', 'b'); 
    legend('FFT');
     
    if ~isempty(fFTHandle) 
        hFigure.CurrentAxes.NextPlot = 'add';
        plot(lsp, real(fFTHandle(lsp)), 'r');
        legend('FFT', 'AFT(analytical)', 'Orientation', 'horizontal');
%         ylim([min(real(fourier)) max(real(fourier)) + step]);
    end

    hFigure.CurrentAxes = SPlotInfo.axIm;
    hFigure.CurrentAxes.NextPlot = 'replacechildren';
    plot(lsp, imag(fourier), 'Color', 'b');
    legend('FFT');
    
%     fFTHandle(lsp)
%     lsp
    if ~isempty(fFTHandle) 
        hFigure.CurrentAxes.NextPlot = 'add';
        plot(lsp, imag(fFTHandle(lsp)), 'r');
        legend('FFT', 'AFT(analytical)', 'Orientation', 'horizontal');
    end
end