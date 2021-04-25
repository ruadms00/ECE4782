function [F] = extract(heartRate, RR)

%%HR feature: mesor, acrophase, amplitude
time = 1:length(heartRate);
%time  = 1:86400;
[mesor, acrophase, amplitude] = cosinor(time,heartRate,2*pi/length(heartRate),0.05);
%%
%%HRV feature: HRRmean, SDNN(standard deviation of IBI)
HRRmean = mean(60./RR);
SDNN = mean(RR);

F = [mesor, acrophase, amplitude, HRRmean, SDNN];

end
