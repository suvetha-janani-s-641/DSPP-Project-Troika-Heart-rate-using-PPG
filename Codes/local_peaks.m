function [peak_locs, peak_mags] = local_peaks(spectrum, min_peak_dist)
    if nargin < 2 || isempty(min_peak_dist), min_peak_dist = 1; end
    [peak_mags, peak_locs] = findpeaks(spectrum(:), 'MinPeakDistance', min_peak_dist);
end