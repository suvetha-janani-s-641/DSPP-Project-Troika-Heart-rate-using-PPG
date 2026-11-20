function [filt_sig, freqs, fft_mag] = bandpass_fft(sig, fs)
% Canonical TROIKA Section III: 4th-order zero-phase Butterworth bandpass [0.4, 5.0] Hz
    persistent b a cached_fs
    if isempty(b) || isempty(cached_fs) || cached_fs ~= fs
        [b, a] = butter(4, [0.4, 5.0] / (fs / 2), 'bandpass');
        cached_fs = fs;
    end

    N = length(sig);
    filt_sig = filtfilt(b, a, double(sig(:)));

    N_fft = 4096;
    half_fft = floor(N_fft / 2);
    fft_res = fft(filt_sig, N_fft);
    fft_mag = abs(fft_res(1:half_fft));
    freqs   = (0:half_fft - 1)' * (fs / N_fft);
end