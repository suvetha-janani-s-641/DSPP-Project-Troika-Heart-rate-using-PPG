function [x_sparse, f_grid] = focuss_spectrum(diff_ppg, fs, ~)
% Canonical TROIKA Section III-C: 4x-Decimated Toeplitz-FFT Regularized FOCUSS
    sig = double(diff_ppg(:));
    sig_sub = sig(1:4:end); % 31.25 Hz -> M = 1024 matches 4096-pt grid at 125 Hz
    fs_sub  = fs / 4.0;

    sig_sub = (sig_sub - mean(sig_sub)) / (std(sig_sub) + 1e-8);
    N_diff  = length(sig_sub);

    M = 1024;
    f_full = (0:M - 1)' * (fs_sub / M);

    x_sparse = ones(M, 1);
    lambda   = 1e-3;
    p        = 0.8;

    for iter = 1:6
        w2  = abs(x_sparse) .^ (2 - p);
        r   = fft(w2, M) / N_diff;
        col = r(1:N_diff);
        AAh = toeplitz(col, conj(col)) + lambda * eye(N_diff);

        u        = AAh \ sig_sub;
        PhiHu    = conj(fft(conj(u), M)) / sqrt(N_diff);
        x_sparse = w2 .* PhiHu;
    end

    spec_full = abs(x_sparse);
    phys_mask = (f_full >= 0.4) & (f_full <= 5.0);
    f_grid    = f_full(phys_mask);
    x_sparse  = spec_full(phys_mask);
end