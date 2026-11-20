function [clean_ppg, acc_peaks_bpm] = ssa_denoise(filt_ppg, acc_matrix, L, fs, prev_bpm)
    persistent b a cached_fs cached_L cached_N diag_idx diag_counts
    if nargin < 3 || isempty(L),        L = 400;       end
    if nargin < 4 || isempty(fs),       fs = 125;      end
    if nargin < 5 || isempty(prev_bpm), prev_bpm = -1; end

    if isempty(b) || isempty(cached_fs) || cached_fs ~= fs
        [b, a] = butter(4, [0.4, 5.0] / (fs / 2), 'bandpass');
        cached_fs = fs;
    end

    filt_ppg = double(filt_ppg(:));
    N = length(filt_ppg);
    K = N - L + 1;

    if isempty(diag_idx) || isempty(cached_L) || cached_L ~= L || cached_N ~= N
        [I, J] = ndgrid(1:L, 1:K);
        diag_idx = I(:) + J(:) - 1;
        diag_counts = accumarray(diag_idx, 1, [N, 1]);
        cached_L = L;
        cached_N = N;
    end

    X_hankel = zeros(L, K);
    for i = 1:L
        X_hankel(i, :) = filt_ppg(i : i + K - 1)';
    end
    [U, S, V] = svd(X_hankel, 'econ');
    s_vals = diag(S);

    N_fft = 4096;
    half_fft = floor(N_fft / 2);
    freq_hz  = (0:half_fft - 1)' * (fs / N_fft);
    freq_bpm = freq_hz * 60.0;
    band_idx = find(freq_hz >= 0.4 & freq_hz <= 5.0);

    num_axes = size(acc_matrix, 2);
    acc_specs = zeros(length(band_idx), num_axes);
    for ax = 1:num_axes
        filt_acc = filtfilt(b, a, double(acc_matrix(:, ax)));
        spec_acc = abs(fft(filt_acc, N_fft));
        acc_specs(:, ax) = spec_acc(band_idx);
    end

    global_acc_max = max(acc_specs(:));
    Delta = 4;
    f_acc_mask = false(half_fft, 1);
    acc_peaks_bpm = [];

    if global_acc_max > 1e-9
        for ax = 1:num_axes
            [pks, locs] = findpeaks(acc_specs(:, ax));
            dom_locs = band_idx(locs(pks >= 0.50 * global_acc_max));
            if ~isempty(dom_locs)
                acc_peaks_bpm = [acc_peaks_bpm; freq_bpm(dom_locs)]; %#ok<AGROW>
                for d = 1:length(dom_locs)
                    lo = max(1, dom_locs(d) - Delta);
                    hi = min(half_fft, dom_locs(d) + Delta);
                    f_acc_mask(lo:hi) = true;
                end
            end
        end
    end

    clean_X = zeros(L, K);
    kept_count = 0;
    max_rank = find(s_vals >= 0.015 * s_vals(1), 1, 'last');
    if isempty(max_rank), max_rank = min(40, length(s_vals)); end

    for i = 1:max_rank
        comp_mat = s_vals(i) * (U(:, i) * V(:, i)');
        comp_sig = accumarray(diag_idx, comp_mat(:), [N, 1]) ./ diag_counts;

        comp_spec = abs(fft(comp_sig, N_fft));
        comp_spec = comp_spec(1:half_fft);
        [~, rel_bin] = max(comp_spec(band_idx));
        dom_bin  = band_idx(rel_bin);
        comp_bpm = freq_bpm(dom_bin);

        is_near_hr = (prev_bpm > 0) && (abs(comp_bpm - prev_bpm) <= 4.5);

        if ~f_acc_mask(dom_bin) || is_near_hr
            clean_X = clean_X + comp_mat;
            kept_count = kept_count + 1;
        end
    end

    clean_ppg = accumarray(diag_idx, clean_X(:), [N, 1]) ./ diag_counts;
    if kept_count == 0 || std(clean_ppg) < 1e-6
        clean_ppg = filt_ppg;
    end
end