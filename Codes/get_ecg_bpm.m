function bpm_ecg = get_ecg_bpm(ecg_signal, fs)
% Direct QRS R-peak detector (8-25 Hz zero-phase filter + dual-polarity consistency)
    persistent b a cached_fs
    if isempty(b) || isempty(cached_fs) || cached_fs ~= fs
        [b, a] = butter(2, [8, 25] / (fs / 2), 'bandpass');
        cached_fs = fs;
    end

    ef = filtfilt(b, a, double(ecg_signal(:)));
    scale = prctile(abs(ef), 92);
    if scale <= 1e-6
        bpm_ecg = NaN;
        return;
    end

    min_dist = round(0.28 * fs); % 280 ms refractory period (~214 BPM ceiling)

    % Evaluate both positive and negative R-wave polarities and pick the steadier RR train
    [bpm_pos, cv_pos, n_pos] = detect_polarity_bpm( ef, fs, min_dist, 0.45 * scale);
    [bpm_neg, cv_neg, n_neg] = detect_polarity_bpm(-ef, fs, min_dist, 0.45 * scale);

    if isnan(bpm_pos) && isnan(bpm_neg)
        bpm_ecg = NaN;
    elseif isnan(bpm_neg)
        bpm_ecg = bpm_pos;
    elseif isnan(bpm_pos)
        bpm_ecg = bpm_neg;
    elseif n_pos >= 5 && n_neg >= 5
        if cv_pos <= cv_neg
            bpm_ecg = bpm_pos;
        else
            bpm_ecg = bpm_neg;
        end
    elseif n_pos >= n_neg
        bpm_ecg = bpm_pos;
    else
        bpm_ecg = bpm_neg;
    end
end

function [bpm, rr_cv, n_valid] = detect_polarity_bpm(sig, fs, min_dist, min_prom)
    [~, locs] = findpeaks(sig, 'MinPeakDistance', min_dist, 'MinPeakProminence', min_prom);
    if length(locs) < 3
        bpm = NaN; rr_cv = Inf; n_valid = 0;
        return;
    end
    rr = diff(locs) / fs;
    valid_rr = rr(rr >= 0.28 & rr <= 1.45); % 41 - 214 BPM
    n_valid = length(valid_rr);
    if n_valid < 2
        bpm = NaN; rr_cv = Inf;
        return;
    end
    med_rr = median(valid_rr);
    rr_cv  = std(valid_rr) / (med_rr + 1e-6);
    bpm    = 60.0 / med_rr;
end