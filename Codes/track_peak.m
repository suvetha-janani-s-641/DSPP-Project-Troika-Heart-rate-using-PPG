function [est_bpm, state] = track_peak(spectrum, f_grid, state, is_init, acc_peaks_bpm)
    if nargin < 5, acc_peaks_bpm = []; end

    bpm_grid = f_grid(:) * 60.0;
    spec     = spectrum(:);
    bin_bpm  = bpm_grid(2) - bpm_grid(1);

    MIN_BPM = 45.0;
    MAX_BPM = 195.0;

    for k = 1:length(acc_peaks_bpm)
        ap = acc_peaks_bpm(k);
        if is_init || ~isfield(state, 'prev_bpm') || abs(ap - state.prev_bpm) > 5.0
            m_bins = abs(bpm_grid - ap) <= 3.5;
            spec(m_bins) = spec(m_bins) * 0.15;
        end
    end

    %% Stage 1: Initialization (Section III-D1)
    if is_init || ~isfield(state, 'prev_bpm') || isempty(state.prev_bpm)
        init_mask = (bpm_grid >= 55.0) & (bpm_grid <= 135.0);
        sub_spec  = spec(init_mask);
        sub_grid  = bpm_grid(init_mask);
        [m_val, idx] = max(sub_spec);
        est_bpm   = sub_grid(idx);

        if est_bpm >= 115.0
            half_mask = (bpm_grid >= est_bpm/2 - 4.0) & (bpm_grid <= est_bpm/2 + 4.0);
            if any(half_mask) && max(spec(half_mask)) >= 0.45 * m_val
                h_grid = bpm_grid(half_mask);
                [~, h_idx] = max(spec(half_mask));
                est_bpm = h_grid(h_idx);
            end
        end

        state.prev_bpm    = est_bpm;
        state.bpm_hist    = est_bpm;
        state.stuck_count = 0;
        return;
    end

    prev_bpm = state.prev_bpm;

    %% Stage 2: Peak Selection in R0 (Fundamental) & R1 (2nd Harmonic) (Section III-D2)
    Delta1 = 12.0;
    Delta2 = 24.0;
    eta    = 0.30;

    r0_mask = (bpm_grid >= max(MIN_BPM, prev_bpm - Delta1)) & (bpm_grid <= min(MAX_BPM, prev_bpm + Delta1));
    r1_mask = (bpm_grid >= max(90.0, 2*prev_bpm - Delta2)) & (bpm_grid <= min(300.0, 2*prev_bpm + Delta2));

    [pks0, locs0] = extract_top_peaks(spec, bpm_grid, r0_mask, eta, 3);
    [~,    locs1] = extract_top_peaks(spec, bpm_grid, r1_mask, eta, 3);

    cand_bpm = NaN;

    % Case 1: Harmonic peak pair (f0 in R0, ~2*f0 in R1)
    if ~isempty(locs0) && ~isempty(locs1)
        best_score = -Inf;
        for i = 1:length(locs0)
            for j = 1:length(locs1)
                if abs(locs1(j) - 2.0 * locs0(i)) <= 2.5 * bin_bpm
                    score = pks0(i) - 0.03 * abs(locs0(i) - prev_bpm);
                    if score > best_score
                        best_score = score;
                        cand_bpm = locs0(i);
                    end
                end
            end
        end
    end

    % Case 2: Nearest significant peak in R0
    if isnan(cand_bpm) && ~isempty(locs0)
        tight_mask = abs(locs0 - prev_bpm) <= 8.0;
        if any(tight_mask)
            t_locs = locs0(tight_mask);
            t_pks  = pks0(tight_mask);
            [~, idx] = max(t_pks);
            cand_bpm = t_locs(idx);
        else
            [~, idx] = min(abs(locs0 - prev_bpm));
            cand_bpm = locs0(idx);
        end
    end

    % Case 3: Severe motion dropout -> smooth local trend prediction
    if isnan(cand_bpm)
        n_hist = length(state.bpm_hist);
        if n_hist >= 4
            trend = mean(diff(state.bpm_hist(n_hist-3:n_hist)));
            trend = max(-3.0, min(3.0, trend));
        else
            trend = 0.0;
        end
        cand_bpm = prev_bpm + trend;
    end

    %% Stage 3: Verification & Stuck-Lock Escape (Section III-D3)
    max_jump = 8.0;
    step = cand_bpm - prev_bpm;
    if abs(step) > max_jump
        est_bpm = prev_bpm + sign(step) * max_jump;
    else
        est_bpm = cand_bpm;
    end

    if abs(est_bpm - prev_bpm) < 0.25 * bin_bpm
        state.stuck_count = state.stuck_count + 1;
    else
        state.stuck_count = 0;
    end

    if state.stuck_count >= 3 && length(state.bpm_hist) >= 5
        h_len = length(state.bpm_hist);
        n_trend = (state.bpm_hist(h_len) - state.bpm_hist(h_len - 4)) / 4.0;
        n_trend = max(-3.0, min(3.0, n_trend));
        est_bpm = prev_bpm + 2.0 * n_trend;
        state.stuck_count = 0;
    end

    est_bpm = min(MAX_BPM, max(MIN_BPM, est_bpm));
    state.prev_bpm = est_bpm;
    state.bpm_hist = [state.bpm_hist; est_bpm];
end

function [top_pks, top_bpms] = extract_top_peaks(spec, bpm_grid, mask, eta, max_k)
    sub_s = spec(mask);
    sub_b = bpm_grid(mask);
    if numel(sub_s) < 3
        top_pks = []; top_bpms = [];
        return;
    end
    [pks, locs] = findpeaks(sub_s);
    if isempty(pks)
        top_pks = []; top_bpms = [];
        return;
    end
    keep = pks >= eta * max(pks);
    pks  = pks(keep);
    locs = locs(keep);
    [pks, order] = sort(pks, 'descend');
    locs = locs(order);
    k = min(max_k, length(pks));
    top_pks  = pks(1:k);
    top_bpms = sub_b(locs(1:k));
end