%% Master Driver Script — Canonical Dual-PPG TROIKA with Report-Ready Analytics
clear; clc; close all;

fs = 125;
step_samples = 250; % 2-second window shift at 125 Hz

% Explicitly load IEEEPPG_TRAIN.ts if present (otherwise first .ts file, or .mat)
if isfile('IEEEPPG_TRAIN.ts')
    ts_name = 'IEEEPPG_TRAIN.ts';
    fprintf('Loading %s...\n', ts_name);
    [ieeeppg_data, bpm_true] = load_ieeeppg_ts(ts_name);
elseif ~isempty(dir('*.ts'))
    ts_files = dir('*.ts');
    ts_name = ts_files(1).name;
    fprintf('Loading %s...\n', ts_name);
    [ieeeppg_data, bpm_true] = load_ieeeppg_ts(ts_name);
else
    ts_name = 'ieeeppg_train_5channel.mat';
    data = load(ts_name);
    ieeeppg_data = data.ieeeppg_data;
    bpm_true = [];
end

[num_windows, win_len, ~] = size(ieeeppg_data);
fprintf('Loaded %d windows (%d samples/window, 5 channels).\n', num_windows, win_len);

% Automatic subject-session boundary detection (6-second overlap correlation < 0.8)
is_new_subject = false(num_windows, 1);
is_new_subject(1) = true;
for w = 2:num_windows
    seg_prev = squeeze(ieeeppg_data(w-1, step_samples+1:end, 1))';
    seg_curr = squeeze(ieeeppg_data(w, 1:end-step_samples, 1))';
    if std(seg_prev) < 1e-8 || std(seg_curr) < 1e-8 || corr(seg_prev, seg_curr) < 0.8
        is_new_subject(w) = true;
    end
end
fprintf('Detected %d subject sessions.\n', sum(is_new_subject));

bpm_history   = zeros(num_windows, 1);
win_times_ms  = zeros(num_windows, 1);
tracker_state = struct();

tic;
for w = 1:num_windows
    t_win = tic;
    if mod(w, 250) == 0 || w == 1
        fprintf('Processing window %d/%d (%.1f s)...\n', w, num_windows, toc);
    end

    ppg1 = double(ieeeppg_data(w, :, 1))';
    ppg2 = double(ieeeppg_data(w, :, 2))';
    acc  = double(squeeze(ieeeppg_data(w, :, 3:5)));

    is_init = is_new_subject(w);
    if is_init || ~isfield(tracker_state, 'prev_bpm')
        prev_hr_hint = -1;
    else
        prev_hr_hint = tracker_state.prev_bpm;
    end

    % Step 0: Zero-phase bandpass filter [0.4, 5.0] Hz
    [filt_ppg1, ~, ~] = bandpass_fft(ppg1, fs);
    [filt_ppg2, ~, ~] = bandpass_fft(ppg2, fs);

    % Step 1 (TROIKA III-A): Trajectory-protected SSA MA removal (L = 400)
    [clean_ppg1, acc_peaks_bpm] = ssa_denoise(filt_ppg1, acc, 400, fs, prev_hr_hint);
    [clean_ppg2, ~]             = ssa_denoise(filt_ppg2, acc, 400, fs, prev_hr_hint);

    % Step 2 (TROIKA III-B): Second-order temporal difference
    diff2_ppg1 = diff(clean_ppg1, 2);
    diff2_ppg2 = diff(clean_ppg2, 2);

    % Step 3 (TROIKA III-C): Regularized FOCUSS sparse spectrum & dual-PPG fusion
    [spec1, f_grid] = focuss_spectrum(diff2_ppg1, fs);
    [spec2, ~]      = focuss_spectrum(diff2_ppg2, fs);
    spec1 = spec1 / (max(spec1) + 1e-12);
    spec2 = spec2 / (max(spec2) + 1e-12);
    x_sparse = 0.5 * (spec1 + spec2);

    % Step 4 (TROIKA III-D): 3-Stage Spectral Peak Tracking & Verification
    [est_bpm, tracker_state] = track_peak(x_sparse, f_grid, tracker_state, is_init, acc_peaks_bpm);
    bpm_history(w)  = est_bpm;
    win_times_ms(w) = toc(t_win) * 1000;
end
total_time_s = toc;
fprintf('Completed all %d windows in %.1f s.\n', num_windows, total_time_s);

%% ========================================================================
%  REPORT-READY PERFORMANCE EVALUATION & VISUALIZATION
%  ========================================================================
subj_starts  = find(is_new_subject);
subj_ends    = [subj_starts(2:end) - 1; num_windows];
num_subjects = length(subj_starts);

% Computational Throughput & Real-Time Feasibility
step_sec     = step_samples / fs; % 2.0 s window shift
mean_lat_ms  = mean(win_times_ms);
p95_lat_ms   = interp1(linspace(0, 100, num_windows), sort(win_times_ms), 95);
rtf          = (mean_lat_ms / 1000) / step_sec; % < 1.0 indicates faster-than-real-time

if ~isempty(bpm_true) && any(~isnan(bpm_true))
    valid = ~isnan(bpm_true) & ~isnan(bpm_history);
    gt    = bpm_true(valid);
    est   = bpm_history(valid);
    N_val = numel(gt);

    signed_err = est - gt;
    abs_err    = abs(signed_err);
    pct_err    = (abs_err ./ gt) * 100;

    % 1. Error Magnitude & Distribution Metrics
    MAE     = mean(abs_err);
    SD_AE   = std(abs_err);
    RMSE    = sqrt(mean(signed_err .^ 2));
    MAPE    = mean(pct_err);
    MedAE   = median(abs_err);
    sorted_ae = sort(abs_err);
    q25_ae  = interp1(linspace(0, 100, N_val), sorted_ae, 25);
    q75_ae  = interp1(linspace(0, 100, N_val), sorted_ae, 75);
    IQR_AE  = q75_ae - q25_ae;
    Max_AE  = max(abs_err);

    % 2. Bland-Altman Agreement Metrics
    bias      = mean(signed_err);
    sd_diff   = std(signed_err);
    loa_upper = bias + 1.96 * sd_diff;
    loa_lower = bias - 1.96 * sd_diff;
    in_loa_pct = mean((signed_err >= loa_lower) & (signed_err <= loa_upper)) * 100;

    % 3. Correlation, Regression & Lin's Concordance (CCC)
    [R_mat, P_mat] = corrcoef(gt, est);
    r_val  = R_mat(1, 2);
    p_val  = P_mat(1, 2);
    R2_val = r_val ^ 2;

    var_gt  = var(gt, 1);
    var_est = var(est, 1);
    cov_ge  = mean((gt - mean(gt)) .* (est - mean(est)));
    CCC     = (2 * cov_ge) / (var_gt + var_est + (mean(gt) - mean(est))^2);

    reg_p   = polyfit(gt, est, 1); % est = reg_p(1)*gt + reg_p(2)

    % 4. Clinical Accuracy Thresholds (ANSI/AAMI EC13 & IEEE SPC Standards)
    acc_5bpm   = mean(abs_err <= 5.0) * 100;
    acc_10bpm  = mean(abs_err <= 10.0) * 100;
    acc_10pct  = mean(pct_err <= 10.0) * 100;
    outlier_15 = mean(abs_err > 15.0) * 100;

    % 5. Per-Subject Breakdown Table
    Subj_ID   = (1:num_subjects)';
    Windows   = subj_ends - subj_starts + 1;
    Mean_GT   = zeros(num_subjects, 1);
    Mean_Est  = zeros(num_subjects, 1);
    Subj_MAE  = zeros(num_subjects, 1);
    Subj_RMSE = zeros(num_subjects, 1);
    Subj_MAPE = zeros(num_subjects, 1);
    Subj_Bias = zeros(num_subjects, 1);
    Subj_r    = zeros(num_subjects, 1);
    Acc_5BPM  = zeros(num_subjects, 1);
    Acc_10BPM = zeros(num_subjects, 1);

    for s = 1:num_subjects
        idx   = subj_starts(s):subj_ends(s);
        v_sub = ~isnan(bpm_true(idx)) & ~isnan(bpm_history(idx));
        g_s   = bpm_true(idx(v_sub));
        e_s   = bpm_history(idx(v_sub));
        d_s   = e_s - g_s;

        Mean_GT(s)   = round(mean(g_s), 2);
        Mean_Est(s)  = round(mean(e_s), 2);
        Subj_MAE(s)  = round(mean(abs(d_s)), 2);
        Subj_RMSE(s) = round(sqrt(mean(d_s .^ 2)), 2);
        Subj_MAPE(s) = round(mean(abs(d_s) ./ g_s) * 100, 2);
        Subj_Bias(s) = round(mean(d_s), 2);
        if numel(g_s) > 1 && std(g_s) > 0 && std(e_s) > 0
            rm = corrcoef(g_s, e_s);
            Subj_r(s) = round(rm(1, 2), 4);
        else
            Subj_r(s) = NaN;
        end
        Acc_5BPM(s)  = round(mean(abs(d_s) <= 5.0) * 100, 1);
        Acc_10BPM(s) = round(mean(abs(d_s) <= 10.0) * 100, 1);
    end

    SummaryTable = table(Subj_ID, Windows, Mean_GT, Mean_Est, Subj_MAE, Subj_RMSE, ...
                         Subj_MAPE, Subj_Bias, Subj_r, Acc_5BPM, Acc_10BPM);

    % Print Full Report Summary to Command Window
    fprintf('\n====================================================================\n');
    fprintf('  REPORT-READY PERFORMANCE SUMMARY (%s | N = %d windows)\n', ts_name, N_val);
    fprintf('====================================================================\n');
    fprintf('1. Accuracy & Error Magnitude:\n');
    fprintf('   - Mean Absolute Error (MAE +/- SD):    %.2f +/- %.2f BPM\n', MAE, SD_AE);
    fprintf('   - Root Mean Square Error (RMSE):       %.2f BPM\n', RMSE);
    fprintf('   - Mean Absolute Percentage Error:      %.2f%%\n', MAPE);
    fprintf('   - Median Absolute Error (MedAE / IQR): %.2f BPM (IQR: %.2f BPM)\n', MedAE, IQR_AE);
    fprintf('   - Maximum Absolute Error:              %.2f BPM\n', Max_AE);
    fprintf('\n2. Agreement & Correlation:\n');
    fprintf('   - Pearson Correlation (r):             %.4f (p = %.2e, R^2 = %.4f)\n', r_val, p_val, R2_val);
    fprintf('   - Lin''s Concordance Correlation (CCC): %.4f\n', CCC);
    fprintf('   - Linear Fit (Est vs. True):           Est = %.3f * True + %.2f BPM\n', reg_p(1), reg_p(2));
    fprintf('   - Bland-Altman Bias +/- SD:            %.2f +/- %.2f BPM\n', bias, sd_diff);
    fprintf('   - Bland-Altman 95%% LOA:                [%.2f, %.2f] BPM (%.1f%% inside)\n', loa_lower, loa_upper, in_loa_pct);
    fprintf('\n3. Clinical Thresholds (AAMI / IEEE Benchmarks):\n');
    fprintf('   - Within +/- 5 BPM:                    %.2f%% (%d/%d windows)\n', acc_5bpm, sum(abs_err <= 5), N_val);
    fprintf('   - Within +/- 10 BPM:                   %.2f%% (%d/%d windows)\n', acc_10bpm, sum(abs_err <= 10), N_val);
    fprintf('   - Within +/- 10%% of Ground Truth:      %.2f%%\n', acc_10pct);
    fprintf('   - Severe Track-Loss (> 15 BPM error):  %.2f%% (%d/%d windows)\n', outlier_15, sum(abs_err > 15), N_val);
    fprintf('\n4. Computational Latency & Real-Time Feasibility:\n');
    fprintf('   - Mean Window Latency:                 %.2f ms (95th pct: %.2f ms)\n', mean_lat_ms, p95_lat_ms);
    fprintf('   - Real-Time Factor (vs. %.1f s step):   %.4f (< 1.0 = Real-Time)\n', step_sec, rtf);
    fprintf('====================================================================\n\n');
    disp(SummaryTable);

    % Export Per-Subject Table & Window-Level Predictions for Report Tables/Plots
    writetable(SummaryTable, 'troika_subject_performance_summary.csv');
    results_report = struct('dataset', ts_name, 'MAE', MAE, 'SD_AE', SD_AE, 'RMSE', RMSE, ...
        'MAPE', MAPE, 'MedAE', MedAE, 'IQR_AE', IQR_AE, 'Max_AE', Max_AE, ...
        'Pearson_r', r_val, 'R2', R2_val, 'CCC', CCC, 'Bias', bias, ...
        'LOA_lower', loa_lower, 'LOA_upper', loa_upper, ...
        'Acc_5BPM', acc_5bpm, 'Acc_10BPM', acc_10bpm, 'Acc_10pct', acc_10pct, ...
        'Outlier_15BPM', outlier_15, 'Mean_Latency_ms', mean_lat_ms, 'RTF', rtf, ...
        'SubjectTable', SummaryTable, 'bpm_true', bpm_true, 'bpm_est', bpm_history);
    save('troika_report_metrics.mat', 'results_report');
    fprintf('Saved CSV table -> troika_subject_performance_summary.csv\n');
    fprintf('Saved MAT struct -> troika_report_metrics.mat\n');

    %% FIGURE 1: HR Trajectory + Instantaneous Absolute Error Across Sessions
    figure('Color', 'w', 'Position', [80, 80, 1100, 560]);
    subplot(3, 1, [1 2]);
    plot(bpm_true, 'k-', 'LineWidth', 1.4, 'DisplayName', 'Ground-Truth (ECG BPM0)'); hold on;
    plot(bpm_history, 'r--', 'LineWidth', 1.15, 'DisplayName', 'Canonical Dual-PPG TROIKA');
    for s = 2:num_subjects
        line([subj_starts(s) subj_starts(s)], [50 190], 'Color', [0.7 0.7 0.7], ...
             'LineStyle', ':', 'HandleVisibility', 'off');
    end
    grid on; ylabel('Heart Rate (BPM)'); ylim([50 190]);
    title(sprintf('Dual-PPG TROIKA Tracking (MAE = %.2f BPM, RMSE = %.2f BPM, r = %.4f, CCC = %.4f)', ...
          MAE, RMSE, r_val, CCC));
    legend('Location', 'best');

    subplot(3, 1, 3);
    win_abs_err = abs(bpm_history - bpm_true);
    area(win_abs_err, 'FaceColor', [0.85 0.33 0.10], 'FaceAlpha', 0.35, 'EdgeColor', [0.85 0.33 0.10]); hold on;
    line([1 num_windows], [5 5], 'Color', [0.1 0.6 0.2], 'LineStyle', '--', 'LineWidth', 1.1);
    line([1 num_windows], [MAE MAE], 'Color', 'k', 'LineStyle', '-.', 'LineWidth', 1.1);
    for s = 2:num_subjects
        line([subj_starts(s) subj_starts(s)], [0 max(win_abs_err)+5], 'Color', [0.7 0.7 0.7], 'LineStyle', ':');
    end
    grid on; xlabel('Window Index (2 s step)'); ylabel('|Error| (BPM)');
    legend('|HR_{est} - HR_{true}|', '5 BPM Clinical Limit', sprintf('Mean MAE (%.2f BPM)', MAE), 'Location', 'northeast');

    %% FIGURE 2: Correlation Scatter Plot & Bland-Altman Agreement Plot
    figure('Color', 'w', 'Position', [120, 120, 1080, 440]);

    % Left Panel: Correlation & Regression
    subplot(1, 2, 1);
    scatter(gt, est, 14, [0.15 0.45 0.75], 'filled', 'MarkerFaceAlpha', 0.45); hold on;
    hr_lims = [50 190];
    plot(hr_lims, hr_lims, 'k-', 'LineWidth', 1.4, 'DisplayName', 'Identity (y = x)');
    plot(hr_lims, hr_lims + 5, 'g--', 'LineWidth', 1.0, 'DisplayName', '\pm5 BPM Band');
    plot(hr_lims, hr_lims - 5, 'g--', 'LineWidth', 1.0, 'HandleVisibility', 'off');
    plot(hr_lims, polyval(reg_p, hr_lims), 'r-.', 'LineWidth', 1.3, ...
         'DisplayName', sprintf('Fit: y = %.2fx + %.1f', reg_p(1), reg_p(2)));
    grid on; axis equal; xlim(hr_lims); ylim(hr_lims);
    xlabel('Ground-Truth ECG HR (BPM)'); ylabel('Estimated PPG HR (BPM)');
    title(sprintf('Correlation Plot (r = %.4f, R^2 = %.4f, CCC = %.4f)', r_val, R2_val, CCC));
    legend('Location', 'northwest');

    % Right Panel: Bland-Altman Plot
    subplot(1, 2, 2);
    ba_mean = 0.5 * (gt + est);
    scatter(ba_mean, signed_err, 14, [0.55 0.25 0.65], 'filled', 'MarkerFaceAlpha', 0.45); hold on;
    line([50 190], [bias bias], 'Color', 'k', 'LineStyle', '-', 'LineWidth', 1.4);
    line([50 190], [loa_upper loa_upper], 'Color', 'r', 'LineStyle', '--', 'LineWidth', 1.2);
    line([50 190], [loa_lower loa_lower], 'Color', 'r', 'LineStyle', '--', 'LineWidth', 1.2);
    line([50 190], [0 0], 'Color', [0.5 0.5 0.5], 'LineStyle', ':');
    grid on; xlim([50 190]);
    xlabel('Mean of ECG & PPG HR (BPM)'); ylabel('Difference: PPG - ECG (BPM)');
    title(sprintf('Bland-Altman Agreement (Bias = %.2f, 95%% LOA = [%.1f, %.1f])', bias, loa_lower, loa_upper));
    legend('Windows', sprintf('Bias (%.2f BPM)', bias), sprintf('+1.96 SD (%.2f)', loa_upper), ...
           sprintf('-1.96 SD (%.2f)', loa_lower), 'Zero Error', 'Location', 'best');
else
    figure('Color', 'w', 'Position', [100, 100, 1000, 420]);
    plot(bpm_history, 'r-', 'LineWidth', 1.3, 'DisplayName', 'Canonical Dual-PPG TROIKA');
    title('Canonical Dual-PPG TROIKA HR Trajectory');
    grid on; xlabel('Window Index (2 s step)'); ylabel('Heart Rate (BPM)');
    legend('Location', 'best');
end