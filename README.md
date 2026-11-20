# Dual-PPG TROIKA: Heart Rate Estimation During Intensive Physical Exercise

MATLAB implementation of the canonical **TROIKA** framework (*Zhang et al., IEEE TBME, 2015*) adapted for **dual-channel wrist-worn PPG** and **tri-axial accelerometry** on the IEEE Signal Processing Cup 2015 benchmark (`IEEEPPG_TRAIN.ts`).

## Signal Processing Pipeline

Each $8\text{ s}$ ($999$-sample) window sampled at $125\text{ Hz}$ (shifted every $2\text{ s}$ / $250$ samples) is processed through four stages:

1. **Zero-Phase Bandpass Filtering (`bandpass_fft.m`):** 4th-order Butterworth bandpass filter ($0.4\text{–}5.0\text{ Hz}$, $24\text{–}300\text{ BPM}$) via `filtfilt` with a $4096$-point FFT.
2. **Trajectory-Protected SSA Denoising (`ssa_denoise.m`):** Hankel trajectory matrix decomposition ($L = 400$) per PPG channel that removes singular components overlapping dominant tri-axial accelerometer frequencies while protecting components within $\pm 4.5\text{ BPM}$ of the prior heart rate estimate.
3. **Sparse Spectrum Reconstruction & Dual-PPG Fusion (`focuss_spectrum.m`):** Second-order temporal differencing (`diff(clean_ppg, 2)`), $2\times$ decimation ($62.5\text{ Hz}$), and $6$ iterations of regularized FOCUSS ($p = 0.8$, $\lambda = 10^{-3}$, $M = 2048$), followed by peak-normalized dual-channel averaging.
4. **3-Stage Spectral Peak Tracking (`track_peak.m`):** Motion-bin attenuation, $[55, 135]\text{ BPM}$ initialization with sub-harmonic verification, fundamental/second-harmonic ($R_0 / R_1$) peak pairing, $\pm 8.0\text{ BPM}$ slew limiting, and 5-window trend escape.

---

## Repository Structure

```text
├── run_troika.m                              # Master driver script, subject segmentation & evaluation
├── load_ieeeppg_ts.m                         # Parser for Monash TSER / IEEE PPG .ts dataset
├── bandpass_fft.m                            # 0.4–5.0 Hz zero-phase Butterworth filter & FFT
├── ssa_denoise.m                             # Trajectory-protected Singular Spectrum Analysis (L = 400)
├── focuss_spectrum.m                         # 2nd-order difference + regularized FOCUSS spectrum
├── track_peak.m                              # 3-stage spectral peak tracking & verification
├── get_ecg_bpm.m                             # Dual-polarity QRS R-peak reference BPM estimator
├── local_peaks.m                             # Local spectral peak detection utility
├── split_subjects.m                          # Leave-One-Subject-Out (LOSO) data splitter
├── IEEEPPG_TRAIN.ts                          # Raw 5-channel time-series dataset (1,768 windows, 12 subjects)
├── ieeeppg_train_5channel.mat                # Pre-parsed MAT fallback tensor [1768 x 999 x 5]
├── troika_report_metrics.mat                 # Saved aggregate & window-level benchmark metrics
├── troika_subject_performance_summary.csv    # Per-subject MAE, RMSE, MAPE, Bias, r, and hit rates
└── figures/                                  # Exported PNG plots from fig1.fig – fig4.fig
