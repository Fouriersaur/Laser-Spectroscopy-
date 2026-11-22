# D2 Laser Spectroscopy – Rb saturated absorption

```
data/raw/     oscilloscope CSVs (x = time in s, y = voltage in V)  — git-ignored, local only
data/cache/   .npy copies of the CSVs for fast loading (auto-created) — git-ignored
scripts/      analysis scripts (run from anywhere)
figures/      output plots, one subfolder per dataset:
  doppler_free_6Nov/    raw Doppler-free trace
  doppler_free_13Nov/   FP frequency calibration + Doppler-background subtraction
  zeeman_19Nov/         Zeeman intensity/area vs B, theory fits (+ sigma_fits_per_field/)
  zeeman_20Nov/         Zeeman intensity vs B for 20 Nov, Model B fits
results/      processed data, e.g. frequency-calibrated spectra   — CSVs git-ignored
```

## Data

| file | contents |
|---|---|
| `5th_Nov_Doppler_Broad.CSV` | Doppler-broadened spectrum (C3) |
| `6th_Nov_Doppler_Free.CSV` | Doppler-free spectrum (C3) |
| `13_th_Nov_Doppler_Broad.CSV` | Doppler-broadened spectrum (C3) |
| `13th_Nov_Doppler_Free_and_FB.CSV` | C2 = Fabry-Perot etalon peaks, C3 = Doppler-free spectrum |

## Scripts

- `plot_doppler_free.py` – 6 Nov Doppler-free trace vs time → `figures/doppler_free_6Nov/doppler_free_6Nov.png`
- `plot_doppler_free_zoom.py` – zoom on one 6 Nov dip → `figures/doppler_free_6Nov/doppler_free_6Nov_zoom.png`
- `freq_calibrate_13Nov.py` – converts the 13 Nov time axis to frequency using the FP peaks
  (FSR = c/4L ≈ 375 MHz), checks it against literature Rb lines →
  `figures/doppler_free_13Nov/doppler_free_13Nov_freq{,_zoom}.png`, `figures/doppler_free_13Nov/calibration_curve_13Nov.png`,
  `results/doppler_free_13Nov_freq.csv`

- `subtract_doppler_13Nov.py` – aligns the 13 Nov Doppler-broadened trace to the Doppler-free
  trace (time warp + gain + baseline) and subtracts it →
  `figures/doppler_free_13Nov/doppler_free_13Nov_subtracted{,_zoom}.png`, `results/doppler_free_13Nov_subtracted.csv`
- `rb_lines.py` – literature Rb D2 hyperfine transitions/crossovers (Steck), shared module
- `zeeman_intensity_19Nov.py` – 19 Nov 87Rb F=2→F'=3 peak vs solenoid current
  (`data/raw/19th November Data/B<mA>PK.CSV` / `B<mA>BG.CSV`, B = 108 G/A × I): subtracts the
  smoothed Doppler-broadened trace, fits 2 Lorentzians + quadratic baseline, gives peak
  height/area vs field → `figures/zeeman_19Nov/zeeman_19Nov_{fits,intensity_vs_B}.png`,
  `results/zeeman_19Nov_intensity.csv`
- `zeeman_area_sigma_19Nov.py` – same data, fitted with the eq (9)+(11) line shape of Ummal Momeen
  et al. 2007 for σ⁺σ⁺: 5 Zeeman Lorentzians at μ_B·B·m/6 with a common, per-spectrum fitted width γ
  and free heights; intensity = fitted line area → `figures/zeeman_19Nov/zeeman_19Nov_sigma_{fits,area_vs_B}.png`,
  one plot per field in `figures/zeeman_19Nov/sigma_fits_per_field/`,
  `results/zeeman_19Nov_sigma_area.csv`
- `zeeman_theory_fit_19Nov.py` – four-level-model intensity vs field (eqs 1–11 of Ummal Momeen et al.
  2007: exact Zeeman/hyperfine diagonalisation, field-mixed transition strengths, Nakayama weights,
  area = −Σ I_ij) for σ⁺σ⁺ and σ⁻σ⁻, fitted to the measured areas (run after
  `zeeman_area_sigma_19Nov.py`) → `figures/zeeman_19Nov/zeeman_19Nov_sigma_area_theory_fit.png`,
  `results/zeeman_19Nov_theory_curve.csv`
- `zeeman_intensity_20Nov.py` – same peak-height/area analysis for 20 Nov
  (`data/raw/20th November Data/M<mA>PK.CSV` / `M<mA>BG.CSV`, `M094*01` = repeat at 0.094 A). Each spectrum
  is cut to −0.30…+0.40 ms around the F'=3 peak → `figures/zeeman_20Nov/zeeman_20Nov_{fits,intensity_vs_B}.png`,
  `results/zeeman_20Nov_intensity.csv`
- `model_B.py` – Model B (sublevel rate equations, Zeeman-resolved SAS): exact Zeeman/hyperfine levels and
  strengths, 5 + 16 sublevel steady state with σ⁺ pump and σ⁻ probe at full strength, transit replacement
  (Γτ = 284), balanced-detection subtraction, Maxwell–Boltzmann velocity sum; plus the weak-beam
  sub-resonances of the F'=3 cluster (Option 1 multiplet). `python3 scripts/model_B.py` runs self-checks
- `zeeman_modelB_fit_20Nov.py` – Model B fitted to the 20 Nov spectra: Option 1 multiplet fit (A·πG/2) and
  Option 2 full Model B template with one shared s_pump (s_probe = s_pump/9), both field signs; then
  height and intensity vs B = K × Model B (same extraction on model spectra), s_pump free, B ≥ 10 G
  (~40 min first run on 10 cores, templates cached in `data/cache/modelB/`) →
  `figures/zeeman_20Nov/zeeman_20Nov_modelB_{spectral_fits,intensity_vs_B}.png`,
  `results/zeeman_20Nov_modelB_{fit.csv,log.txt}`

```
python3 scripts/freq_calibrate_13Nov.py     # run first: makes the frequency axis
python3 scripts/subtract_doppler_13Nov.py
```
