"""Convert the oscilloscope time axis of the 13 Nov Doppler-free trace into optical frequency.

Method
------
1. C2 = transmission of the confocal Fabry-Perot etalon (L = 20 cm). Adjacent peaks are
   separated by one free spectral range, FSR = c / 4L = 374.7 MHz (confocal, re-entrant path 4L).
2. Each FP peak time t_k is found (centroid), and peak number k is fitted as a smooth polynomial
   k(t). This corrects for the non-linear PZT sweep (peak spacing drifts 1.89 -> 1.59 ms).
   nu(t) = FSR * k(t) + const.
3. The FSR is then cross-checked (and refined) by fitting the measured positions of identified
   Doppler-free hyperfine/crossover peaks against literature values (D. A. Steck, Rb85/Rb87 D-line
   data), which also fixes the absolute offset.

Note: the FP peak train shows a discontinuity at ~61-63 ms (spacing jumps 1.60 -> 2.01 ms), i.e.
a mode hop / scan reset. Data after that point are NOT on the same frequency axis and are dropped.

Usage:  python3 scripts/freq_calibrate_13Nov.py   (needs data/raw/13th_Nov_Doppler_Free_and_FB.CSV or its cache)
Output: figures/doppler_free_13Nov/doppler_free_13Nov_freq{,_zoom}.png, results/doppler_free_13Nov_freq.csv
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW, CACHE_DIR, FIG, RES = ROOT / "data/raw", ROOT / "data/cache", ROOT / "figures", ROOT / "results"
FIG = FIG / "doppler_free_13Nov"
FIG.mkdir(parents=True, exist_ok=True)

FILE = RAW / "13th_Nov_Doppler_Free_and_FB.CSV"
CACHE = CACHE_DIR / (FILE.stem + ".npy")
C = 299_792_458.0
L_FP = 0.20                      # m, nominal confocal etalon length
FSR_NOM = C / (4 * L_FP) / 1e6   # MHz
T_HOP = 61.5e-3                  # s, mode hop / scan reset seen in FP train
POLY_DEG = 5

# ---------------------------------------------------------------- load
try:
    data = np.load(CACHE)
except FileNotFoundError:
    with open(FILE, "rb") as f:
        f.readline()  # header: "in s,C2 in V,C3 in V"
        raw = f.read().replace(b"\r", b"").replace(b"\n", b",")
    data = np.fromstring(raw, sep=",").reshape(-1, 3)
    np.save(CACHE, data)
t_raw, fp_raw, df_raw = data[:, 0], data[:, 1], data[:, 2]


def block_avg(x, n):
    k = len(x) // n * n
    return x[:k].reshape(-1, n).mean(axis=1)


# ---------------------------------------------------------------- 1. FP peak times
t_fp, v_fp = block_avg(t_raw, 10), block_avg(fp_raw, 10)
dt = t_fp[1] - t_fp[0]
idx, _ = find_peaks(v_fp, height=0.4 * v_fp.max(), distance=int(0.8e-3 / dt))
half = int(30e-6 / dt)
t_pk = []
for i in idx:
    s = slice(max(i - half, 0), i + half)
    w = np.clip(v_fp[s] - 0.1 * v_fp[i], 0, None)
    t_pk.append((t_fp[s] * w).sum() / w.sum())
t_pk = np.array(t_pk)
t_pk = t_pk[t_pk < T_HOP]
k_pk = np.arange(len(t_pk))
print(f"{len(t_pk)} FP peaks before mode hop; spacing {np.diff(t_pk).max()*1e3:.3f} -> "
      f"{np.diff(t_pk).min()*1e3:.3f} ms")

# ---------------------------------------------------------------- 2. time -> FSR number
TS = 1e3  # fit in ms for conditioning
coef = np.polyfit(t_pk * TS, k_pk, POLY_DEG)
resid = k_pk - np.polyval(coef, t_pk * TS)
print(f"poly deg {POLY_DEG}: rms residual {resid.std():.4f} FSR = {resid.std()*FSR_NOM:.2f} MHz")


def t_to_k(t):
    return np.polyval(coef, t * TS)


# Doppler-free signal, lightly averaged (0.32 us blocks, << feature width)
N = 50
t = block_avg(t_raw, N)
v = block_avg(df_raw, N)
keep = t < T_HOP
t, v = t[keep], v[keep]
k = t_to_k(t)

# ---------------------------------------------------------------- 3. literature lines (MHz)
from rb_lines import LINES, REF, GROUPS, group_span, mark_lines  # noqa: E402

# ---------------------------------------------------------------- 4. find + identify DF peaks
# remove Doppler pedestal with a wide Savitzky-Golay filter, then fit sharp peaks
dk = np.median(np.diff(k))
v_hp = v - savgol_filter(v, int(0.15 / dk) | 1, 2)
pk, prop = find_peaks(v_hp, prominence=0.008, distance=int(0.03 / dk))


def lor(x, a, x0, g, c0, c1):
    return a / (1 + ((x - x0) / (g / 2)) ** 2) + c0 + c1 * (x - x0)


# rough absolute offset: anchor the strongest peak near 85Rb F=3 group to its CO34
fits = []
for i in pk:
    w = int(0.025 / dk)
    s = slice(max(i - w, 0), i + w)
    try:
        p, cov = curve_fit(lor, k[s], v[s], p0=[v_hp[i], k[i], 0.02, v[s].min(), 0])
        if abs(p[1] - k[i]) < 0.02 and 0.003 < p[2] < 0.1:
            fits.append((p[1], np.sqrt(cov[1, 1]), abs(p[2]), p[0]))
    except RuntimeError:
        pass
fits = np.array(fits)

lit_nu = np.array([l[1] for l in LINES])
# initial mapping with nominal FSR: anchor on the tallest peak (expected 85Rb F=3 CO34)
k_tall = fits[np.argmax(fits[:, 3]), 0]
co34 = [l[1] for l in LINES if l[0] == "85Rb F=3 CO34"][0]
off = co34 - FSR_NOM * k_tall
matched = []
for _ in range(3):  # iterate: match -> refit FSR/offset -> rematch
    nu_obs = off + FSR_NOM * fits[:, 0] if not matched else off + fsr * fits[:, 0]
    matched = []
    for j, nu in enumerate(nu_obs):
        d = np.abs(lit_nu - nu)
        if d.min() < 12:
            matched.append((j, int(d.argmin())))
    kk = np.array([fits[j, 0] for j, _ in matched])
    ll = np.array([lit_nu[m] for _, m in matched])
    (fsr, off), cov = np.polyfit(kk, ll, 1, cov=True)
res = ll - (off + fsr * kk)
print(f"\nFSR fitted from {len(kk)} identified Rb features: {fsr:.2f} ± {np.sqrt(cov[0,0]):.2f} MHz "
      f"(nominal c/4L = {FSR_NOM:.2f} MHz) -> L = {C/4/fsr/1e6*100:.3f} cm; rms residual {res.std():.2f} MHz")
print(f"{'feature':24s} {'measured':>10s} {'literature':>10s} {'diff':>7s} {'FWHM':>6s}   (MHz, rel. 85Rb F=3→F'=4)")
for (j, m), r in zip(matched, res):
    print(f"{LINES[m][0]:24s} {off+fsr*fits[j,0]:10.1f} {lit_nu[m]:10.1f} {-r:7.1f} {fits[j,2]*fsr:6.1f}")

nu = off + fsr * k                    # MHz relative to 85Rb F=3 -> F'=4
nu_GHz = nu / 1e3

# ---------------------------------------------------------------- 5. save + plot
hdr = (f"Doppler-free spectrum 13 Nov, frequency-calibrated with confocal FP (FSR={fsr:.2f} MHz).\n"
       f"freq_rel = detuning from 85Rb F=3->F'=4 ({REF/1e6:.6f} THz). Data after mode hop (t>{T_HOP*1e3} ms) dropped.\n"
       "time_s,freq_rel_MHz,abs_freq_THz,C3_V")
np.savetxt(RES / "doppler_free_13Nov_freq.csv",
           np.column_stack([t, nu, (REF + nu) / 1e6, v]), delimiter=",", header=hdr,
           fmt=["%.8e", "%.3f", "%.9f", "%.6e"])

fig, ax = plt.subplots(figsize=(13, 5.5))
ax.plot(nu_GHz, v, lw=0.5, color="C0")
for name, x in [("$^{87}$Rb F=2", 7.0), ("$^{85}$Rb F=3", 10.0), ("$^{85}$Rb F=2", 17.7), ("$^{87}$Rb F=1", 24.4)]:
    xx = (off + fsr * x) / 1e3
    ax.annotate(name, (xx, ax.get_ylim()[0]), xytext=(0, 8), textcoords="offset points", ha="center")
ax.set_xlabel("Laser detuning from $^{85}$Rb F=3→F'=4 (GHz)")
ax.set_ylabel("Photodiode voltage, C3 (V)")
ax.set_title(f"Doppler-free saturated absorption spectrum of Rb (13 Nov), FP-calibrated (FSR = {fsr:.1f} MHz)")
sec = ax.secondary_xaxis("top", functions=(lambda g: g * 1e3 / fsr, lambda n: n * fsr / 1e3))
sec.set_xlabel("Fabry-Perot fringe number")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "doppler_free_13Nov_freq.png", dpi=200)

fig, axs = plt.subplots(2, 2, figsize=(14, 9))
for a, (title, key) in zip(axs.flat, GROUPS):
    lo, hi = group_span(key)
    m = (nu > lo - 250) & (nu < hi + 250)
    a.plot(nu[m], v[m], lw=0.6, color="C0")
    mark_lines(a, key)
    a.set_title(title)
    a.set_xlabel("Detuning from $^{85}$Rb F=3→F'=4 (MHz)")
    a.set_ylabel("C3 (V)")
    a.grid(alpha=0.3)
fig.suptitle("Doppler-free features vs. literature (red: transitions, green dashed: crossovers)")
fig.tight_layout()
fig.savefig(FIG / "doppler_free_13Nov_freq_zoom.png", dpi=200)

# ---------------------------------------------------------------- 6. calibration curve figure
# full calibration nu(t) [MHz] = off + fsr * poly(t[ms]); print it for the lab book
cal = fsr * coef
cal[-1] += off
print("\ncalibration curve  nu(t) = sum_i a_i * t^i   (nu in MHz rel. 85Rb F=3→F'=4, t in ms):")
for i, a in enumerate(cal[::-1]):
    print(f"  a_{i} = {a:+.6e}")
nu_of_t = lambda tt: np.polyval(cal, tt * TS)
rate = np.polyval(np.polyder(cal), t * TS)                      # MHz / ms
print(f"scan rate dnu/dt: {rate.min():.0f} .. {rate.max():.0f} MHz/ms")

nu_pk = off + fsr * k_pk                                         # FP peaks: one FSR apart
t_feat = np.interp(kk, k, t)                                     # times of identified Rb features
tt = np.linspace(t[0], t[-1], 2000)
lin = np.polyfit(t_pk * TS, nu_pk, 1)                            # straight line through FP peaks

fig, axs = plt.subplots(4, 1, figsize=(12, 13), sharex=True,
                        gridspec_kw=dict(height_ratios=[1, 1.6, 1, 1]))
a0, a1, a2, a3 = axs
ms = lambda x: x * TS

a0.plot(ms(t_fp), v_fp * 1e3, lw=0.4, color="0.4")
a0.plot(ms(t_pk), np.interp(t_pk, t_fp, v_fp) * 1e3, "v", color="C3", ms=5, label="FP peaks used")
a0.axvspan(ms(T_HOP), ms(t_fp[-1]), color="0.85", label="after mode hop (discarded)")
a0.set_ylabel("FP transmission,\nC2 (mV)")
a0.set_title("(a) Fabry-Perot etalon transmission: peaks are one FSR apart")
a0.legend(loc="upper right", fontsize=8)

a1.plot(ms(tt), nu_of_t(tt) / 1e3, color="C0", lw=1.5, label=f"calibration curve (deg-{POLY_DEG} polynomial)")
a1.plot(ms(t_pk), nu_pk / 1e3, "o", color="C3", ms=4, label=f"FP peaks: ν = ν₀ + n·FSR,  FSR = {fsr:.2f} MHz")
a1.plot(ms(t_feat), ll / 1e3, "s", mfc="none", color="C2", ms=7, mew=1.3,
        label="Rb Doppler-free features (literature ν)")
a1.plot(ms(tt), np.polyval(lin, ms(tt)) / 1e3, "--", color="0.5", lw=1, label="linear (constant scan rate)")
a1.set_ylabel("Detuning from $^{85}$Rb F=3→F'=4 (GHz)")
a1.set_title("(b) Time → frequency calibration")
a1.legend(loc="upper left", fontsize=8)

a2.plot(ms(tt), nu_of_t(tt) - np.polyval(lin, ms(tt)), color="C0", lw=1.5, label="calibration − linear")
a2.plot(ms(t_pk), nu_pk - np.polyval(lin, ms(t_pk)), "o", color="C3", ms=4)
a2.set_ylabel("Non-linearity (MHz)")
a2t = a2.twinx()
a2t.plot(ms(t), rate, color="C1", lw=1, ls=":")
a2t.set_ylabel("Scan rate dν/dt (MHz/ms)", color="C1")
a2t.tick_params(axis="y", colors="C1")
a2.set_title("(c) PZT scan non-linearity (deviation from straight line) and local scan rate")
a2.legend(loc="upper left", fontsize=8)

a3.axhline(0, color="0.5", lw=0.8)
a3.plot(ms(t_pk), (resid * fsr), "o", color="C3", ms=4,
        label=f"FP peaks − curve (rms {resid.std()*fsr:.2f} MHz)")
a3.plot(ms(t_feat), ll - nu_of_t(t_feat), "s", mfc="none", color="C2", ms=7, mew=1.3,
        label=f"Rb literature − calibrated (rms {res.std():.2f} MHz)")
a3.set_ylabel("Residual (MHz)")
a3.set_xlabel("Oscilloscope time (ms)")
a3.set_title("(d) Calibration residuals")
a3.legend(loc="upper left", fontsize=8)

for a in axs:
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "calibration_curve_13Nov.png", dpi=200)
print(f"\nsaved figures to {FIG}, calibrated data to {RES}")
