"""Subtract the Doppler-broadened background (13 Nov) from the Doppler-free spectrum (13 Nov).

The two traces were recorded one after the other, so they do not overlap exactly: the
laser scan drifted slightly between them (dips shifted by up to ~0.3 ms) and the photodiode
baseline/gain differ (pump beam on vs off, power drift). A direct subtraction therefore leaves
large residues. Instead the broadened trace B(t) is first mapped onto the Doppler-free trace D(t):

    D(t) ≈ g · B(t + w(t)) + b(t)

    w(t) = cubic time warp     (scan drift between the two recordings)
    g    = gain                (pump beam removes part of the absorption)
    b(t) = quadratic baseline  (laser power / offset drift)

The fit uses only the parts of the trace away from the hyperfine clusters (where the
narrow saturated-absorption features live). The aligned background is then subtracted, leaving
the Doppler-free peaks on a flat baseline.

The frequency axis comes from freq_calibrate_13Nov.py (Fabry-Perot calibration), so run that first.

Usage:  python3 scripts/subtract_doppler_13Nov.py
Output: figures/doppler_free_13Nov/doppler_free_13Nov_subtracted{,_zoom}.png, results/doppler_free_13Nov_subtracted.csv
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from rb_lines import GROUPS, REF, group_span, mark_lines

ROOT = Path(__file__).resolve().parent.parent
RAW, CACHE_DIR, FIG, RES = ROOT / "data/raw", ROOT / "data/cache", ROOT / "figures", ROOT / "results"
FIG = FIG / "doppler_free_13Nov"
FIG.mkdir(parents=True, exist_ok=True)
BROAD = RAW / "13_th_Nov_Doppler_Broad.CSV"
CALIB = RES / "doppler_free_13Nov_freq.csv"     # from freq_calibrate_13Nov.py
MASK_PAD = 60.0                                 # MHz excluded either side of each cluster in the fit


def block_avg(x, n):
    k = len(x) // n * n
    return x[:k].reshape(-1, n).mean(axis=1)


def load_broad():
    cache = CACHE_DIR / (BROAD.stem + ".npy")
    try:
        return np.load(cache)
    except FileNotFoundError:
        with open(BROAD, "rb") as f:
            f.readline()  # header: "in s,C3 in V"
            raw = f.read().replace(b"\r", b"").replace(b"\n", b",")
        d = np.fromstring(raw, sep=",").reshape(-1, 2)
        np.save(cache, d)
        return d


# ---------------------------------------------------------------- load
if not CALIB.exists():
    raise SystemExit(f"{CALIB} not found - run scripts/freq_calibrate_13Nov.py first")
t, nu, _, v = np.loadtxt(CALIB, delimiter=",", unpack=True)   # Doppler-free, calibrated
b = load_broad()
tb, vb = block_avg(b[:, 0], 50), block_avg(b[:, 1], 50)       # same 0.32 us blocks as calib

# ---------------------------------------------------------------- fit alignment
T0 = t.mean()
x_t = lambda p, tt: tt + p[0] + p[1] * (tt - T0) + p[2] * (tt - T0) ** 2 + p[3] * (tt - T0) ** 3


def model(p, tt):
    dtt = tt - T0
    return p[4] * np.interp(x_t(p, tt), tb, vb) + p[5] + p[6] * dtt + p[7] * dtt ** 2


# fit on coarser points, excluding the hyperfine clusters
step = 20
use = np.ones_like(nu, bool)
for _, key in GROUPS:
    lo, hi = group_span(key)
    use &= ~((nu > lo - MASK_PAD) & (nu < hi + MASK_PAD))
tf, vf = t[use][::step], v[use][::step]

sol = least_squares(lambda p: model(p, tf) - vf, [0, 0, 0, 0, 1, 0, 0, 0],
                    x_scale=[1e-4, 1e-2, 1, 100, 1, 1e-2, 1, 100])
p = sol.x
bg = model(p, t)
diff = v - bg
shift = x_t(p, t) - t
print(f"alignment fit: rms residual (outside clusters) = {np.sqrt(np.mean(sol.fun**2))*1e3:.2f} mV")
print(f"  time shift of broadened trace: {shift.min()*1e3:+.3f} .. {shift.max()*1e3:+.3f} ms")
print(f"  gain g = {p[4]:.3f}")
print(f"  baseline noise of subtracted spectrum (outside clusters): {diff[use].std()*1e3:.2f} mV rms")

# ---------------------------------------------------------------- save
hdr = ("13 Nov Doppler-free minus aligned Doppler-broadened background.\n"
       f"freq_rel = detuning from 85Rb F=3->F'=4 ({REF/1e6:.6f} THz), FP-calibrated.\n"
       "time_s,freq_rel_MHz,doppler_free_V,broad_aligned_V,subtracted_V")
np.savetxt(RES / "doppler_free_13Nov_subtracted.csv", np.column_stack([t, nu, v, bg, diff]),
           delimiter=",", header=hdr, fmt=["%.8e", "%.3f", "%.6e", "%.6e", "%.6e"])

# ---------------------------------------------------------------- plot: full range
g = nu / 1e3
fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw=dict(height_ratios=[1, 1]))
a1.plot(g, v, lw=0.5, color="C0", label="Doppler-free (pump on)")
a1.plot(g, bg, lw=0.8, color="C1", label="Doppler-broadened (pump off), aligned")
a1.set_ylabel("Photodiode voltage, C3 (V)")
a1.legend(loc="lower right")
a1.set_title("13 Nov: Doppler-broadened background subtraction")
a2.plot(g, diff * 1e3, lw=0.5, color="k")
a2.set_ylabel("Doppler-free − broadened (mV)")
a2.set_xlabel("Laser detuning from $^{85}$Rb F=3→F'=4 (GHz)")
for name, key in GROUPS:
    lo, hi = group_span(key)
    a2.annotate(name.split(" →")[0], ((lo + hi) / 2e3, a2.get_ylim()[1]), xytext=(0, -14),
                textcoords="offset points", ha="center")
for a in (a1, a2):
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "doppler_free_13Nov_subtracted.png", dpi=200)

# ---------------------------------------------------------------- plot: each hyperfine group
fig, axs = plt.subplots(2, 2, figsize=(14, 9))
for a, (title, key) in zip(axs.flat, GROUPS):
    lo, hi = group_span(key)
    m = (nu > lo - 150) & (nu < hi + 150)
    a.plot(nu[m], diff[m] * 1e3, lw=0.6, color="k")
    a.margins(y=0.25)
    mark_lines(a, key)
    a.set_title(title)
    a.set_xlabel("Detuning from $^{85}$Rb F=3→F'=4 (MHz)")
    a.set_ylabel("Doppler-free − broadened (mV)")
    a.grid(alpha=0.3)
fig.suptitle("Background-subtracted Doppler-free spectra (red: transitions, green dashed: crossovers)")
fig.tight_layout()
fig.savefig(FIG / "doppler_free_13Nov_subtracted_zoom.png", dpi=200)
print(f"\nsaved figures to {FIG}, data to {RES}")
