"""Relative intensity of the 87Rb F=2 -> F'=3 Doppler-free peak vs solenoid magnetic field (20 Nov).
Same method as zeeman_intensity_19Nov.py.

Data: data/raw/20th November Data/M<III>PK.CSV and M<III>BG.CSV, where <III> is the solenoid
current in mA (e.g. M054 = 0.054 A, M0 = 0 A); M094PK01/BG01 = repeat at 0.094 A.
Up to 0.333 A the 1.6 ms window also holds the CO(2,3) crossover; every spectrum is cut to
t_F3 - 0.30 .. t_F3 + 0.40 ms around the F'=3 peak (as in zeeman_area_sigma_20Nov.py).
    PK = Doppler-free trace (pump on)     BG = Doppler-broadened trace (pump off), same current
Field: B = 108 G/A * I.

For each current:
 1. Block-average both traces (100 samples -> 80 ns, ~10k points).
 2. Smooth BG with an 8th-order Chebyshev polynomial (BG has no narrow structure) and subtract:
        d(t) = PK(t) - BG_smooth(t)
    d(t) is the Doppler-free peak sitting on a smooth pedestal (pump-induced offset/slope and the
    tail of the neighbouring crossover). A free gain on BG was tried and rejected: it is degenerate
    with the polynomial baseline (fitted gains came out between -2 and +4).
 3. Fit  d(t) = polynomial baseline + 2 line profiles.
    Two lines are needed because the Zeeman sub-components broaden the peak asymmetrically; a
    single profile leaves up to 17 mV rms residual at high field, two leave ~3-7 mV.
    Fitted variants: profile in {Lorentzian, Voigt} x baseline order in {1, 2, 3}.
    Central value = 2 Lorentzians + quadratic baseline; uncertainty = std over the 6 variants
    (systematic, dominates the statistical fit error).
 4. Intensity measures (both from the fitted peak function, baseline removed):
        height = maximum of the summed line profiles      (robust, ~few % model spread)
        area   = integral of the summed line profiles     (model-dependent: the split between
                                                            broad wings and pedestal is not unique)
 5. Also: height normalised by the transmitted BG level at the peak, |BG(t_peak)|, as a crude
    correction for laser-power drift between recordings (assumes 0 V = no light).

Usage:  python3 scripts/zeeman_intensity_20Nov.py
Output: figures/zeeman_20Nov/zeeman_20Nov_fits.png, figures/zeeman_20Nov/zeeman_20Nov_intensity_vs_B.png,
        results/zeeman_20Nov_intensity.csv
"""
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.special import voigt_profile

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/20th November Data"
CACHE = ROOT / "data/cache/20Nov"
CO_TO_F3, S_SCAN = 266.650 / 2, 207.5     # MHz, MHz/ms (scan rate from CO(2,3)->F'=3 spacing)
LEFT, RIGHT = -0.30, 0.40                 # fit window around F'=3 (ms)
FIG, RES = ROOT / "figures", ROOT / "results"
FIG = FIG / "zeeman_20Nov"
FIG.mkdir(parents=True, exist_ok=True)
G_PER_A = 108.0                # solenoid calibration (G/A)
BLOCK = 100                    # samples per block average
W_MAX = 0.12                   # max line half-width (ms) - keeps lines from turning into pedestal
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#222222", "#8a8a85"


# ---------------------------------------------------------------- io
def load(path):
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / (path.stem + ".npy")
    if cache.exists():
        return np.load(cache)
    with open(path, "rb") as f:
        f.readline()                      # header: "in s,C3 in V"
        raw = f.read().replace(b"\r", b"").replace(b"\n", b",")
    d = np.fromstring(raw, sep=",").reshape(-1, 2)
    np.save(cache, d)
    return d


def block_avg(x, n=BLOCK):
    return x[: len(x) // n * n].reshape(-1, n).mean(axis=1)


def recordings():
    """[(label, current_A, tag, suffix)] sorted by current; suffix '01' = repeat recording."""
    out = []
    for p in RAW.glob("M*PK*.CSV"):
        m = re.fullmatch(r"M(\d+)PK(\d*)", p.stem)
        out.append((m.group(1) + (f"-{m.group(2)}" if m.group(2) else ""), int(m.group(1)) / 1000.0,
                    m.group(1), m.group(2)))
    return sorted(out, key=lambda r: (r[1], r[3]))


def smooth_peak(t, v, mask):
    vs = np.convolve(v, np.ones(21) / 21, "same")
    i = np.argmax(np.where(mask, vs, -np.inf))
    s = slice(max(i - 40, 0), i + 41)
    c = np.polyfit(t[s] - t[i], vs[s], 2)
    return t[i] - c[1] / (2 * c[0])


# ---------------------------------------------------------------- line models
def lor(x, A, x0, g):
    return A * g**2 / ((x - x0) ** 2 + g**2)


def voi(x, A, x0, s, g):
    return A * voigt_profile(x - x0, s, g) / voigt_profile(0, s, g)


PROFILES = {  # name: (function, n params, area of one line)
    "L": (lor, 3, lambda A, x0, g: np.pi * A * g),
    "V": (voi, 4, lambda A, x0, s, g: A / voigt_profile(0, s, g)),
}


def split(p, prof, order):
    f, k, _ = PROFILES[prof]
    nb = order + 1
    return p[:nb], [p[nb:nb + k], p[nb + k:nb + 2 * k]]


def peak_fn(x, p, prof, order):
    f = PROFILES[prof][0]
    return sum(f(x, *q) for q in split(p, prof, order)[1])


def baseline_fn(x, p, prof, order):
    return np.polyval(split(p, prof, order)[0][::-1], x)


def fit(x, d, prof, order):
    i = np.argmax(np.convolve(d, np.ones(51) / 51, "same"))
    x0, A0 = x[i], d[i] - d.min()
    inf = np.inf
    b0, blo, bhi = [d.min()] + [0] * order, [-inf] * (order + 1), [inf] * (order + 1)
    if prof == "L":
        l1, lo1, hi1 = [A0, x0, .04], [0, x0 - .1, 1e-3], [3 * A0, x0 + .1, W_MAX]
        l2, lo2, hi2 = [A0 / 4, x0 + .05, .04], [0, x0 - .2, 1e-3], [3 * A0, x0 + .2, W_MAX]
    else:
        l1, lo1, hi1 = [A0, x0, .02, .02], [0, x0 - .1, 1e-4, 1e-4], [3 * A0, x0 + .1, W_MAX, W_MAX]
        l2, lo2, hi2 = [A0 / 4, x0 + .05, .02, .02], [0, x0 - .2, 1e-4, 1e-4], [3 * A0, x0 + .2, W_MAX, W_MAX]
    s = least_squares(lambda p: baseline_fn(x, p, prof, order) + peak_fn(x, p, prof, order) - d,
                      b0 + l1 + l2, bounds=(blo + lo1 + lo2, bhi + hi1 + hi2), x_scale="jac")
    p = s.x
    xx = np.linspace(x.min(), x.max(), 20001)
    pk = peak_fn(xx, p, prof, order)
    area = sum(PROFILES[prof][2](*q) for q in split(p, prof, order)[1])
    return dict(p=p, rms=s.fun.std(), height=pk.max(), t_peak=xx[np.argmax(pk)], area=area)


VARIANTS = [(prof, order) for prof in ("L", "V") for order in (1, 2, 3)]
CENTRAL = ("L", 2)

# ---------------------------------------------------------------- analyse each current
rows, panels = [], []
for label, I, tag, suffix in recordings():
    pk, bg = load(RAW / f"M{tag}PK{suffix}.CSV"), load(RAW / f"M{tag}BG{suffix}.CSV")
    t, v, vb = block_avg(pk[:, 0]) * 1e3, block_avg(pk[:, 1]), block_avg(bg[:, 1])
    xc = (t - t.mean()) / (np.ptp(t) / 2)
    cheb = np.polynomial.chebyshev.chebfit(xc, vb, 8)
    bg_s = np.polynomial.chebyshev.chebval(xc, cheb)
    edge = np.zeros_like(t, bool)
    edge[50:-50] = True
    if np.ptp(t) > 1.0:                   # 1.6 ms window: crossover + F'=3
        t_co = smooth_peak(t, v, edge)
        t3 = smooth_peak(t, v, edge & (np.abs(t - t_co - CO_TO_F3 / S_SCAN) < 0.1))
    else:                                 # 0.8 ms zoom on F'=3
        t3 = smooth_peak(t, v, edge)
    win = (t - t3 >= LEFT) & (t - t3 <= RIGHT)
    d = (v - bg_s)[win] * 1e3             # mV
    x = (t - t3)[win]                     # ms, relative to the F'=3 peak
    bg_s = bg_s[win]
    res = {var: fit(x, d, *var) for var in VARIANTS}
    c = res[CENTRAL]
    H = np.array([r["height"] for r in res.values()])
    Ar = np.array([r["area"] for r in res.values()])
    # transmitted background at the peak position (power-drift proxy)
    bg_at_pk = abs(np.interp(c["t_peak"], x, bg_s))
    rows.append([I, I * G_PER_A, c["height"], H.std(ddof=1), c["area"], Ar.std(ddof=1),
                 bg_at_pk, c["height"] / bg_at_pk, H.std(ddof=1) / bg_at_pk, c["rms"],
                 1.0 if suffix else 0.0])
    panels.append((label, I, x, d, c))
    print(f"{label:7s} I={I:.3f} A  B={I*G_PER_A:5.1f} G  height={c['height']:6.1f}±{H.std(ddof=1):4.1f} mV  "
          f"area={c['area']:6.1f}±{Ar.std(ddof=1):5.1f} mV·ms  |BG|={bg_at_pk:.3f} V  "
          f"fit rms={c['rms']:.1f} mV")

rows = np.array(rows)
I_A, B_G, H, dH, A, dA, BGp, Hn, dHn, rms, rep = rows.T

hdr = ("87Rb F=2->F'=3 Doppler-free peak vs solenoid field, 20 Nov. B = 108 G/A * I.\n"
       "height/area from 2-Lorentzian + quadratic-baseline fit of (PK - smoothed BG);\n"
       "errors = std over {Lorentzian,Voigt} x baseline order {1,2,3}. "
       "height_norm = height / |BG voltage at peak| (laser-power drift proxy).\n"
       "current_A,B_G,height_mV,height_err_mV,area_mV_ms,area_err_mV_ms,"
       "BG_at_peak_V,height_norm,height_norm_err,fit_rms_mV,repeat")
RES.mkdir(exist_ok=True)
np.savetxt(RES / "zeeman_20Nov_intensity.csv", rows, delimiter=",", header=hdr, fmt="%.6g")

# ---------------------------------------------------------------- plot: fits
n = len(panels)
ncol = 5
fig, axs = plt.subplots(int(np.ceil(n / ncol)), ncol, figsize=(20, 3.2 * np.ceil(n / ncol)),
                        sharex=False)
def draw_fit(a, tag, I, x, d, c):
    p = c["p"]
    base = baseline_fn(x, p, *CENTRAL)
    a.plot(x, d, lw=0.5, color=MUTED, label="PK − BG")
    a.plot(x, base + peak_fn(x, p, *CENTRAL), lw=1.2, color=BLUE, label="fit")
    a.plot(x, base, lw=1, color=ORANGE, ls="--", label="baseline")
    for q in split(p, *CENTRAL)[1]:
        a.plot(x, base + lor(x, *q), lw=0.7, color=BLUE, ls=":")
    a.plot(x, (d - base - peak_fn(x, p, *CENTRAL)) - 0.1 * c["height"], lw=0.4, color=INK)
    a.set_title(f"I = {I:.3f} A,  B = {I*G_PER_A:.1f} G" + (" (repeat)" if tag.endswith("-01") else ""),
                fontsize=10)
    a.set_xlabel("t − t$_{F'=3}$ (ms)")
    a.set_ylabel("mV")
    a.grid(alpha=0.25)


for a, panel in zip(axs.flat, panels):
    draw_fit(a, *panel)
axs.flat[0].legend(fontsize=8, loc="upper left")
for a in list(axs.flat)[n:]:
    a.axis("off")
fig.suptitle("20 Nov, $^{87}$Rb F=2→F'=3: Doppler-free − Doppler-broadened, fit = 2 Lorentzians "
             "(dotted) + quadratic baseline; black = residual (offset)", fontsize=11)
fig.tight_layout()
fig.savefig(FIG / "zeeman_20Nov_fits.png", dpi=150)

# ---------------------------------------------------------------- plot: each fit on its own
SINGLE = FIG / "fits_2lorentzian"
SINGLE.mkdir(exist_ok=True)
for k, panel in enumerate(panels):
    tag, I = panel[0], panel[1]
    fig, a = plt.subplots(figsize=(7, 4.8))
    draw_fit(a, *panel)
    a.set_title(a.get_title() + "\nfit = 2 Lorentzians (dotted) + quadratic baseline; black = residual (offset)",
                fontsize=10)
    a.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(SINGLE / (f"{k:02d}_B{I*G_PER_A:04.1f}G" + ("_repeat" if tag.endswith("-01") else "") + ".png"),
                dpi=150)
    plt.close(fig)

# ---------------------------------------------------------------- plot: intensity vs B
fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
for a, y, dy, title in [
    (axs[0], H, dH, "Peak height"),
    (axs[1], A, dA, "Peak area (integrated)"),
    (axs[2], Hn, dHn, "Peak height / |BG level|  (power-drift corrected)"),
]:
    s = y.max()
    m = rep == 0
    a.errorbar(B_G[m], y[m] / s, yerr=dy[m] / s, fmt="o", color=BLUE, ms=6, elinewidth=1.5,
               capsize=3, mec="white", mew=1)
    a.errorbar(B_G[~m], y[~m] / s, yerr=dy[~m] / s, fmt="D", color=BLUE, mfc="white", ms=6,
               elinewidth=1.5, capsize=3, label="repeat recording (10.2 G)")
    a.legend(fontsize=8, loc="lower right")
    a.set_title(title, fontsize=11)
    a.set_xlabel("Magnetic field (G)")
    a.set_ylabel("Relative intensity (a.u.)")
    a.set_ylim(0, 1.12)
    a.grid(alpha=0.25)
fig.suptitle("$^{87}$Rb D$_2$ F=2→F'=3 Doppler-free peak vs solenoid field (20 Nov), "
             "normalised to maximum", fontsize=12)
fig.tight_layout()
fig.savefig(FIG / "zeeman_20Nov_intensity_vs_B.png", dpi=200)
print(f"\nsaved figures to {FIG}, table to {RES / 'zeeman_20Nov_intensity.csv'}")
