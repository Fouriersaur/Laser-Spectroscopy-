"""Relative intensity of the 87Rb F=2 -> F'=3 Doppler-free peak vs solenoid magnetic field (19 Nov).

Data: data/raw/19th November Data/B<III>PK.CSV and B<III>BG.CSV, where <III> is the solenoid
current in mA (e.g. B054 = 0.054 A, B0 = 0 A).
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

Usage:  python3 scripts/zeeman_intensity_19Nov.py
Output: figures/zeeman_19Nov/zeeman_19Nov_fits.png, figures/zeeman_19Nov/zeeman_19Nov_intensity_vs_B.png,
        results/zeeman_19Nov_intensity.csv
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.special import voigt_profile

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/19th November Data"
CACHE = ROOT / "data/cache/19Nov"
FIG, RES = ROOT / "figures", ROOT / "results"
FIG = FIG / "zeeman_19Nov"
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


def current_of(tag):                      # "054" -> 0.054 A, "0" -> 0 A
    return int(tag) / 1000.0


tags = sorted({p.stem[1:-2] for p in RAW.glob("B*PK.CSV")}, key=int)


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
for tag in tags:
    pk, bg = load(RAW / f"B{tag}PK.CSV"), load(RAW / f"B{tag}BG.CSV")
    t, v, vb = block_avg(pk[:, 0]), block_avg(pk[:, 1]), block_avg(bg[:, 1])
    xc = (t - t.mean()) / (np.ptp(t) / 2)
    cheb = np.polynomial.chebyshev.chebfit(xc, vb, 8)
    bg_s = np.polynomial.chebyshev.chebval(xc, cheb)
    d = (v - bg_s) * 1e3                  # mV
    x = (t - t.mean()) * 1e3              # ms, centred
    res = {var: fit(x, d, *var) for var in VARIANTS}
    c = res[CENTRAL]
    H = np.array([r["height"] for r in res.values()])
    Ar = np.array([r["area"] for r in res.values()])
    # transmitted background at the peak position (power-drift proxy)
    bg_at_pk = abs(np.interp(c["t_peak"], x, bg_s))
    I = current_of(tag)
    rows.append([I, I * G_PER_A, c["height"], H.std(ddof=1), c["area"], Ar.std(ddof=1),
                 bg_at_pk, c["height"] / bg_at_pk, H.std(ddof=1) / bg_at_pk, c["rms"]])
    panels.append((tag, I, x, d, c))
    print(f"I={I:.3f} A  B={I*G_PER_A:5.1f} G  height={c['height']:6.1f}±{H.std(ddof=1):4.1f} mV  "
          f"area={c['area']:6.1f}±{Ar.std(ddof=1):5.1f} mV·ms  |BG|={bg_at_pk:.3f} V  "
          f"fit rms={c['rms']:.1f} mV")

rows = np.array(rows)
I_A, B_G, H, dH, A, dA, BGp, Hn, dHn, rms = rows.T

hdr = ("87Rb F=2->F'=3 Doppler-free peak vs solenoid field, 19 Nov. B = 108 G/A * I.\n"
       "height/area from 2-Lorentzian + quadratic-baseline fit of (PK - smoothed BG);\n"
       "errors = std over {Lorentzian,Voigt} x baseline order {1,2,3}. "
       "height_norm = height / |BG voltage at peak| (laser-power drift proxy).\n"
       "current_A,B_G,height_mV,height_err_mV,area_mV_ms,area_err_mV_ms,"
       "BG_at_peak_V,height_norm,height_norm_err,fit_rms_mV")
RES.mkdir(exist_ok=True)
np.savetxt(RES / "zeeman_19Nov_intensity.csv", rows, delimiter=",", header=hdr, fmt="%.6g")

# ---------------------------------------------------------------- plot: fits
n = len(panels)
ncol = 4
fig, axs = plt.subplots(int(np.ceil(n / ncol)), ncol, figsize=(16, 3.2 * np.ceil(n / ncol)),
                        sharex=False)
for a, (tag, I, x, d, c) in zip(axs.flat, panels):
    p = c["p"]
    base = baseline_fn(x, p, *CENTRAL)
    a.plot(x, d, lw=0.5, color=MUTED, label="PK − BG")
    a.plot(x, base + peak_fn(x, p, *CENTRAL), lw=1.2, color=BLUE, label="fit")
    a.plot(x, base, lw=1, color=ORANGE, ls="--", label="baseline")
    for q in split(p, *CENTRAL)[1]:
        a.plot(x, base + lor(x, *q), lw=0.7, color=BLUE, ls=":")
    a.plot(x, (d - base - peak_fn(x, p, *CENTRAL)) - 0.1 * c["height"], lw=0.4, color=INK)
    a.set_title(f"I = {I:.3f} A,  B = {I*G_PER_A:.1f} G", fontsize=10)
    a.set_xlabel("t − t$_{mid}$ (ms)")
    a.set_ylabel("mV")
    a.grid(alpha=0.25)
axs.flat[0].legend(fontsize=8, loc="upper left")
for a in list(axs.flat)[n:]:
    a.axis("off")
fig.suptitle("19 Nov, $^{87}$Rb F=2→F'=3: Doppler-free − Doppler-broadened, fit = 2 Lorentzians "
             "(dotted) + quadratic baseline; black = residual (offset)", fontsize=11)
fig.tight_layout()
fig.savefig(FIG / "zeeman_19Nov_fits.png", dpi=150)

# ---------------------------------------------------------------- plot: intensity vs B
fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
for a, y, dy, title in [
    (axs[0], H, dH, "Peak height"),
    (axs[1], A, dA, "Peak area (integrated)"),
    (axs[2], Hn, dHn, "Peak height / |BG level|  (power-drift corrected)"),
]:
    s = y.max()
    a.errorbar(B_G, y / s, yerr=dy / s, fmt="o", color=BLUE, ms=6, elinewidth=1.5, capsize=3,
               mec="white", mew=1)
    a.set_title(title, fontsize=11)
    a.set_xlabel("Magnetic field (G)")
    a.set_ylabel("Relative intensity (a.u.)")
    a.set_ylim(0, 1.12)
    a.grid(alpha=0.25)
fig.suptitle("$^{87}$Rb D$_2$ F=2→F'=3 Doppler-free peak vs solenoid field (19 Nov), "
             "normalised to maximum", fontsize=12)
fig.tight_layout()
fig.savefig(FIG / "zeeman_19Nov_intensity_vs_B.png", dpi=200)
print(f"\nsaved figures to {FIG}, table to {RES / 'zeeman_19Nov_intensity.csv'}")
