"""Integrated intensity (area) of the 87Rb F=2 -> F'=3 Doppler-free line vs solenoid field, 19 Nov,
fitted with the Zeeman-resolved Lorentzian line shape of eqs (9)+(11) of
Ummal Momeen et al., J. Phys. B 40 3163 (2007), for sigma+ / sigma+ (or sigma- / sigma-) pump/probe.

Line shape
----------
Im of eq (9) = sum of Lorentzians with a common half-width gamma. With the field along the beams and
both beams sigma+, the F=2 -> F'=3 line splits into five principal resonances
    (F=2, m) -> (F'=3, m+1),   nu_m = nu_0 + mu_B B (2/3 + m/6),   m = -2 .. 2
(g_F = 1/2, g_F' = 2/3, mu_B = 1.400 MHz/G). sigma-/sigma- gives the mirror image (m -> -m); both fit
identically here because the scan direction (sign of the frequency axis) is unknown.
The four-level weights of eq (10) predict the wrong sign for this closed transition (single-cycle
model, no saturation of the cycling m=2 -> m'=3 line), so the five heights are left free (>= 0):

    d(t) = c0 + c1 t + c2 t^2 + sum_m A_m gamma^2 / ([s (t - t0) - mu_B B m/6]^2 + gamma^2)

    free per spectrum : t0, gamma (common to the 5 lines), A_-2..A_2, baseline c0..c2
    fixed             : B = 108 G/A * I, Zeeman offsets mu_B B m/6
    s (MHz/ms)        : no frequency reference on 19 Nov; fixed at 224 MHz/ms (13 Nov Fabry-Perot
                        scan rate). The fit itself barely constrains s (rms 6.8-7.5 mV for 150-400).

Intensity = area of the fitted line = pi * gamma * sum(A_m)   [mV MHz], also given in mV ms (/ s).
Uncertainty = spread over s in {150, 224, 300} MHz/ms x baseline order {1, 2, 3} (systematic, std)
              combined in quadrature with the statistical fit error of the central fit.

d(t) = PK - smoothed BG, exactly as in zeeman_intensity_19Nov.py.

Usage:  python3 scripts/zeeman_area_sigma_19Nov.py
Output: figures/zeeman_19Nov/zeeman_19Nov_sigma_fits.png, figures/zeeman_19Nov/zeeman_19Nov_sigma_area_vs_B.png,
        results/zeeman_19Nov_sigma_area.csv
"""
from itertools import product
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/19th November Data"
CACHE = ROOT / "data/cache/19Nov"
FIG, RES = ROOT / "figures", ROOT / "results"
FIG = FIG / "zeeman_19Nov"
FIG.mkdir(parents=True, exist_ok=True)
G_PER_A = 108.0                   # solenoid calibration (G/A)
MU_B = 1.39962                    # MHz/G
M = np.arange(-2, 3)              # ground m of the five sigma+ components
S_CENTRAL, S_RANGE = 224.0, (150.0, 224.0, 300.0)   # MHz/ms
ORDER_CENTRAL, ORDERS = 2, (1, 2, 3)
BLOCK = 100
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


def prepare(tag):
    pk, bg = load(RAW / f"B{tag}PK.CSV"), load(RAW / f"B{tag}BG.CSV")
    t, v, vb = block_avg(pk[:, 0]), block_avg(pk[:, 1]), block_avg(bg[:, 1])
    xc = (t - t.mean()) / (np.ptp(t) / 2)
    bg_s = np.polynomial.chebyshev.chebval(xc, np.polynomial.chebyshev.chebfit(xc, vb, 8))
    return (t - t.mean()) * 1e3, (v - bg_s) * 1e3          # ms (centred), mV


# ---------------------------------------------------------------- model
def components(x, B, s, x0, g, A):
    """The five Lorentzians (rows = points, cols = m), heights A, common HWHM g (MHz)."""
    nu = s * (x - x0)
    off = MU_B * B * M / 6
    return A[None, :] * g**2 / ((nu[:, None] - off[None, :]) ** 2 + g**2)


def baseline(x, c):
    return np.polyval(c[::-1], x)


def fit(x, d, B, s, order):
    i = np.argmax(np.convolve(d, np.ones(51) / 51, "same"))
    A0 = d[i] - d.min()
    p0 = np.r_[x[i], 8.0, np.full(5, A0 / 3), d.min(), np.zeros(order)]
    lo = np.r_[x[i] - 0.2, 0.5, np.zeros(5), np.full(order + 1, -np.inf)]
    hi = np.r_[x[i] + 0.2, 80.0, np.full(5, 5 * A0), np.full(order + 1, np.inf)]
    res = lambda p: baseline(x, p[7:]) + components(x, B, s, p[0], p[1], p[2:7]).sum(1) - d
    sol = least_squares(res, p0, bounds=(lo, hi), x_scale="jac")
    p = sol.x
    g, A = p[1], p[2:7]
    area = np.pi * g * A.sum()                               # mV MHz
    # statistical error of the area from the fit covariance
    J = sol.jac
    dof = len(d) - len(p)
    cov = np.linalg.pinv(J.T @ J) * (sol.fun @ sol.fun) / dof
    grad = np.zeros_like(p)
    grad[1] = np.pi * A.sum()
    grad[2:7] = np.pi * g
    area_err = np.sqrt(grad @ cov @ grad)
    g_err = np.sqrt(cov[1, 1])
    return dict(p=p, gamma=g, gamma_err=g_err, A=A, area=area, area_err=area_err,
                rms=np.sqrt(np.mean(sol.fun**2)), s=s, order=order)


# ---------------------------------------------------------------- run
tags = sorted({p.stem[1:-2] for p in RAW.glob("B*PK.CSV")}, key=int)
rows, panels = [], []
for tag in tags:
    I = int(tag) / 1000.0
    B = I * G_PER_A
    x, d = prepare(tag)
    var = {(s, o): fit(x, d, B, s, o) for s, o in product(S_RANGE, ORDERS)}
    c = var[(S_CENTRAL, ORDER_CENTRAL)]
    area_ms = np.array([r["area"] / r["s"] for r in var.values()])        # mV ms, s-independent unit
    a_c = c["area"] / c["s"]
    err = np.hypot(area_ms.std(ddof=1), c["area_err"] / c["s"])
    rows.append([I, B, a_c, err, c["area"], err * c["s"], c["gamma"], c["gamma_err"], c["rms"],
                 *c["A"]])
    panels.append((I, B, x, d, c))
    print(f"I={I:.3f} A  B={B:5.1f} G  area={a_c:6.1f} ± {err:4.1f} mV·ms  "
          f"gamma={c['gamma']:5.2f} MHz (HWHM)  rms={c['rms']:.1f} mV")

rows = np.array(rows)
I_A, B_G, area, area_err = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]

hdr = ("87Rb F=2->F'=3 Doppler-free line area vs solenoid field, 19 Nov. B = 108 G/A * I.\n"
       "Fit: 5 sigma+ Zeeman Lorentzians (positions mu_B B m/6, common HWHM gamma, free heights) "
       f"+ quadratic baseline; s = {S_CENTRAL:.0f} MHz/ms.\n"
       "area = pi*gamma*sum(A_m). err = std over s in {150,224,300} x baseline order {1,2,3} "
       "(+) fit error.\n"
       "current_A,B_G,area_mV_ms,area_err_mV_ms,area_mV_MHz,area_err_mV_MHz,gamma_HWHM_MHz,"
       "gamma_err_MHz,fit_rms_mV,A_m-2_mV,A_m-1_mV,A_m0_mV,A_m+1_mV,A_m+2_mV")
RES.mkdir(exist_ok=True)
np.savetxt(RES / "zeeman_19Nov_sigma_area.csv", rows, delimiter=",", header=hdr, fmt="%.6g")

# ---------------------------------------------------------------- plot: fits
def draw_fit(a, I, B, x, d, c, labels=False):
    p = c["p"]
    base = baseline(x, p[7:])
    comp = components(x, B, c["s"], p[0], p[1], p[2:7])
    model = base + comp.sum(1)
    a.plot(x, d, lw=0.5, color=MUTED, label="PK − BG")
    a.plot(x, model, lw=1.2, color=BLUE, label="fit")
    a.plot(x, base, lw=1, color=ORANGE, ls="--", label="baseline")
    for k in range(5):
        a.plot(x, base + comp[:, k], lw=0.6, color=BLUE, ls=":",
               label="σ$^+$ components" if (labels and k == 0) else None)
    a.plot(x, d - model - 0.1 * comp.sum(1).max(), lw=0.4, color=INK,
           label="residual (offset)" if labels else None)
    a.set_xlabel("t − t$_{mid}$ (ms)")
    a.set_ylabel("mV")
    a.grid(alpha=0.25)


n, ncol = len(panels), 4
nrow = int(np.ceil(n / ncol))
fig, axs = plt.subplots(nrow, ncol, figsize=(16, 3.3 * nrow))
for a, (I, B, x, d, c) in zip(axs.flat, panels):
    draw_fit(a, I, B, x, d, c)
    a.set_title(f"B = {B:.1f} G   γ = {c['gamma']:.1f} MHz", fontsize=10)
axs.flat[0].legend(fontsize=8, loc="upper left")
for a in list(axs.flat)[n:]:
    a.axis("off")
fig.suptitle("19 Nov, $^{87}$Rb F=2→F'=3: fit = 5 σ$^+$ Zeeman Lorentzians (dotted, common width γ, "
             f"s = {S_CENTRAL:.0f} MHz/ms) + quadratic baseline; black = residual (offset)", fontsize=11)
fig.tight_layout()
fig.savefig(FIG / "zeeman_19Nov_sigma_fits.png", dpi=150)
plt.close(fig)

# one figure per field
IND = FIG / "sigma_fits_per_field"
IND.mkdir(exist_ok=True)
for row, (I, B, x, d, c) in zip(rows, panels):
    fig, a = plt.subplots(figsize=(8, 5))
    draw_fit(a, I, B, x, d, c, labels=True)
    a.set_title(f"19 Nov, $^{{87}}$Rb F=2→F'=3, I = {I:.3f} A, B = {B:.1f} G\n"
                f"γ = {c['gamma']:.2f} ± {c['gamma_err']:.2f} MHz (HWHM),  "
                f"area = {row[2]:.1f} ± {row[3]:.1f} mV·ms,  fit rms = {c['rms']:.1f} mV",
                fontsize=10)
    a.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(IND / f"zeeman_19Nov_sigma_fit_B{B:04.1f}G.png", dpi=150)
    plt.close(fig)

# ---------------------------------------------------------------- plot: area vs B
fig, ax = plt.subplots(figsize=(7, 4.8))
k = area.max()
ax.errorbar(B_G, area / k, yerr=area_err / k, fmt="o", color=BLUE, ms=6, elinewidth=1.5,
            capsize=3, mec="white", mew=1)
ax.set_xlabel("Magnetic field (G)")
ax.set_ylabel("Relative intensity (a.u.)")
ax.set_title("$^{87}$Rb D$_2$ F=2→F'=3 (σ$^+$σ$^+$): fitted line area vs field, 19 Nov", fontsize=11)
ax.set_ylim(0, 1.15)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "zeeman_19Nov_sigma_area_vs_B.png", dpi=200)
print(f"\nsaved figures to {FIG}, table to {RES / 'zeeman_19Nov_sigma_area.csv'}")
