"""Model B (sublevel rate equations, scripts/model_B.py) fitted to the 20 Nov 87Rb F=2 -> F'=3 Zeeman data
(sigma+ pump, sigma- probe, B along the beams).

Spectra: same preprocessing as zeeman_intensity_20Nov.py: PK - Chebyshev-smoothed BG, cut to
t_F3 - 0.30 .. t_F3 + 0.40 ms, frequency delta = 207.5 MHz/ms * (t - t_F3).

Option 1 (model note, Step 10) - multiplet fit of every spectrum:
    f(delta) = A sum_k w_k L(delta - delta0 - nu_k; G) + c0 + c1 x + c2 x^2,   x = delta - delta0
    nu_k(B), w_k(B) fixed by Model B's weak-beam limit (sub-resonances (nu_i + nu_j)/2 of the F'=3 cluster);
    free A, G, delta0, baseline.  Intensity = A pi G / 2.  (The C(2,3) crossover term is left out: its
    centre, -133 MHz, is outside the -62 .. +83 MHz window; its tail is absorbed by the baseline.)
    Uncertainty = fit covariance (sigma = residual rms) (+) spread over baseline orders 1, 2, 3.
Option 2 - full Model B spectrum as template:
    f(delta) = a S_B(delta - delta0; B, s_pump, s_probe = s_pump/9) + quadratic baseline,
    one s_pump shared by all spectra (scan over a log grid + parabolic refinement of the total chi^2),
    a, delta0, baseline per spectrum.  chi^2 uses sigma = white-noise level of that spectrum (rms of data minus
    a 3 MHz running mean, x sqrt(31/30)).
Intensity vs B:
    Option 1 applied to the noiseless Model B spectra gives the model height H_B(B, s), intensity I_B(B, s)
    and width G_B(B, s) (identical extraction for data and model).  Data = K * model, K and s_pump free,
    weighted least squares over B >= B_MIN_FIT (the 0-10 G rise is not in Model B, see model note);
    the full-range chi^2 is also reported.
The sign of B relative to the pump direction is not known: both signs are fitted with Option 2
(-B = swapped sigma+/sigma-); the better one is then used everywhere (Option 1 positions, model curves).

Usage:  python3 scripts/zeeman_modelB_fit_20Nov.py      (~5 min on 10 cores first time; templates cached)
Output: figures/zeeman_20Nov/zeeman_20Nov_modelB_spectral_fits.png,
        figures/zeeman_20Nov/zeeman_20Nov_modelB_intensity_vs_B.png,
        results/zeeman_20Nov_modelB_fit.csv, results/zeeman_20Nov_modelB_log.txt
"""
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_B as mb

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/20th November Data"
CACHE = ROOT / "data/cache/20Nov"
TCACHE = ROOT / "data/cache/modelB"
FIG, RES = ROOT / "figures/zeeman_20Nov", ROOT / "results"
CO_TO_F3, S_SCAN = 266.650 / 2, 207.5     # MHz, MHz/ms
LEFT, RIGHT = -0.30, 0.40                 # window around F'=3 (ms)
G_PER_A, BLOCK = 108.0, 100
B_MIN_FIT = 10.0                          # G, intensity-vs-B fit range
S_GRID = np.geomspace(1, 1000, 13)        # s_pump grid for templates
DELTA = np.arange(-110, 130.01, 0.5)      # template frequency grid (MHz)
BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#222222", "#8a8a85"


# ---------------------------------------------------------------- data (as zeeman_intensity_20Nov.py)
def load(path):
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / (path.stem + ".npy")
    if cache.exists():
        return np.load(cache)
    with open(path, "rb") as f:
        f.readline()
        raw = f.read().replace(b"\r", b"").replace(b"\n", b",")
    d = np.fromstring(raw, sep=",").reshape(-1, 2)
    np.save(cache, d)
    return d


def block_avg(x, n=BLOCK):
    return x[: len(x) // n * n].reshape(-1, n).mean(axis=1)


def recordings():
    out = []
    for p in RAW.glob("M*PK*.CSV"):
        m = re.fullmatch(r"M(\d+)PK(\d*)", p.stem)
        out.append((int(m.group(1)) / 1000.0, m.group(1), m.group(2)))
    return sorted(out, key=lambda r: (r[0], r[2]))


def smooth_peak(t, v, mask):
    vs = np.convolve(v, np.ones(21) / 21, "same")
    i = np.argmax(np.where(mask, vs, -np.inf))
    s = slice(max(i - 40, 0), i + 41)
    c = np.polyfit(t[s] - t[i], vs[s], 2)
    return t[i] - c[1] / (2 * c[0])


def spectra():
    out = []
    for I, tag, suffix in recordings():
        pk, bg = load(RAW / f"M{tag}PK{suffix}.CSV"), load(RAW / f"M{tag}BG{suffix}.CSV")
        t, v, vb = block_avg(pk[:, 0]) * 1e3, block_avg(pk[:, 1]), block_avg(bg[:, 1])
        xc = (t - t.mean()) / (np.ptp(t) / 2)
        bg_s = np.polynomial.chebyshev.chebval(xc, np.polynomial.chebyshev.chebfit(xc, vb, 8))
        edge = np.zeros_like(t, bool)
        edge[50:-50] = True
        if np.ptp(t) > 1.0:
            t_co = smooth_peak(t, v, edge)
            t3 = smooth_peak(t, v, edge & (np.abs(t - t_co - CO_TO_F3 / S_SCAN) < 0.1))
        else:
            t3 = smooth_peak(t, v, edge)
        win = (t - t3 >= LEFT) & (t - t3 <= RIGHT)
        out.append(dict(I=I, B=I * G_PER_A, repeat=bool(suffix), delta=(t - t3)[win] * S_SCAN,
                        d=(v - bg_s)[win] * 1e3))
    return out


# ---------------------------------------------------------------- Option 1: multiplet fit
def lor_sum(delta, pos, wt, G):
    h = G / 2
    return (wt[None] * h**2 / ((delta[:, None] - pos[None]) ** 2 + h**2)).sum(axis=1)


def fit_option1(delta, d, B, order=2, sigma=None):
    pos, wt = mb.multiplet(float(B))
    i = np.argmax(np.convolve(d, np.ones(31) / 31, "same"))
    A0 = d[i] - d.min()

    def model(p):
        A, G, d0 = p[:3]
        x = delta - d0
        return A * lor_sum(delta - d0, pos, wt, G) + np.polyval(p[3:][::-1], x)

    p0 = [A0, 20.0, delta[i] - np.dot(pos, wt), d.min()] + [0.0] * order
    lo = [0, 2.0, -40] + [-np.inf] * (order + 1)
    hi = [5 * A0 + 1e-9, 150.0, 40] + [np.inf] * (order + 1)
    r = least_squares(lambda p: model(p) - d, p0, bounds=(lo, hi), x_scale="jac")
    A, G, d0 = r.x[:3]
    rms = r.fun.std()
    s = rms if sigma is None else sigma
    J = r.jac
    try:
        cov = np.linalg.inv(J.T @ J) * s**2
    except np.linalg.LinAlgError:
        cov = np.full((len(r.x), len(r.x)), np.nan)
    inten = A * np.pi * G / 2
    gI = np.zeros(len(r.x))
    gI[0], gI[1] = np.pi * G / 2, np.pi * A / 2
    xx = np.linspace(-60, 80, 2801)
    height = A * lor_sum(xx, pos, wt, G).max()
    return dict(p=r.x, A=A, G=G, d0=d0, I=inten, dI=np.sqrt(gI @ cov @ gI), dG=np.sqrt(cov[1, 1]),
                H=height, dH=height * np.sqrt(cov[0, 0]) / A, rms=rms, fit=model(r.x),
                base=np.polyval(r.x[3:][::-1], delta - d0), pos=pos, wt=wt)


# ---------------------------------------------------------------- Model B templates (cached)
def template(args):
    B, s = args
    TCACHE.mkdir(parents=True, exist_ok=True)
    f = TCACHE / f"S_B{B:+.3f}_s{s:.4g}.npy"
    if f.exists():
        return np.load(f)
    S = mb.spectrum(B, DELTA, s)
    np.save(f, S)
    return S


def templates(Bs, ss, pool):
    jobs = [(float(B), float(s)) for s in ss for B in Bs]
    out = pool.map(template, jobs)
    return {j: o for j, o in zip(jobs, out)}


# ---------------------------------------------------------------- Option 2: template fit
def fit_option2(delta, d, S, sigma, order=2):
    def lin(d0):
        x = delta - d0
        T = np.interp(delta - d0, DELTA, S)
        X = np.column_stack([T] + [x**k for k in range(order + 1)])
        c, *_ = np.linalg.lstsq(X, d, rcond=None)
        if c[0] < 0:
            c[0] = 0
            c[1:], *_ = np.linalg.lstsq(X[:, 1:], d, rcond=None)
        return c, X @ c

    def chi2(d0):
        return (((lin(d0)[1] - d) / sigma) ** 2).sum()

    grid = np.arange(-30, 30.01, 1.0)
    k = np.argmin([chi2(g) for g in grid])
    r = minimize_scalar(chi2, bounds=(grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]), method="bounded")
    c, fit = lin(r.x)
    return dict(a=c[0], d0=r.x, chi2=r.fun, fit=fit, base=fit - c[0] * np.interp(delta - r.x, DELTA, S))


# ---------------------------------------------------------------- helpers
def parabola_min(x, y):
    """minimum of a parabola through the lowest grid point and its neighbours (x on log scale)."""
    k = int(np.clip(np.argmin(y), 1, len(y) - 2))
    c = np.polyfit(x[k - 1:k + 2], y[k - 1:k + 2], 2)
    if c[0] <= 0:
        return x[np.argmin(y)], y.min(), np.nan
    xm = -c[1] / (2 * c[0])
    return xm, np.polyval(c, xm), 1 / np.sqrt(c[0])      # 1-sigma from Delta chi^2 = 1


def model_curves(Bs, s, sign, tdict):
    """Option 1 applied to the noiseless Model B spectra -> H, I, G arrays over Bs."""
    H, Iv, G = [], [], []
    for B in Bs:
        S = tdict[(float(sign * B), float(s))]
        k = np.argmax(S)
        win = (DELTA - DELTA[k] >= LEFT * S_SCAN) & (DELTA - DELTA[k] <= RIGHT * S_SCAN)
        r = fit_option1(DELTA[win] - DELTA[k], S[win], sign * B)
        H.append(r["H"]), Iv.append(r["I"]), G.append(r["G"])
    return np.array(H), np.array(Iv), np.array(G)


def scale_fit(y, dy, m, mask):
    """best K for y = K m (weighted) on mask; returns K, chi2(mask), chi2(all)."""
    w = 1 / dy**2
    K = np.sum(w[mask] * y[mask] * m[mask]) / np.sum(w[mask] * m[mask] ** 2)
    r2 = ((y - K * m) / dy) ** 2
    return K, r2[mask].sum(), r2.sum()


# ---------------------------------------------------------------- main
def main():
    log = []

    def say(s=""):
        print(s)
        log.append(s)

    data = spectra()
    B_all = np.array([r["B"] for r in data])
    Bu = np.unique(B_all)
    # white-noise level per spectrum (3 MHz running mean removes the line shape)
    noise = []
    for r in data:
        k = max(int(round(3.0 / np.median(np.diff(r["delta"])))), 3)
        sm = np.convolve(r["d"], np.ones(k) / k, "same")
        noise.append((r["d"] - sm)[k:-k].std() * np.sqrt(k / (k - 1)))
    noise = np.array(noise)
    sig = noise
    say("noise sigma per spectrum (mV): " + " ".join(f"{n_:.1f}" for n_ in noise))

    # ---- Model B templates over the s grid, both signs
    with Pool(10) as pool:
        tdict = templates(np.concatenate([Bu, -Bu]), S_GRID, pool)

        # ---- Option 2: shared s
        chi2_o2 = {sg: np.array([[fit_option2(r["delta"], r["d"], tdict[(float(sg * r["B"]), float(s))], sg_)["chi2"]
                                  for r, sg_ in zip(data, sig)] for s in S_GRID]) for sg in (+1, -1)}
        tot = {sg: chi2_o2[sg].sum(axis=1) for sg in (+1, -1)}
        sign2 = +1 if tot[+1].min() <= tot[-1].min() else -1
        ls, chi_min, dls = parabola_min(np.log10(S_GRID), tot[sign2])
        s_o2 = 10**ls
        say(f"\nOption 2 (full Model B template, shared s_pump, s_probe = s_pump/9):")
        say("  s_pump grid : " + " ".join(f"{s:7.3g}" for s in S_GRID))
        for sg in (+1, -1):
            say(f"  chi2 ({'+' if sg > 0 else '-'}B)   : " + " ".join(f"{c:7.0f}" for c in tot[sg]))
        ndof2 = sum(len(r["d"]) for r in data) - len(data) * 5 - 1
        say(f"  best: {'+' if sign2 > 0 else '-'}B, s_pump = {s_o2:.1f} (log10 s = {ls:.3f} ± {dls:.3f} stat), "
            f"chi2 = {chi_min:.0f} / {ndof2} dof (chi2_red = {chi_min/ndof2:.1f})")
        best_ind = [10 ** parabola_min(np.log10(S_GRID), chi2_o2[sign2][:, i])[0] for i in range(len(data))]
        tfin = templates(sign2 * Bu, [s_o2], pool)
        tdict.update(tfin)
    fits2 = [fit_option2(r["delta"], r["d"], tdict[(float(sign2 * r["B"]), float(s_o2))], sg_)
             for r, sg_ in zip(data, sig)]
    say("  per spectrum (global s):   B (G)   a        d0 (MHz)  chi2_red   best own s_pump")
    for f, r, sb in zip(fits2, data, best_ind):
        say(f"                           {r['B']:6.2f}  {f['a']:8.0f}  {f['d0']:+6.1f}    "
            f"{f['chi2']/(len(r['d'])-5):5.2f}      {sb:7.1f}")

    # ---- Option 1 on data (field sign from Option 2)
    o1 = {sg: [fit_option1(r["delta"], r["d"], sg * r["B"]) for r in data] for sg in (+1, -1)}
    chi_sign = {sg: sum((((f["fit"] - r["d"]) / n_) ** 2).sum() for f, r, n_ in zip(o1[sg], data, noise))
                for sg in (+1, -1)}
    say(f"\nOption 1 total chi2 (sigma = noise): +B {chi_sign[+1]:.0f}, -B {chi_sign[-1]:.0f}"
        f"  -> using {'+' if sign2 > 0 else '-'}B from Option 2")
    sign = sign2
    fits1 = o1[sign]
    # uncertainties: covariance (+) spread over baseline orders
    for f, r in zip(fits1, data):
        alt = [fit_option1(r["delta"], r["d"], sign * r["B"], order=o) for o in (1, 3)]
        f["dI"] = np.hypot(f["dI"], np.std([f["I"]] + [a["I"] for a in alt], ddof=1))
        f["dH"] = np.hypot(f["dH"], np.std([f["H"]] + [a["H"] for a in alt], ddof=1))
        f["dG"] = np.hypot(f["dG"], np.std([f["G"]] + [a["G"] for a in alt], ddof=1))
    I1 = np.array([f["I"] for f in fits1])
    dI1 = np.array([f["dI"] for f in fits1])
    H1 = np.array([f["H"] for f in fits1])
    dH1 = np.array([f["dH"] for f in fits1])
    G1 = np.array([f["G"] for f in fits1])
    dG1 = np.array([f["dG"] for f in fits1])
    say("\nOption 1 (multiplet, weak-beam positions/weights) per spectrum:")
    say("   B (G)   G (MHz)     d0 (MHz)  height (mV)       intensity A*pi*G/2 (mV MHz)   rms (mV)")
    for f, r in zip(fits1, data):
        say(f"  {r['B']:6.2f}  {f['G']:5.1f}±{f['dG']:4.1f}  {f['d0']:+6.1f}    {f['H']:6.1f}±{f['dH']:5.1f}"
            f"      {f['I']:8.0f}±{f['dI']:5.0f}                {f['rms']:.1f}")
    say(f"  -> F'=3 width at 0 G: G = {G1[0]:.1f} MHz "
        f"({'Option 1 regime (< 25 MHz)' if G1[0] < 25 else 'strong saturation: Option 2 regime'})")

    # ---- intensity vs B: K * Model B (Option 1 extraction on model spectra), s free
    mask = B_all >= B_MIN_FIT
    curves = {s: model_curves(B_all, s, sign2, tdict) for s in S_GRID}
    res_vsB = {}
    say(f"\nIntensity vs B: data = K * Model B, fitted for B >= {B_MIN_FIT:.0f} G "
        f"({mask.sum()} points), sign {'+' if sign2 > 0 else '-'}B")
    for name, y, dy, idx in [("height", H1, dH1, 0), ("intensity", I1, dI1, 1)]:
        c2 = np.array([scale_fit(y, dy, curves[s][idx], mask)[1] for s in S_GRID])
        ls_b, c2min, dls_b = parabola_min(np.log10(S_GRID), c2)
        res_vsB[name] = dict(chi2=c2, s=10**ls_b, dls=dls_b)
        say(f"  {name:9s}: chi2 vs s = " + " ".join(f"{c:6.1f}" for c in c2))
        say(f"             best s_pump = {10**ls_b:.1f} (log10 s = {ls_b:.3f} ± {dls_b:.3f})")
    with Pool(10) as pool:
        tdict.update(templates(sign2 * Bu, [res_vsB["height"]["s"], res_vsB["intensity"]["s"]], pool))
    final = {}
    for name, y, dy, idx in [("height", H1, dH1, 0), ("intensity", I1, dI1, 1)]:
        s_b = res_vsB[name]["s"]
        m_b = model_curves(B_all, s_b, sign2, tdict)
        K, c2fit, c2all = scale_fit(y, dy, m_b[idx], mask)
        m_o2 = model_curves(B_all, s_o2, sign2, tdict)
        K2, c2fit2, c2all2 = scale_fit(y, dy, m_o2[idx], mask)
        final[name] = dict(s=s_b, K=K, model=m_b, K2=K2, model2=m_o2)
        say(f"  {name:9s}: s_pump = {s_b:.1f}, chi2 = {c2fit:.1f} / {mask.sum() - 2} dof (B >= {B_MIN_FIT:.0f} G), "
            f"{c2all:.1f} over all {len(y)} points;  with s from spectra ({s_o2:.0f}): chi2 = {c2fit2:.1f} "
            f"/ {mask.sum() - 1} dof, {c2all2:.1f} all")

    # ---- save table + log
    RES.mkdir(exist_ok=True)
    rows = np.array([[r["B"], float(r["repeat"]), f["G"], f["dG"], f["d0"], f["H"], f["dH"], f["I"], f["dI"], f["rms"],
                      g["a"], g["d0"], g["chi2"] / (len(r["d"]) - 5), sb]
                     for r, f, g, sb in zip(data, fits1, fits2, best_ind)])
    hdr = (f"Model B fits, 20 Nov 87Rb F=2->F'=3, sigma+ pump / sigma- probe, field sign {'+' if sign2 > 0 else '-'}B.\n"
           "Option 1: multiplet fit (weak-beam positions/weights), intensity = A*pi*G/2. "
           f"Option 2: a * Model B template, shared s_pump = {s_o2:.2f}.\n"
           "B_G,repeat,G_MHz,G_err,delta0_MHz,height_mV,height_err,intensity_mV_MHz,intensity_err,rms_mV,"
           "opt2_a,opt2_delta0_MHz,opt2_chi2_red,opt2_best_own_s_pump")
    np.savetxt(RES / "zeeman_20Nov_modelB_fit.csv", rows, delimiter=",", header=hdr, fmt="%.6g")
    (RES / "zeeman_20Nov_modelB_log.txt").write_text("\n".join(log) + "\n")

    # ---- figure 1: spectral fits
    n, ncol = len(data), 5
    fig, axs = plt.subplots(int(np.ceil(n / ncol)), ncol, figsize=(20, 3.3 * np.ceil(n / ncol)))
    def draw_fit(a, r, f, g, extras=True):
        """extras=False leaves out the sub-resonance sticks and offset residual (for a log y axis)."""
        x, d = r["delta"], r["d"]
        a.plot(x, d, lw=0.5, color=MUTED, label="PK − BG")
        a.plot(x, f["fit"], lw=1.3, color=BLUE, label="Option 1: multiplet")
        a.plot(x, g["fit"], lw=1.1, color=ORANGE, ls="--", label=f"Option 2: Model B, s={s_o2:.0f}")
        a.plot(x, f["base"], lw=0.8, color=BLUE, ls=":", label="Option 1 baseline")
        h = f["H"]
        if extras:
            for p_, w_ in zip(f["pos"] + f["d0"], f["wt"]):
                a.vlines(p_, f["base"].min() - 0.12 * h, f["base"].min() - 0.12 * h + 0.25 * h * w_ / f["wt"].max(),
                         color=AQUA, lw=1)
            a.plot(x, d - f["fit"] - 0.3 * h + f["base"].min(), lw=0.4, color=INK)
        a.set_title(f"B = {r['B']:.1f} G" + (" (repeat)" if r["repeat"] else "") + f",  G = {f['G']:.0f} MHz,"
                    f"  χ²$_r$(opt2) = {g['chi2'] / (len(x) - 5):.0f}",
                    fontsize=10)
        a.set_xlabel("δ − δ$_{peak}$ (MHz)")
        a.set_ylabel("mV")
        a.grid(alpha=0.25)

    for a, r, f, g in zip(axs.flat, data, fits1, fits2):
        draw_fit(a, r, f, g)
    axs.flat[0].legend(fontsize=7, loc="upper left")
    for a in list(axs.flat)[n:]:
        a.axis("off")
    fig.suptitle("20 Nov, $^{87}$Rb F=2→F'=3, σ⁺ pump / σ⁻ probe: Model B fits.  Blue = Option 1 multiplet "
                 "(green sticks = sub-resonance positions/weights), orange dashed = Option 2 full Model B "
                 "template (shared s), black = Option 1 residual (offset)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(FIG / "zeeman_20Nov_modelB_spectral_fits.png", dpi=150)

    # ---- figure 1b: each spectral fit on its own, fitted pedestal (quadratic baseline) subtracted so the
    #      spectrum sits on a flat zero line (same baseline removed from data, Option 1 and Option 2); symlog y axis (linear within +-LINTHRESH, log above);
    #      residual + sub-resonance sticks in a linear panel below
    single = FIG / "fits_modelB"
    single.mkdir(exist_ok=True)
    LINTHRESH = 5.0                                   # mV, ~3x the per-spectrum noise
    for k, (r, f, g) in enumerate(zip(data, fits1, fits2)):
        x, d = r["delta"], r["d"]
        fig, (a, b) = plt.subplots(2, 1, figsize=(7, 6.2), sharex=True, gridspec_kw=dict(height_ratios=(3, 1)))
        a.plot(x, d - f["base"], lw=0.5, color=MUTED, label="PK − BG − baseline")
        a.plot(x, f["fit"] - f["base"], lw=1.3, color=BLUE, label="Option 1: multiplet")
        a.plot(x, g["fit"] - f["base"], lw=1.1, color=ORANGE, ls="--", label=f"Option 2: Model B, s={s_o2:.0f}")
        a.axhline(0, lw=0.8, color=INK, ls=":")
        a.set_yscale("symlog", linthresh=LINTHRESH, linscale=0.5)
        top = 1.5 * (d - f["base"]).max()
        a.set_ylim(-3 * LINTHRESH, top)
        a.set_yticks([t_ for t_ in (-10, 0, 10, 100, 1000) if t_ < top])
        a.set_yticklabels([f"{t_:g}" for t_ in (-10, 0, 10, 100, 1000) if t_ < top])
        a.set_ylabel("mV (symlog)")
        a.grid(alpha=0.25, which="both")
        a.set_title(f"B = {r['B']:.1f} G" + (" (repeat)" if r["repeat"] else "") + f",  G = {f['G']:.0f} MHz,"
                    f"  χ²$_r$(opt2) = {g['chi2'] / (len(x) - 5):.0f}\n"
                    f"Option 1 quadratic baseline subtracted from all curves; linear within ±{LINTHRESH:.0f} mV, log above",
                    fontsize=10)
        a.legend(fontsize=8, loc="upper right")
        b.plot(x, d - f["fit"], lw=0.5, color=INK)
        b.axhline(0, lw=0.6, color=MUTED)
        lo = (d - f["fit"]).min()
        for p_, w_ in zip(f["pos"] + f["d0"], f["wt"]):
            b.vlines(p_, lo, lo + 0.5 * abs(lo) * w_ / f["wt"].max(), color=AQUA, lw=1.2)
        b.set_xlabel("δ − δ$_{peak}$ (MHz)")
        b.set_ylabel("resid. (mV)")
        b.set_title("Option 1 residual (black), sub-resonance positions/weights (green)", fontsize=9)
        b.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(single / (f"{k:02d}_B{r['B']:04.1f}G" + ("_repeat" if r["repeat"] else "") + ".png"), dpi=150)
        plt.close(fig)

    # ---- figure 2: intensity vs B
    Bf = np.linspace(0, 56, 300)
    fig, axs = plt.subplots(2, 2, figsize=(13, 9.5))
    for a, name, y, dy, idx, title in [(axs[0, 0], "height", H1, dH1, 0, "(a) Peak height (Option 1 fit)"),
                                       (axs[0, 1], "intensity", I1, dI1, 1, "(b) Intensity A·πG/2 (Option 1 fit)")]:
        fr = final[name]
        nrm = y.max()
        m = ~np.array([r["repeat"] for r in data])
        a.errorbar(B_all[m], y[m] / nrm, yerr=dy[m] / nrm, fmt="o", color=BLUE, ms=5, capsize=3, mec="white",
                   label="data")
        a.errorbar(B_all[~m], y[~m] / nrm, yerr=dy[~m] / nrm, fmt="D", color=BLUE, mfc="white", ms=5, capsize=3,
                   label="repeat (10.2 G)")
        o = np.argsort(B_all)
        mod = fr["K"] * fr["model"][idx] / nrm
        mod2 = fr["K2"] * fr["model2"][idx] / nrm
        fit_r = B_all[o] >= B_MIN_FIT
        a.plot(B_all[o][fit_r], mod[o][fit_r], color=ORANGE, lw=1.8,
               label=f"Model B fit, s$_{{pump}}$ = {fr['s']:.0f}")
        a.plot(B_all[o], mod[o], color=ORANGE, lw=1, ls=":")
        a.plot(B_all[o], mod2[o], color=AQUA, lw=1.3, ls="--",
               label=f"Model B, s$_{{pump}}$ = {s_o2:.0f} (from spectra), K fitted")
        a.axvspan(0, B_MIN_FIT, color=MUTED, alpha=0.12, lw=0)
        a.set_xlabel("Magnetic field (G)")
        a.set_ylabel("Relative intensity (a.u.)")
        a.set_title(title, fontsize=11)
        a.set_ylim(0, 1.15)
        a.grid(alpha=0.25)
        a.legend(fontsize=8, loc="lower right")
    a = axs[1, 0]
    for c2, col, mk, lab in [(res_vsB["height"]["chi2"], BLUE, "o", "height vs B"),
                             (res_vsB["intensity"]["chi2"], ORANGE, "o", "intensity vs B"),
                             (tot[sign2], AQUA, "s", "all spectra (Option 2)")]:
        a.loglog(S_GRID, c2 / c2.min(), marker=mk, color=col, ms=4, lw=1, label=lab)
    a.set_xlabel("s$_{pump}$ (s$_{probe}$ = s$_{pump}$/9)")
    a.set_ylabel("χ² / χ²$_{min}$")
    a.set_title("(c) χ² against s$_{pump}$ (each normalised to its minimum)", fontsize=11)
    a.grid(alpha=0.25, which="both")
    a.legend(fontsize=8)
    a = axs[1, 1]
    a.errorbar(B_all, G1, yerr=dG1, fmt="o", color=BLUE, ms=5, capsize=3, mec="white", label="data (Option 1 G)")
    o = np.argsort(B_all)
    for s_, col, ls_ in [(final["height"]["s"], ORANGE, "-"), (s_o2, AQUA, "--")]:
        a.plot(B_all[o], model_curves(B_all, s_, sign2, tdict)[2][o], color=col, ls=ls_, lw=1.3,
               label=f"Model B spectra, s$_{{pump}}$ = {s_:.0f}")
    for s_ in (1.0, 10.0, 100.0):
        a.plot(B_all[o], curves[S_GRID[np.argmin(abs(S_GRID - s_))]][2][o], color=MUTED, lw=0.8, ls=":")
        a.annotate(f"s={s_:.0f}", (B_all[o][-1], curves[S_GRID[np.argmin(abs(S_GRID - s_))]][2][o][-1]),
                   fontsize=7, color=MUTED, xytext=(3, 0), textcoords="offset points", va="center")
    a.set_xlabel("Magnetic field (G)")
    a.set_ylabel("multiplet width G (MHz)")
    a.set_title("(d) Fitted sub-resonance width G: data vs Model B spectra", fontsize=11)
    a.grid(alpha=0.25)
    a.legend(fontsize=8)
    fig.suptitle("$^{87}$Rb D$_2$ F=2→F'=3, 20 Nov (σ⁺ pump, σ⁻ probe): Model B vs field.  "
                 f"Fit range B ≥ {B_MIN_FIT:.0f} G (grey band excluded; dotted = extrapolation)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "zeeman_20Nov_modelB_intensity_vs_B.png", dpi=200)
    print(f"\nsaved figures to {FIG}, table + log to {RES}")


if __name__ == "__main__":
    main()
