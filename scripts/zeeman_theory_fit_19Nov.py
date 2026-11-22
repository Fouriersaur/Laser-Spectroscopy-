"""Fit the four-level-model intensity-vs-field function to the measured 19 Nov line areas.

Theory (Ummal Momeen et al., J. Phys. B 40 3163 (2007); Hu & Ummal Momeen, IOP Conf. Ser. 263 022003
(2017)):
  * eqs (1)-(4): hyperfine + Zeeman Hamiltonian of 5S1/2 and 5P3/2, diagonalised exactly (done here in
    the |mJ, mI> basis, equivalent to the |F, mF> matrix) -> sublevel energies and mixed eigenstates
  * eqs (5)-(8): field-mixed dipole strengths |<e|d_q|g>|^2 between eigenstates
  * eq (10):     four-level weight of a pump(i)/probe(j) resonance
                     I_ij = |mu_i|^2 |mu_j|^2 ( -delta_{i,sp} + |mu_sp|^2 / Gamma )
                 delta = 1 if pump and probe start from the same ground sublevel (I-, V-type), else 0
                 (Lambda-, N-type); mu_sp = decay from the pumped excited sublevel to the probe's ground
                 sublevel; Gamma = total decay of that excited sublevel (to F=1 and F=2).
  * eqs (9)+(11): each resonance is a Lorentzian gamma/((nu-nubar)^2+gamma^2) of area pi*I_ij*D_ij,
                 D_ij = exp[-((nu_i-nu_j)/(2u/lambda))^2] -> the integrated line area is
                     T(B) = -sum_{ij in F'=3 line} I_ij D_ij
                 (minus sign: I < 0 = reduced absorption = positive peak in PK - BG).
  Pump and probe both sigma+ (q = +1 along B) or both sigma- (q = -1). The two are NOT equivalent once
  the field mixes F' levels (sigma-sigma- at +B = sigma+sigma+ at -B, and the mixing is linear in B),
  so both are computed and fitted.
  "F'=3 line" = resonances whose pump and probe both end on sublevels adiabatically labelled F'=3.

Fit to the areas from zeeman_area_sigma_19Nov.py (run that first), for each polarisation:
  model 1:  area(B) = K * T(B)             (pure scale, the theory's absolute B-dependence)
  model 2:  area(B) = a + K * T(B)         (scale + offset, as the papers compare model and data on
                                            independent axes)
Weighted least squares with the area error bars; repeated without the B = 0 point.
Note: T(B) < 0 at every field (the single-cycle model predicts enhanced absorption on this closed
line), so model 1 needs K < 0; model 2 fits only the shape of T(B), not its sign.

Usage:  python3 scripts/zeeman_theory_fit_19Nov.py
Output: figures/zeeman_19Nov/zeeman_19Nov_sigma_area_theory_fit.png, results/zeeman_19Nov_theory_curve.csv
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sympy.physics.wigner import clebsch_gordan

ROOT = Path(__file__).resolve().parent.parent
FIG, RES = ROOT / "figures", ROOT / "results"
FIG = FIG / "zeeman_19Nov"
FIG.mkdir(parents=True, exist_ok=True)
AREA_CSV = RES / "zeeman_19Nov_sigma_area.csv"
BLUE, ORANGE, AQUA, INK = "#2a78d6", "#eb6834", "#1baf7a", "#222222"

# ---------------------------------------------------------------- 87Rb constants (Steck)
MU_B = 1.39962449361            # MHz/G
I_NUC, G_I = 1.5, -0.0009951414
GROUND = dict(J=0.5, A=3417.341305452145, B=0.0, gJ=2.00233113)
EXCITED = dict(J=1.5, A=84.7185, B=12.4965, gJ=1.3362)
U_OVER_LAMBDA = np.sqrt(2 * 1.380649e-23 * 295 / (86.909180527 * 1.66053907e-27)) / 780.241e-9 / 1e6  # MHz


def jmats(j):
    m = np.arange(j, -j - 1, -1)
    jp = np.zeros((len(m), len(m)))
    for k in range(1, len(m)):
        jp[k - 1, k] = np.sqrt(j * (j + 1) - m[k] * (m[k] + 1))
    return np.diag(m), jp, jp.T, m


def manifold(level, B):
    """Eigen-energies (MHz), eigenvectors (columns, basis |mJ>x|mI>), F label, mF of H_hfs + H_Z."""
    J = level["J"]
    jz, jp, jm, mJ = jmats(J)
    iz, ip, im, mI = jmats(I_NUC)
    eJ, eI = np.eye(len(mJ)), np.eye(len(mI))
    Jz, Jp, Jm = (np.kron(a, eI) for a in (jz, jp, jm))
    Iz, Ip, Im = (np.kron(eJ, a) for a in (iz, ip, im))
    IJ = Iz @ Jz + 0.5 * (Ip @ Jm + Im @ Jp)
    H = level["A"] * IJ
    if level["B"]:
        n = len(mJ) * len(mI)
        H = H + level["B"] * (3 * IJ @ IJ + 1.5 * IJ - I_NUC * (I_NUC + 1) * J * (J + 1) * np.eye(n)) \
            / (2 * I_NUC * (2 * I_NUC - 1) * J * (2 * J - 1))
    H = H + MU_B * B * (level["gJ"] * Jz + G_I * Iz)
    E, V = np.linalg.eigh(H)
    Fz, Fp, Fm = Jz + Iz, Jp + Ip, Jm + Im
    F2 = Fz @ Fz + 0.5 * (Fp @ Fm + Fm @ Fp)
    f2 = np.einsum("ik,ij,jk->k", V, F2, V)
    F = np.rint(-0.5 + np.sqrt(0.25 + f2))           # adiabatic F label (weak field)
    mF = np.rint(2 * np.einsum("ik,ij,jk->k", V, Fz, V)) / 2
    return E, V, F, mF, [(a, b) for a in mJ for b in mI]


_D = {}


def dipole_basis():
    """<J' mJ' mI'| d_q |J mJ mI> = delta(mI,mI') CG(J mJ; 1 q | J' mJ'), array [q+1, e, g]."""
    if "D" not in _D:
        _, _, _, _, bg = manifold(GROUND, 1.0)
        _, _, _, _, be = manifold(EXCITED, 1.0)
        D = np.zeros((3, len(be), len(bg)))
        for qi, q in enumerate((-1, 0, 1)):
            for a, (mJp, mIp) in enumerate(be):
                for b, (mJ, mI) in enumerate(bg):
                    if mIp == mI and abs(mJ + q - mJp) < 1e-9:
                        D[qi, a, b] = float(clebsch_gordan(GROUND["J"], 1, EXCITED["J"], mJ, q, mJp))
        _D["D"] = D
    return _D["D"]


def theory_area(B, q_pump=+1, q_probe=+1):
    """T(B) = -sum I_ij D_ij over the F=2 -> F'=3 resonances (positive = reduced absorption)."""
    B = max(B, 1e-3)                                  # lift degeneracy so eigenstates have good mF
    Eg, Vg, Fg, _, _ = manifold(GROUND, B)
    Ee, Ve, Fe, _, _ = manifold(EXCITED, B)
    S = np.einsum("ae,qab,bg->qeg", Ve, dipole_basis(), Vg) ** 2
    S /= S.sum(axis=(0, 2)).max()                     # excited-state total decay strength = 1
    gamma_tot = S.sum(axis=(0, 2))                    # Gamma of each excited sublevel (F=1 and F=2)
    sp = S.sum(axis=0)                                # |mu_sp|^2 [e, g]
    g2, e3 = np.where(Fg == 2)[0], np.where(Fe == 3)[0]
    pump = [(g, e, S[q_pump + 1, e, g], Ee[e] - Eg[g]) for g in g2 for e in e3 if S[q_pump + 1, e, g] > 1e-12]
    probe = [(g, e, S[q_probe + 1, e, g], Ee[e] - Eg[g]) for g in g2 for e in e3 if S[q_probe + 1, e, g] > 1e-12]
    total = 0.0
    for ga, ea, si, ni in pump:
        for gb, eb, sj, nj in probe:
            I = si * sj * (-float(ga == gb) + sp[ea, gb] / gamma_tot[ea])
            total += I * np.exp(-((ni - nj) / (2 * U_OVER_LAMBDA)) ** 2)
    return -total


# ---------------------------------------------------------------- theory curves
B_grid = np.linspace(0, 56, 113)
CONFIGS = {"σ⁺σ⁺": (+1, +1), "σ⁻σ⁻": (-1, -1)}
T_grid = {name: np.array([theory_area(b, *q) for b in B_grid]) for name, q in CONFIGS.items()}
print("four-level theory T(B), F=2->F'=3 (+ = reduced absorption, - = enhanced absorption):")
for b in (0, 10, 20, 30, 40, 50, 55):
    print(f"  B = {b:4.0f} G   " + "   ".join(
        f"{n}: {np.interp(b, B_grid, T):+.5f}" for n, T in T_grid.items()))

# ---------------------------------------------------------------- data
if not AREA_CSV.exists():
    raise SystemExit(f"{AREA_CSV} not found - run scripts/zeeman_area_sigma_19Nov.py first")
dat = np.loadtxt(AREA_CSV, delimiter=",")
B, A, dA = dat[:, 1], dat[:, 2], dat[:, 3]


def report(name, f, p0, Bs, As, dAs):
    p, cov = curve_fit(f, Bs, As, p0=p0, sigma=dAs, absolute_sigma=True, maxfev=20000)
    chi2 = np.sum(((As - f(Bs, *p)) / dAs) ** 2)
    dof = len(Bs) - len(p)
    print(f"{name}: params = {np.round(p, 2)} ± {np.round(np.sqrt(np.diag(cov)), 2)}   "
          f"chi2/dof = {chi2:.1f}/{dof} = {chi2/dof:.2f}")
    return p, chi2, dof


fits = {}
for name, T in T_grid.items():
    T_of = (lambda TT: lambda b: np.interp(b, B_grid, TT))(T)
    scale = (lambda f: lambda b, K: K * f(b))(T_of)
    affine = (lambda f: lambda b, a, K: a + K * f(b))(T_of)
    K0 = A.mean() / T_of(B).mean()
    print(f"\n{name}  (area in mV ms)")
    print(" all 11 points:")
    fits[(name, "scale")] = (scale, *report("  model 1  area = K T(B)    ", scale, [K0], B, A, dA))
    fits[(name, "affine")] = (affine, *report("  model 2  area = a + K T(B)", affine, [A.mean(), K0], B, A, dA))
    m = B > 0
    print(" without B = 0:")
    fits[(name, "scale", "noB0")] = (scale, *report("  model 1  area = K T(B)    ", scale, [K0],
                                                    B[m], A[m], dA[m]))
    fits[(name, "affine", "noB0")] = (affine, *report("  model 2  area = a + K T(B)", affine,
                                                      [A.mean(), K0], B[m], A[m], dA[m]))

np.savetxt(RES / "zeeman_19Nov_theory_curve.csv",
           np.column_stack([B_grid, T_grid["σ⁺σ⁺"], T_grid["σ⁻σ⁻"]]), delimiter=",",
           header="four-level model, 87Rb F=2->F'=3: T(B) = -sum I_ij D_ij "
                  "(+ = reduced absorption)\nB_G,T_sigma_plus,T_sigma_minus", fmt="%.6g")

# ---------------------------------------------------------------- plot
# The B = 0 point is excluded from the plotted fits: it was recorded under different conditions
# (different trigger window and background level) and no smooth model passes through it and the rest.
k = A.max()
m = B > 0
fig, ax = plt.subplots(figsize=(8, 5.2))
ax.errorbar(B[m], A[m] / k, yerr=dA[m] / k, fmt="o", color=BLUE, ms=6, elinewidth=1.5, capsize=3,
            mec="white", mew=1, label="measured line area", zorder=3)
ax.errorbar(B[~m], A[~m] / k, yerr=dA[~m] / k, fmt="s", color=BLUE, mfc="white", ms=6,
            elinewidth=1.5, capsize=3, label="B = 0 (excluded from fits)", zorder=3)
for key, col in [(("σ⁺σ⁺", "affine", "noB0"), ORANGE), (("σ⁻σ⁻", "affine", "noB0"), AQUA)]:
    f, p, chi2, dof = fits[key]
    ax.plot(B_grid, f(B_grid, *p) / k, color=col, lw=2,
            label=f"four-level {key[0]}: a + K·T(B)   (χ²/dof = {chi2:.1f}/{dof})")
ax.set_xlabel("Magnetic field (G)")
ax.set_ylabel("Relative intensity (a.u.)")
ax.set_title("$^{87}$Rb D$_2$ F=2→F'=3: line area vs field with four-level model fits, 19 Nov",
             fontsize=11)
ax.set_ylim(0, 1.15)
ax.grid(alpha=0.25)
ax.legend(fontsize=8.5, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "zeeman_19Nov_sigma_area_theory_fit.png", dpi=200)
print(f"\nsaved {FIG / 'zeeman_19Nov_sigma_area_theory_fit.png'}")
