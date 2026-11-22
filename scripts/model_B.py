"""Model B: sublevel rate-equation model of Zeeman-resolved 87Rb D2 saturated absorption
(project note "Model B: sublevel rate-equation model ...", Nov 22 2026, Steps 1-10).

Steps
 1. H = A I.J + B_q (...) + mu_B B (g_J J_z + g_I I_z) in the uncoupled basis |m_J, m_I>, diagonalised
    numerically -> 5 ground (F=2) + 16 excited (F'=0..3) sublevels, transition frequencies nu_ge(B).
 2. S^q_eg = |<e|d_q|g>|^2, sum_{g in all 8 ground, q} S^q_eg = 1 (cycling line S = 1);
    branching b_eg = sum_q S^q_eg for g in F=2 (decay to F=1 = loss).
 3. velocity group x = kv/2pi: pump (+z, sigma+) detuning  D_p  = delta - nu_ge - x,
                               probe (-z, sigma-) detuning D_pr = delta - nu_ge + x.
 4. X_ge = s_pump/2 S^+_ge L(D_p) + s_probe/2 S^-_ge L(D_pr),  L(D) = (G/G_h) / (1 + 4 D^2/G_h^2),
    G_h = Gamma + Gamma_laser.
 5./6. steady state of  dN_e/dt = sum_g X (N_g - N_e) - N_e - eps N_e
                       dN_g/dt = -sum_e X (N_g - N_e) + sum_e b_eg N_e + eps (1 - N_g),  eps = 1/(Gamma tau)
    (21 x 21 linear system per velocity group and laser frequency; time unit 1/Gamma).
 7. a(v) = sum_ge S^-_ge L(D_pr) (N_g - N_e)
 8. signal(v) = a_probe_alone(v) - a_probe+pump(v)   (balanced detection; + = reduced absorption)
 9. S(delta) = sum_v f(v) signal(v), Maxwell-Boltzmann, ku/2pi = 305 MHz; x grid 0.4 MHz for |x| < 260 MHz,
    3 MHz out to +-960 MHz.
Frequencies in MHz, delta measured from the B = 0 F=2 -> F'=3 line. B > 0 is along +z (pump direction);
B -> -B is the same as swapping sigma+ <-> sigma-.

Also: weak-beam (Model A) sub-resonances of the F'=3 cluster for the Option 1 multiplet fit function:
    positions nu_k = (nu_i + nu_j)/2, weights W_ij = S_i S_j (delta_{g_i g_j} - b_{e_i -> g_j}) (+ S_i S_j delta_{e_i e_j}/Gamma tau)
    over pump transitions i (sigma+) and probe transitions j (sigma-) into F'=3.
"""
from functools import lru_cache

import numpy as np
from sympy.physics.wigner import clebsch_gordan

# ---------------------------------------------------------------- constants (Steck 87Rb D2; model note)
MU_B = 1.39962449361            # MHz/G
I_NUC, G_I = 1.5, -0.0009951414
GROUND = dict(J=0.5, A=3417.341305452145, B=0.0, gJ=2.00233113)
EXCITED = dict(J=1.5, A=84.7185, B=12.4965, gJ=1.3362)
GAMMA = 6.07                    # natural linewidth (MHz)
GAMMA_LASER = 1.0               # DL pro linewidth (MHz)
GAMMA_H = GAMMA + GAMMA_LASER
GAMMA_TAU = 284.0               # natural lifetimes per transit (D = 2 mm)
KU = np.sqrt(2 * 1.380649e-23 * 295 / (86.909180527 * 1.66053907e-27)) / 780.241e-9 / 1e6   # ku/2pi ~ 305 MHz
PROBE_RATIO = 1 / 9             # s_probe = s_pump / 9


# ---------------------------------------------------------------- Step 1: levels
def _jmats(j):
    m = np.arange(j, -j - 1, -1)
    jp = np.zeros((len(m), len(m)))
    for k in range(1, len(m)):
        jp[k - 1, k] = np.sqrt(j * (j + 1) - m[k] * (m[k] + 1))
    return np.diag(m), jp, jp.T, m


def manifold(level, B):
    """eigen-energies (MHz), eigenvectors (columns, basis |mJ> x |mI>), F label (adiabatic), m, basis labels."""
    J = level["J"]
    jz, jp, jm, mJ = _jmats(J)
    iz, ip, im, mI = _jmats(I_NUC)
    eJ, eI = np.eye(len(mJ)), np.eye(len(mI))
    Jz, Jp, Jm = (np.kron(a, eI) for a in (jz, jp, jm))
    Iz, Ip, Im = (np.kron(eJ, a) for a in (iz, ip, im))
    IJ = Iz @ Jz + 0.5 * (Ip @ Jm + Im @ Jp)
    n = len(mJ) * len(mI)
    H = level["A"] * IJ
    if level["B"]:
        H = H + level["B"] * (3 * IJ @ IJ + 1.5 * IJ - I_NUC * (I_NUC + 1) * J * (J + 1) * np.eye(n)) \
            / (2 * I_NUC * (2 * I_NUC - 1) * J * (2 * J - 1))
    Fz = Jz + Iz
    m_tot = np.diag(Fz)
    E, V, F, M = np.zeros(n), np.zeros((n, n)), np.zeros(n), np.zeros(n)
    # diagonalise block by block in m (m is exact); label F by the B = 0 ordering inside each block
    Fvals = np.arange(abs(J - I_NUC), J + I_NUC + 1)
    for m in np.unique(m_tot):
        idx = np.where(np.isclose(m_tot, m))[0]
        Hb = H[np.ix_(idx, idx)] + MU_B * B * (level["gJ"] * Jz + G_I * Iz)[np.ix_(idx, idx)]
        e, v = np.linalg.eigh(Hb)
        Fs = [f for f in Fvals if f >= abs(m)]           # F values present in this m block
        # at B = 0 the energy ordering in F is fixed (A > 0 both levels, here F increasing in energy)
        for k in range(len(idx)):
            col = idx[k]
            E[col], F[col], M[col] = e[k], Fs[k], m
            V[idx, col] = v[:, k]
    return E, V, F, M, [(a, b) for a in mJ for b in mI]


@lru_cache(maxsize=1)
def _dipole_basis():
    _, _, _, _, bg = manifold(GROUND, 0.0)
    _, _, _, _, be = manifold(EXCITED, 0.0)
    D = np.zeros((3, len(be), len(bg)))
    for qi, q in enumerate((-1, 0, 1)):
        for a, (mJp, mIp) in enumerate(be):
            for b, (mJ, mI) in enumerate(bg):
                if mIp == mI and abs(mJ + q - mJp) < 1e-9:
                    D[qi, a, b] = float(clebsch_gordan(GROUND["J"], 1, EXCITED["J"], mJ, q, mJp))
    return D


@lru_cache(maxsize=1)
def _nu0():
    Eg, _, Fg, _, _ = manifold(GROUND, 0.0)
    Ee, _, Fe, _, _ = manifold(EXCITED, 0.0)
    return Ee[Fe == 3].mean() - Eg[Fg == 2].mean()


# ---------------------------------------------------------------- Step 2: strengths
@lru_cache(maxsize=512)
def atom(B):
    """dict: nu (16 x 5, MHz from B=0 F=2->F'=3), Sp / Sm (sigma+ / sigma-, 16 x 5), b (16 x 5),
    Fe, me (16), mg (5)."""
    Eg, Vg, Fg, mg, _ = manifold(GROUND, B)
    Ee, Ve, Fe, me, _ = manifold(EXCITED, B)
    S = np.einsum("ae,qab,bg->qeg", Ve, _dipole_basis(), Vg) ** 2       # q = -1, 0, +1
    S /= S.sum(axis=(0, 2)).max()
    g2 = Fg == 2
    nu = Ee[:, None] - Eg[None, g2] - _nu0()
    S2 = S[:, :, g2]
    return dict(nu=nu, Sp=S2[2], Sm=S2[0], b=S2.sum(axis=0), Fe=Fe, me=me, mg=mg[g2])


def lorentz_rate(D):
    return (GAMMA / GAMMA_H) / (1 + 4 * D**2 / GAMMA_H**2)


# ---------------------------------------------------------------- Steps 5-7: steady state and probe absorption
def probe_absorption(at, Dp, Dpr, s_pump, s_probe):
    """a for arrays of pump / probe detuning offsets: Dp, Dpr shape (n,) -> detuning of transition ge is
    Dp - nu_ge (pump) and Dpr - nu_ge (probe).  Returns a (n,)."""
    nu, Sp, Sm, b = at["nu"], at["Sp"], at["Sm"], at["b"]
    Lpr = lorentz_rate(Dpr[:, None, None] - nu[None])                    # n x 16 x 5
    X = 0.5 * s_probe * Sm[None] * Lpr
    if s_pump:
        X = X + 0.5 * s_pump * Sp[None] * lorentz_rate(Dp[:, None, None] - nu[None])
    eps = 1.0 / GAMMA_TAU
    n = len(Dpr)
    # excited rows are solved exactly:  N_e = sum_g X_ge N_g / D_e,  D_e = sum_g X_ge + 1 + eps
    # -> 5 x 5 system for the ground populations (identical to the full 21 x 21 steady state)
    De = X.sum(axis=2) + 1 + eps                                         # n x 16
    C = (X + b[None]) / De[:, :, None]
    M = np.einsum("neg,neh->ngh", C, X)
    ig = np.arange(5)
    M[:, ig, ig] -= X.sum(axis=1) + eps
    Ng = np.linalg.solve(M, np.full((n, 5, 1), -eps))[..., 0]
    Ne = np.einsum("neg,ng->ne", X, Ng) / De
    return np.einsum("eg,neg,neg->n", Sm, Lpr, Ng[:, None, :] - Ne[:, :, None])


def x_grid():
    """velocity-group Doppler shifts x (MHz) and Maxwell-Boltzmann weights f(x) dx."""
    fine = np.arange(-260, 260 + 1e-9, 0.4)
    coarse = np.arange(263, 960 + 1e-9, 3.0)
    x = np.concatenate([-coarse[::-1], fine, coarse])
    dx = np.gradient(x)
    w = np.exp(-(x / KU) ** 2) / (np.sqrt(np.pi) * KU) * dx
    return x, w


# ---------------------------------------------------------------- Steps 8-9: spectrum
def spectrum(B, delta, s_pump, s_probe=None, chunk=24):
    """Model B balanced-detection signal S(delta) (+ = pump reduces probe absorption)."""
    if s_probe is None:
        s_probe = s_pump * PROBE_RATIO
    at = atom(float(B))
    x, w = x_grid()
    delta = np.asarray(delta, float)
    # probe alone depends only on y = delta + x: tabulate once and interpolate
    y = np.arange(delta.min() + x.min() - 1, delta.max() + x.max() + 1.01, 0.1)
    a0_y = probe_absorption(at, y, y, 0.0, s_probe)
    a0 = np.interp(delta[:, None] + x[None, :], y, a0_y) @ w
    a1 = np.empty_like(delta)
    for i in range(0, len(delta), chunk):
        d = delta[i:i + chunk]
        Dp = (d[:, None] - x[None, :]).ravel()
        Dpr = (d[:, None] + x[None, :]).ravel()
        a1[i:i + chunk] = probe_absorption(at, Dp, Dpr, s_pump, s_probe).reshape(len(d), -1) @ w
    return a0 - a1


# ---------------------------------------------------------------- Option 1: weak-beam sub-resonances of F'=3
@lru_cache(maxsize=512)
def multiplet(B, fprime=3):
    """(positions nu_k [MHz from B=0 line], normalised weights w_k) of the weak-beam (Model A + 1/Gamma tau)
    sub-resonances of the F'=fprime peak for sigma+ pump, sigma- probe."""
    at = atom(float(B))
    nu, Sp, Sm, b, Fe = at["nu"], at["Sp"], at["Sm"], at["b"], at["Fe"]
    pos, wt = [], []
    for ei in np.where(Fe == fprime)[0]:
        for gi in range(5):
            if Sp[ei, gi] < 1e-10:
                continue
            for ej in np.where(Fe == fprime)[0]:
                for gj in range(5):
                    if Sm[ej, gj] < 1e-10:
                        continue
                    W = Sp[ei, gi] * Sm[ej, gj] * (float(gi == gj) - b[ei, gj] + float(ei == ej) / GAMMA_TAU)
                    W *= np.exp(-((nu[ei, gi] - nu[ej, gj]) / (2 * KU)) ** 2)
                    if abs(W) > 1e-12:
                        pos.append(0.5 * (nu[ei, gi] + nu[ej, gj]))
                        wt.append(W)
    pos, wt = np.array(pos), np.array(wt)
    return pos, wt / wt.sum()


def spectrum_A(B, delta, gamma, fprime=3):
    """weak-beam Model A F'=3 cluster: sum_k w_k L(delta - nu_k; FWHM gamma), unit-height Lorentzians."""
    pos, wt = multiplet(B, fprime)
    h = gamma / 2
    return (wt[None] * h**2 / ((np.asarray(delta)[:, None] - pos[None]) ** 2 + h**2)).sum(axis=1)


if __name__ == "__main__":
    import time
    at = atom(0.0)
    print("B=0 checks: sum_g b over F=2 per F' =",
          {int(f): round(at["b"][at["Fe"] == f].sum(axis=1).mean(), 4) for f in (0, 1, 2, 3)})
    print("cycling S(m=2 -> m'=3, sigma+) =", at["Sp"].max().round(4))
    for B in (0.0, 10.0, 30.0, 55.0):
        pos, wt = multiplet(B)
        print(f"B={B:4.0f} G: {len(pos)} sub-resonances, centres {pos.min():6.1f} .. {pos.max():5.1f} MHz")
    d = np.arange(-110, 140.01, 0.5)
    t0 = time.time()
    S = spectrum(10.0, d, 100.0)
    print(f"spectrum: {time.time() - t0:.1f} s, peak {S.max():.3e} at {d[S.argmax()]:.1f} MHz")
    # weak-beam limit vs Model A (scale should be ~ Gamma tau)
    d2 = np.arange(-40, 40.01, 0.25)
    for B in (0.0, 30.0):
        sB = spectrum(B, d2, 1e-4, 1e-4)
        print(f"B={B}: weak-beam S_B peak/area {sB.max():.3e} / {np.trapz(sB, d2):.3e}")
