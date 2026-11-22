"""Literature Rb D2 (5S1/2 -> 5P3/2) hyperfine transitions and crossovers.

Values from D. A. Steck, "Rubidium 87 D Line Data" and "Rubidium 85 D Line Data":
line centroid, ground-state (F) and excited-state (F') hyperfine shifts, all in MHz.
All frequencies are returned relative to REF = 85Rb F=3 -> F'=4.
"""

LIT = {
    "87": dict(nu0=384_230_484.468, g={2: 2563.005979, 1: -4271.676631},
               e={3: 193.7407, 2: -72.9112, 1: -229.8518, 0: -302.0738}),
    "85": dict(nu0=384_230_406.373, g={3: 1264.888516, 2: -1770.843922},
               e={4: 100.205, 3: -20.435, 2: -83.835, 1: -113.208}),
}
REF = LIT["85"]["nu0"] - LIT["85"]["g"][3] + LIT["85"]["e"][4]   # 85Rb F=3 -> F'=4, MHz


def lines():
    """All dipole-allowed transitions + crossovers as (name, MHz rel. REF, 'real'|'co')."""
    out = []
    for iso, d in LIT.items():
        for F, eg in d["g"].items():
            Fp = [f for f in d["e"] if abs(f - F) <= 1]
            nu = {f: d["nu0"] - eg + d["e"][f] - REF for f in Fp}
            for f in Fp:
                out.append((f"{iso}Rb F={F}→F'={f}", nu[f], "real"))
            for i, a in enumerate(Fp):
                for b in Fp[i + 1:]:
                    out.append((f"{iso}Rb F={F} CO{min(a,b)}{max(a,b)}", (nu[a] + nu[b]) / 2, "co"))
    return out


LINES = lines()

# the four hyperfine groups (ground state), in order of increasing frequency
GROUPS = [("$^{87}$Rb F=2 → F'=1,2,3", "87Rb F=2"), ("$^{85}$Rb F=3 → F'=2,3,4", "85Rb F=3"),
          ("$^{85}$Rb F=2 → F'=1,2,3", "85Rb F=2"), ("$^{87}$Rb F=1 → F'=0,1,2", "87Rb F=1")]


def group_span(key):
    """(min, max) frequency in MHz of all lines/crossovers in one ground-state group."""
    xs = [l[1] for l in LINES if l[0].startswith(key)]
    return min(xs), max(xs)


def mark_lines(ax, key, fontsize=7):
    """Draw literature transitions (red) and crossovers (green dashed) of one group."""
    y1 = ax.get_ylim()[1]
    for name, x, kind in LINES:
        if not name.startswith(key):
            continue
        col = "C3" if kind == "real" else "C2"
        ax.axvline(x, color=col, lw=0.7, ls="-" if kind == "real" else "--", alpha=0.7)
        ax.text(x, y1, name.split(" ", 2)[-1].replace("F=", "").split("→")[-1], rotation=90,
                va="top", ha="right", fontsize=fontsize, color=col)
