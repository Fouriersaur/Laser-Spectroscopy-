"""Plot Doppler-free saturated absorption trace (6 Nov) from oscilloscope CSV."""
import sys
import numpy as np
import matplotlib.pyplot as plt

FILE = sys.argv[1] if len(sys.argv) > 1 else "6th_Nov_Doppler_Free.CSV"
CACHE = FILE.rsplit(".", 1)[0] + ".npy"

# 10M rows -> cache as .npy after first parse
try:
    data = np.load(CACHE)
except FileNotFoundError:
    with open(FILE, "rb") as f:
        f.readline()  # header: "in s,C3 in V"
        raw = f.read().replace(b"\r", b"").replace(b"\n", b",")
    data = np.fromstring(raw, sep=",").reshape(-1, 2)
    np.save(CACHE, data)

t, v = data[:, 0], data[:, 1]

# Block-average to reduce noise and point count (boxcar of N samples)
N = 1000
n = len(t) // N * N
t_avg = t[:n].reshape(-1, N).mean(axis=1)
v_avg = v[:n].reshape(-1, N).mean(axis=1)

fig, ax = plt.subplots(figsize=(11, 5))
step = 200
ax.plot(t[::step] * 1e3, v[::step], lw=0.4, color="0.75", label=f"Raw (every {step}th pt)")
ax.plot(t_avg * 1e3, v_avg, lw=1.0, color="C0", label=f"{N}-pt block average")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Photodiode voltage, C3 (V)")
ax.set_title("Doppler-free saturated absorption spectrum of Rb (6 Nov)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
out = "doppler_free_6Nov.png"
fig.savefig(out, dpi=200)
print(f"saved {out}; {len(t)} pts, dt={t[1]-t[0]:.3e} s, span {t[0]*1e3:.2f}..{t[-1]*1e3:.2f} ms")
plt.show()
