"""Zoomed plot of one Doppler-broadened dip (default: 2nd from left, ~-6 ms, likely 85Rb F=3)."""
import sys
import numpy as np
import matplotlib.pyplot as plt

CACHE = "6th_Nov_Doppler_Free.npy"  # created by plot_doppler_free.py
T_MIN, T_MAX = (float(x) for x in sys.argv[1:3]) if len(sys.argv) >= 3 else (-8.5e-3, -3.5e-3)

data = np.load(CACHE)
t, v = data[:, 0], data[:, 1]
m = (t >= T_MIN) & (t <= T_MAX)
t, v = t[m], v[m]


def block_avg(x, n):
    k = len(x) // n * n
    return x[:k].reshape(-1, n).mean(axis=1)


N = 100  # 0.64 us blocks: keeps narrow Lamb-dip features intact
t_avg, v_avg = block_avg(t, N), block_avg(v, N)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8))

ax1.plot(t[::20] * 1e3, v[::20], lw=0.3, color="0.75", label="Raw (every 20th pt)")
ax1.plot(t_avg * 1e3, v_avg, lw=0.9, color="C0", label=f"{N}-pt block average")
ax1.set_title("Zoom: 2nd dip from left (likely $^{85}$Rb F=3)")
ax1.set_xlabel("Time (ms)")
ax1.set_ylabel("Photodiode voltage, C3 (V)")
ax1.legend()
ax1.grid(alpha=0.3)

# Further zoom on the sub-Doppler features near the bottom of the dip
z0, z1 = -7.0e-3, -5.5e-3
ax1.axvspan(z0 * 1e3, z1 * 1e3, color="C1", alpha=0.1)
mz = (t_avg >= z0) & (t_avg <= z1)
ax2.plot(t_avg[mz] * 1e3, v_avg[mz], lw=0.9, color="C1")
ax2.set_title("Close-up of sub-Doppler (saturated absorption) features")
ax2.set_xlabel("Time (ms)")
ax2.set_ylabel("Photodiode voltage, C3 (V)")
ax2.grid(alpha=0.3)

fig.tight_layout()
out = "doppler_free_6Nov_zoom.png"
fig.savefig(out, dpi=200)
print(f"saved {out}")
plt.show()
