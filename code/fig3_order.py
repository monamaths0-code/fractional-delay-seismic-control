"""Figure 3: effect of the fractional order rho on the response (no control, varpi = 0.5 sin 2t)."""
import numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import os
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "numerics.py")).read()
exec(src[:src.index("results = {}")])          # parameters, Solver, plotting style

rhos_fine = np.round(np.arange(1.10, 1.901, 0.05), 2)
rhos_show = [1.2, 1.4, 1.6, 1.8]
curves, peaks, energy = {}, [], []
for r in rhos_fine:
    Pr = dict(P); Pr["rho"] = float(r); s = Solver(Pr)
    with torch.no_grad():
        Y, _ = s.solve(torch.zeros(s.M + 1), torch.tensor(0.5 * np.sin(2.0 * s.t)))
    x = (Y @ s.Phi.T).numpy()
    iw = np.argmin(np.abs(s.S["w"] - 0.5))
    curves[float(r)] = x[:, iw]
    peaks.append(np.abs(x[:, iw]).max())
    nrm = np.sqrt(np.sum(Y.numpy() ** 2, axis=1))
    energy.append(np.sqrt(s.p["h"] * np.sum(nrm ** 2)))
t = s.t
# data behind Figure 3 (CSV)
import csv, os
_d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed"); os.makedirs(_d, exist_ok=True)
with open(os.path.join(_d, "table_order_peaks.csv"), "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["rho", "peak_midpoint_displacement", "L2_energy_norm"])
    for r, pk, en in zip(rhos_fine, peaks, energy): wr.writerow([f"{r:.2f}", f"{pk:.10g}", f"{en:.10g}"])
with open(os.path.join(_d, "order_midpoint_curves.csv"), "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["t"] + [f"xi_mid_rho_{r:.2f}" for r in rhos_fine])
    for k in range(len(t)): wr.writerow([f"{t[k]:.4f}"] + [f"{curves[float(r)][k]:.10g}" for r in rhos_fine])

cols = ["#440154", "#2a788e", "#22a884", "#e8a202"]    # viridis-like, dark purple -> amber = low -> high order
styles = ["-", "--", "-.", (0, (1, 1.2))]
fig = plt.figure(figsize=(7.2, 5.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], hspace=0.45, wspace=0.32)

ax = fig.add_subplot(gs[0, :])
for r, col, ls in zip(rhos_show, cols, styles):
    ax.plot(t, curves[r], color=col, lw=2.0, ls=ls, label=rf"$\varrho={r}$")
    k = np.argmax(np.abs(curves[r]))
    ax.plot(t[k], curves[r][k], "o", ms=6, color=col, mec="white", mew=1.0, zorder=3)
ax.axhline(0, color="#999999", lw=0.8)
ax.set_xlim(0, 4); ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$\xi(t,1/2)$")
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16), fontsize=9, handlelength=3.0)
ax.set_title("(a) midpoint displacement for four orders (dots: peak values)", fontsize=9, pad=24)

ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(rhos_fine, peaks, "-", color="#2a788e", lw=2.0)
ax2.plot(rhos_fine, peaks, "o", ms=5, color="#2a788e", mec="white", mew=1.0)
for r, col in zip(rhos_show, cols):
    i = int(np.argmin(np.abs(rhos_fine - r)))
    ax2.plot(rhos_fine[i], peaks[i], "o", ms=8, color=col, mec="white", mew=1.2, zorder=3)
for r in rhos_show:
    i = int(np.argmin(np.abs(rhos_fine - r)))
    ax2.annotate(f"{peaks[i]:.3f}", (rhos_fine[i], peaks[i]), textcoords="offset points",
                 xytext=(-4, 8), ha="right", fontsize=8, color="#333333")
ax2.set_xlabel(r"order $\varrho$"); ax2.set_ylabel(r"peak $|\xi(t,1/2)|$")
ax2.set_title("(b) peak displacement versus order", fontsize=9)

ax3 = fig.add_subplot(gs[1, 1])
Z = np.array([curves[float(r)] for r in rhos_fine])
lim = np.abs(Z).max()
im = ax3.pcolormesh(t, rhos_fine, Z, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), shading="auto")
cb = fig.colorbar(im, ax=ax3, fraction=0.05, pad=0.03); cb.set_label(r"$\xi(t,1/2)$", fontsize=9)
for r, col, ls in zip(rhos_show, cols, styles):
    ax3.axhline(r, color=col, lw=1.4, ls=ls)
ax3.set_xlabel(r"$t$"); ax3.set_ylabel(r"order $\varrho$"); ax3.grid(False)
ax3.set_title(r"(c) response map over $(t,\varrho)$", fontsize=9)

import os; out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results") + os.sep
fig.savefig(out + "fig3_order.pdf", bbox_inches="tight"); fig.savefig(out + "fig3_order.png", bbox_inches="tight", dpi=200)
print({str(r): round(float(p), 4) for r, p in zip(rhos_fine, peaks)})
