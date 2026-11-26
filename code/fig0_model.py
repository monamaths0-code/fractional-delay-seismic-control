"""Figure: the model problem (schematic, spatial profiles, natural frequencies)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False})
a, c, A0 = 0.1, 1.0, 0.05
w = np.linspace(0, 1, 400)
b = np.exp(-((w - 0.5) / 0.1) ** 2)
phi = A0 * np.sin(np.pi * w)
n = np.arange(1, 9); lam = (n * np.pi) ** 2; om = np.sqrt(c ** 2 * lam / (1 + a * lam))
BLUE, AMBER, GREY, RED = "#2a788e", "#e8a202", "#555555", "#b3261e"

fig = plt.figure(figsize=(7.2, 4.9))
gs = fig.add_gridspec(2, 2, height_ratios=[0.95, 1], hspace=0.38, wspace=0.32)

# (a) schematic
ax = fig.add_subplot(gs[0, :]); ax.set_xlim(-0.12, 1.12); ax.set_ylim(-0.56, 0.62); ax.axis("off")
ax.fill_between([-0.12, 1.12], -0.55, -0.42, color="#d9d9d9", lw=0)
ax.plot([-0.12, 1.12], [-0.42, -0.42], color=GREY, lw=1)
for x0 in [-0.04, 1.0]:
    ax.add_patch(Rectangle((x0, -0.42), 0.04, 0.42, color=GREY))
defl = 0.22 * np.sin(np.pi * w)
ax.plot(w, 0 * w, color="#bbbbbb", lw=1.2, ls="--")
ax.plot(w, defl, color=BLUE, lw=3)
ax.annotate(r"$\xi(t,w)$", (0.25, 0.22 * np.sin(np.pi * 0.25)), xytext=(0.12, 0.42),
            arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8), fontsize=10, color=BLUE)
ax.add_patch(FancyArrowPatch((0.5, 0.58), (0.5, 0.26), arrowstyle="-|>", mutation_scale=14, color=AMBER, lw=2))
ax.text(0.53, 0.45, r"actuator $b(w)\,\mu(t-\kappa)$", color="#8a5d00", fontsize=10)
ax.add_patch(FancyArrowPatch((0.22, -0.49), (0.40, -0.49), arrowstyle="<|-|>", mutation_scale=12, color=RED, lw=1.6))
ax.text(0.43, -0.52, r"ground acceleration $\varpi(t)$", color=RED, fontsize=10)
ax.text(-0.11, 0.08, r"$\xi(t,0)=0$", fontsize=9, color=GREY)
ax.text(0.95, 0.08, r"$\xi(t,1)=0$", fontsize=9, color=GREY)
ax.text(0.72, 0.28, r"$-\zeta\tanh\,\xi(t-\tau,w)$", fontsize=9, color=GREY)
ax.set_title(r"(a) viscoelastic structure on $w\in(0,1)$ with fixed ends", fontsize=9)

# (b) spatial profiles
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(w, b, color=AMBER, lw=2, label=r"actuator profile $b(w)$")
ax2.plot(w, phi / A0, color=BLUE, lw=2, ls="--", label=r"history shape $\phi(w)/A_0$")
ax2.set_xlabel(r"$w$"); ax2.set_ylabel("profile"); ax2.set_ylim(0, 1.6); ax2.grid(alpha=0.3)
ax2.legend(fontsize=8, loc="upper center", ncol=1); ax2.set_title("(b) spatial profiles", fontsize=9)

# (c) natural frequencies
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(n, om, "o-", color=BLUE, lw=1.8, ms=6, mec="white", mew=1.0, label=r"$\omega_n=\sqrt{c^2\lambda_n/(1+a\lambda_n)}$")
ax3.axhline(c / np.sqrt(a), color=GREY, lw=1, ls=":", label=r"limit $c/\sqrt{a}=3.16$")
ax3.annotate(f"$\\omega_1={om[0]:.2f}$", (1, om[0]), xytext=(1.6, 1.35), fontsize=8,
             arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))
ax3.set_xlabel(r"mode $n$"); ax3.set_ylabel(r"$\omega_n$"); ax3.set_ylim(0, 3.6); ax3.grid(alpha=0.3)
ax3.legend(fontsize=8, loc="lower right"); ax3.set_title("(c) natural frequencies", fontsize=9)

import os; out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results") + os.sep
fig.savefig(out + "fig0_model.pdf", bbox_inches="tight"); fig.savefig(out + "fig0_model.png", bbox_inches="tight", dpi=200)
print(om[:3])
