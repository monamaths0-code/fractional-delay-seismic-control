"""Seismic part of the numerical study with the recorded 1940 El Centro NS ground acceleration.
Record: data/raw/elcentro_NS.txt (time [s], acceleration [g]), dt = 0.02 s, 2688 points.
Time scaling: the first natural period of the model structure is set to T1 = 1.5 s,
so one model time unit = omega_1 * T1 / (2 pi) seconds. The record is normalized so that
its peak ground acceleration equals 4 model units."""
import json, math, time, os
import numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "numerics.py")).read()
exec(src[:src.index("results = {}")])          # parameters, Solver, helpers, plotting style
_i = src.index("def train_nn"); _j = src.index("with torch.no_grad():\n    J_unc")
exec(src[_i:_j])                               # train_nn, train_direct
# Controller network with a Fourier-feature input layer (needed for the long, oscillatory record)
class FNet(torch.nn.Module):
    def __init__(s, nf=12, width=48):
        super().__init__()
        s.k = torch.arange(1, nf + 1).double() * math.pi
        s.f = torch.nn.Sequential(torch.nn.Linear(2 * nf + 1, width), torch.nn.Tanh(),
                                  torch.nn.Linear(width, width), torch.nn.Tanh(), torch.nn.Linear(width, 1))
    def forward(s, t): return s.f(torch.cat([t, torch.sin(s.k * t), torch.cos(s.k * t)], 1))

def train_nn(solver, vdist, iters=400, lr=1e-2, seed=0):
    torch.manual_seed(seed); ctrl = FNet()
    tt_ = torch.tensor(solver.t).unsqueeze(1) / solver.p["Theta"]
    opt_ = torch.optim.Adam(ctrl.parameters(), lr=lr); hist = []
    for it in range(iters):
        opt_.zero_grad(); mu = solver.p["umax"] * torch.tanh(ctrl(tt_).squeeze(1))
        Y, g = solver.solve(mu, vdist, iters=2); Jc = solver.cost(Y, mu); Jc.backward(); opt_.step()
        hist.append(float(Jc.detach()))
    with torch.no_grad():
        mu = solver.p["umax"] * torch.tanh(ctrl(tt_).squeeze(1)); Y, g = solver.solve(mu, vdist)
        return mu, Y, g, float(solver.cost(Y, mu)), hist
NPARAM = sum(p.numel() for p in FNet().parameters())


P2 = dict(P); P2.update(Theta=15.0, h=0.05, zeta=0.01)
T1, PGA_ND, SHIFT = 1.5, 4.0, 1.0            # period [s], normalized PGA, shift [s] of 2nd record
sol2 = Solver(P2); S2 = sol2.S; M2 = sol2.M; tg2 = sol2.t; h2 = P2["h"]
om1 = float(np.sqrt(S2["om2"][0]))
tscale = om1 * T1 / (2 * np.pi)              # seconds per model time unit
rec = np.loadtxt(os.path.join(here, "..", "data", "raw", "elcentro_NS.txt"))
tr, ar = rec[:, 0], rec[:, 1]
pga = np.abs(ar).max()
def record(shift=0.0):
    return PGA_ND / pga * np.interp(tg2 * tscale - shift, tr, ar, left=0.0, right=0.0)
eq1 = record(0.0); eq2 = record(SHIFT)
TC2 = theory_constants(P2, S2)
res = dict(nparam_nn=NPARAM, theory=TC2, tscale=tscale, phys_window=float(tg2[-1] * tscale), pga_g=float(pga), omega1=om1)
print("theory", TC2, "window [s]", res["phys_window"])

v1 = torch.tensor(eq1); mu0 = torch.zeros(M2 + 1)
with torch.no_grad():
    Yb, gb = sol2.solve(mu0, v1)
    J_unc = float(sol2.cost(Yb, mu0))
# robustness
rob = []
with torch.no_grad():
    for eps in [0.02, 0.05, 0.1, 0.2, 0.4, 0.8]:
        vp = torch.tensor(eq1 + eps * (eq2 - eq1)); Yp, gp = sol2.solve(mu0, vp)
        d = float(torch.max(torch.cat([sol2.l2norm_t(Yp - Yb), torch.sqrt(torch.sum((gp - gb) ** 2)).unsqueeze(0)])))
        rob.append((eps, l2t(eps * (eq2 - eq1), h2), d))
rob = np.array(rob)
res["robustness"] = dict(rows=rob.tolist(), empirical_gain=float(np.max(rob[:, 2] / rob[:, 1])), theoretical_gain=TC2["Lambda_varpi"])
print("rob", res["robustness"]["empirical_gain"], time.time())

t0 = time.time()
mu_nn, Y_nn, g_nn, J_nn, h_nn = train_nn(sol2, v1); t_nn = time.time() - t0; t0 = time.time()
mu_d, Y_d, g_d, J_d, h_d = train_direct(sol2, v1); t_d = time.time() - t0
X = lambda Y: (Y @ sol2.Phi.T)
res["oc"] = dict(J_uncontrolled=J_unc, J_nn=J_nn, J_direct=J_d, time_nn=t_nn, time_direct=t_d,
                 peak_unc=float(X(Yb).abs().max()), peak_nn=float(X(Y_nn).abs().max()), peak_direct=float(X(Y_d).abs().max()),
                 rms_unc=l2t(sol2.l2norm_t(Yb).numpy(), h2) / math.sqrt(P2["Theta"]),
                 rms_nn=l2t(sol2.l2norm_t(Y_nn).numpy(), h2) / math.sqrt(P2["Theta"]),
                 rms_direct=l2t(sol2.l2norm_t(Y_d).numpy(), h2) / math.sqrt(P2["Theta"]),
                 umax_nn=float(mu_nn.abs().max()), umax_direct=float(mu_d.abs().max()))
print("OC", res["oc"])

iw = int(np.argmin(np.abs(S2["w"] - 0.5))); ts = tg2 * tscale
fig, axs = plt.subplots(3, 1, figsize=(7.0, 5.8), sharex=True)
axs[0].plot(tr, ar, color="#bbbbbb", lw=0.7)
axs[0].plot(ts, eq1 * pga / PGA_ND, color=C["grey"], lw=1.0)
axs[0].set_xlim(0, ts[-1]); axs[0].set_ylabel(r"$a_g$ [g]")
axs[0].set_title("(a) El Centro 1940, NS component (recorded)", fontsize=9)
xb, xn, xd = X(Yb).numpy()[:, iw], X(Y_nn).numpy()[:, iw], X(Y_d).numpy()[:, iw]
axs[1].plot(ts, xb, color=C["red"], lw=1.3, label="uncontrolled")
axs[1].plot(ts, xd, color=C["amber"], lw=1.3, ls="--", label="direct method")
axs[1].plot(ts, xn, color=C["blue"], lw=1.5, label="NN controller")
axs[1].set_ylabel(r"$\xi(t,1/2)$"); axs[1].legend(ncol=3, fontsize=8); axs[1].set_title("(b) midpoint displacement", fontsize=9)
axs[2].plot(ts, mu_d.numpy(), color=C["amber"], lw=1.2, ls="--", label="direct method")
axs[2].plot(ts, mu_nn.numpy(), color=C["blue"], lw=1.5, label="NN controller")
axs[2].axvspan(0, P2["kappa"] * tscale, color=C["grey"], alpha=0.2, lw=0)
axs[2].set_ylabel(r"$\mu(t)$"); axs[2].set_xlabel("physical time [s]"); axs[2].legend(fontsize=8)
axs[2].set_title(r"(c) optimal control (shaded: delay interval $[0,\kappa]$)", fontsize=9)
fig.tight_layout(); fig.savefig(OUT + "fig4_seismic_control.pdf"); fig.savefig(OUT + "fig4_seismic_control.png", dpi=200); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
axs[0].semilogy(h_nn, color=C["blue"], lw=1.4, label="NN controller")
axs[0].semilogy(h_d, color=C["amber"], lw=1.4, ls="--", label="direct method")
axs[0].axhline(J_unc, color=C["red"], lw=1.0, ls=":", label="uncontrolled")
axs[0].set_xlabel("iteration"); axs[0].set_ylabel(r"$\mathcal{J}(\mu)$"); axs[0].legend(fontsize=8); axs[0].set_title("(a) cost during training", fontsize=9)
axs[1].loglog(rob[:, 1], rob[:, 2], "o-", color=C["blue"], ms=4, lw=1.3, label=r"computed $\|\xi^{\varpi^1}-\xi^{\varpi^2}\|_\ast$")
xx = np.array([rob[:, 1].min(), rob[:, 1].max()])
axs[1].loglog(xx, TC2["Lambda_varpi"] * xx, color=C["red"], lw=1.3, ls="--", label=r"bound $\Lambda_\varpi\|\varpi^1-\varpi^2\|$")
axs[1].set_xlabel(r"$\|\varpi^1-\varpi^2\|_{L^2}$"); axs[1].legend(fontsize=8); axs[1].set_title("(b) robustness estimate", fontsize=9)
fig.tight_layout(); fig.savefig(OUT + "fig5_training_robustness.pdf"); fig.savefig(OUT + "fig5_training_robustness.png", dpi=200); plt.close(fig)

kap = {}
for k_ in [0.0, 0.1, 0.3]:
    Pk = dict(P2); Pk["kappa"] = k_; sk = Solver(Pk)
    _, Yk, _, Jk, _ = train_nn(sk, v1)
    kap[str(k_)] = dict(J=Jk, peak=float((Yk @ sk.Phi.T).abs().max()))
res["delay"] = kap; print("delay", kap)
v2 = torch.tensor(eq1 + 0.2 * (eq2 - eq1))
_, Y2, _, J2, _ = train_nn(sol2, v2)
res["value"] = dict(V1=J_nn, V2=J2, diff=abs(J_nn - J2), dw=l2t(0.2 * (eq2 - eq1), h2))
json.dump(res, open(OUT + "results_elcentro.json", "w"), indent=2)
print(json.dumps(res, indent=2))
