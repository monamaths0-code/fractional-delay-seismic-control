"""fPINN for the uncontrolled structure under the recorded El Centro NS ground acceleration."""
import json, math, time, os
import numpy as np, torch, matplotlib, matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "numerics_elcentro.py")).read()
__file__ = os.path.join(here, "numerics_elcentro.py")
exec(src[:src.index("v1 = torch.tensor(eq1)")])      # seismic setting, record, solver sol2

v1 = torch.tensor(eq1); M = sol2.M; Th = P2["Theta"]
with torch.no_grad():
    Yref, gref = sol2.solve(torch.zeros(M + 1), v1)

class Net(torch.nn.Module):
    """Small network for one time window, with a Fourier-feature input layer."""
    def __init__(s, N, width=64, nf=12, L=3.0):
        super().__init__()
        s.k = torch.arange(1, nf + 1).double() * 0.8; s.L = L
        s.f = torch.nn.Sequential(torch.nn.Linear(2 * nf + 1, width), torch.nn.Tanh(),
                                  torch.nn.Linear(width, width), torch.nn.Tanh(),
                                  torch.nn.Linear(width, width), torch.nn.Tanh(), torch.nn.Linear(width, N))
    def forward(s, r):                       # r = local time / L in [0,1]
        z = r * s.L
        return s.f(torch.cat([r, torch.sin(s.k * z), torch.cos(s.k * z)], 1))

# Time-marching fPINN: the interval is split into windows; each window has its own network,
# and the solution on earlier windows is frozen and enters through the memory term.
torch.manual_seed(1)
Aful = sol2.A.clone(); Aful[torch.arange(M + 1), torch.arange(M + 1)] = sol2.Ad
SC = 1e4; NW = 5; Lw = Th / NW; kw = int(round(Lw / P2["h"]))
tt = torch.tensor(tg2)
Yfix = torch.zeros(M + 1, P2["N"]); g_par = torch.zeros(P2["N"], requires_grad=True)
hist = []; it_glob = 0; t0 = time.time()
def assemble(net, j):
    s0, e0 = j * kw, (j + 1) * kw
    r = ((tt[s0:e0 + 1] - tt[s0]) / Lw).unsqueeze(1)
    if j == 0:
        y0 = sol2.phi_n + g_par
        Yw = y0 + r ** 2 * net(r)
    else:
        Yw = Yfix[s0] + r * net(r)
    Y = torch.cat([Yfix[:s0], Yw, Yw[-1:].expand(M - e0, -1)], 0)
    return Y, s0, e0
def loss_w(net, j):
    Y, s0, e0 = assemble(net, j)
    y0 = sol2.phi_n + g_par if j == 0 else (sol2.phi_n + g_par).detach()
    idx = torch.arange(M + 1) - sol2.dtau
    Ydel = torch.where((idx >= 0).unsqueeze(1), Y[idx.clamp(min=0)], y0)
    Jn = (-P2["zeta"] * torch.tanh(Ydel @ sol2.Phi.T)) @ sol2.Wq.T
    F = -sol2.om2 * Y + (Jn - sol2.m_n * v1.unsqueeze(1)) / sol2.E
    R = Y[s0:e0 + 1] - y0 - Aful[s0:e0 + 1] @ F
    l = torch.mean(torch.sum(R ** 2, 1)) / P2["N"]
    if s0 <= sol2.i1 <= e0:
        l = l + torch.sum((g_par - P2["c_g"] * Y[sol2.i1] / sol2.E) ** 2)
    return l, Y
for j in range(NW):
    net = Net(P2["N"], L=Lw)
    params = list(net.parameters()) + ([g_par] if j == 0 else [])
    opt = torch.optim.Adam(params, lr=2e-3)
    for it in range(8000):
        opt.zero_grad(); l, _ = loss_w(net, j); (SC * l).backward(); opt.step(); it_glob += 1
        if it == 5000:
            for gr in opt.param_groups: gr["lr"] = 5e-4
        if it_glob % 100 == 0: hist.append((it_glob, float(l.detach())))
    optL = torch.optim.LBFGS(params, max_iter=1500, tolerance_grad=1e-13, tolerance_change=1e-16,
                             history_size=50, line_search_fn="strong_wolfe")
    def closure():
        optL.zero_grad(); l, _ = loss_w(net, j); (SC * l).backward(); return SC * l
    for k in range(3):
        optL.step(closure); it_glob += 1500
        with torch.no_grad(): hist.append((it_glob, float(loss_w(net, j)[0])))
    with torch.no_grad():
        _, Y = loss_w(net, j); s0, e0 = j * kw, (j + 1) * kw
        Yfix[s0:e0 + 1] = Y[s0:e0 + 1]
    print("window", j, float(hist[-1][1]), time.time() - t0, flush=True)
Yp = Yfix.clone(); lval = hist[-1][1]
ttrain = time.time() - t0
xr = (Yref @ sol2.Phi.T).numpy(); xp = (Yp @ sol2.Phi.T).numpy()
err = np.abs(xr - xp)
res = dict(max_abs=float(err.max()), peak=float(np.abs(xr).max()), rel_l2=float(np.linalg.norm(xr - xp) / np.linalg.norm(xr)),
           g_err=float(torch.max(torch.abs(g_par - gref))), final_loss=float(lval), windows=NW, time=ttrain)
print(res)
json.dump(res, open(OUT + "results_fpinn_elcentro.json", "w"), indent=2)
np.savez(OUT + "fpinn_elcentro_data.npz", xr=xr, xp=xp, hist=np.array(hist))

iw = int(np.argmin(np.abs(S2["w"] - 0.5))); ts = tg2 * tscale
fig = plt.figure(figsize=(7.2, 4.9))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.5, wspace=0.45)
ax = fig.add_subplot(gs[0, :])
ax.plot(ts, xr[:, iw], color=C["blue"], lw=1.6, label="ABM (classical)")
ax.plot(ts[::3], xp[::3, iw], "o", ms=3.2, mfc="none", color=C["red"], label="fPINN")
ax.set_xlim(0, ts[-1]); ax.set_xlabel("physical time [s]"); ax.set_ylabel(r"$\xi(t,1/2)$")
ax.legend(ncol=2, fontsize=8, loc="lower left"); ax.set_title("(a) midpoint displacement under the El Centro record (no control)", fontsize=9)
ax2 = fig.add_subplot(gs[1, 0])
WW, TT = np.meshgrid(S2["w"], ts)
cs = ax2.contourf(TT, WW, 1e3 * err, 30, cmap="magma"); cb = fig.colorbar(cs, ax=ax2, fraction=0.05)
cb.locator = matplotlib.ticker.MaxNLocator(5); cb.update_ticks(); 
ax2.set_xlabel("physical time [s]"); ax2.set_ylabel(r"$w$"); ax2.grid(False)
ax2.set_title(r"(b) $|\xi_{\mathrm{ABM}}-\xi_{\mathrm{fPINN}}|\ (\times 10^{-3})$", fontsize=9)
ax3 = fig.add_subplot(gs[1, 1]); hl = np.array(hist)
ax3.semilogy(hl[:, 0], hl[:, 1], color=C["green"], lw=1.4)
ax3.set_xlabel("iteration"); ax3.set_ylabel("loss"); ax3.set_title("(c) fPINN training loss", fontsize=9)
fig.savefig(OUT + "fig_fpinn_elcentro.pdf", bbox_inches="tight"); fig.savefig(OUT + "fig_fpinn_elcentro.png", bbox_inches="tight", dpi=200)
