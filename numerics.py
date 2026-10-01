"""
Numerical validation for:
Optimal Control of Delayed Sobolev-Type Fractional Evolution Systems of Order (1,2)
under Seismic Disturbances with Neural Network Validation.

Model (1D viscoelastic structure, w in (0,1), Dirichlet ends):
  cD^rho_t (I - a d_ww) xi = c^2 d_ww xi - zeta*tanh(xi(t-tau)) + b(w) mu(t-kappa) - m(w) varpi(t)
  xi(theta) = phi(theta) + g(xi),  theta in [-tau,0],  g(xi) = c_g E^{-1} xi(t_1),  xi_t(0) = 0.
Spectral Galerkin in space (sine modes), Diethelm-Ford-Freed Adams-Bashforth-Moulton in time.
"""
import json, math, time, os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
torch.manual_seed(0); np.random.seed(0)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results") + os.sep
import os; os.makedirs(OUT, exist_ok=True)

# ---------------- parameters ----------------
P = dict(rho=1.6, a=0.1, c=1.0, zeta=0.1, c_g=0.1, t1=2.0, tau=0.2, kappa=0.1,
         Theta=4.0, h=0.02, N=8, Nw=64, A0=0.05, rho_u=1e-4, umax=10.0, eqscale=4.0)

def setup(p):
    N, Nw = p["N"], p["Nw"]
    n = np.arange(1, N + 1)
    lam = (n * np.pi) ** 2
    Ediag = 1 + p["a"] * lam                     # eigenvalues of E
    om2 = p["c"] ** 2 * lam / Ediag              # -eigenvalues of G E^{-1}
    w = (np.arange(Nw) + 0.5) / Nw               # midpoint grid
    Phi = np.sqrt(2) * np.sin(np.outer(w, n * np.pi))   # (Nw,N) basis values
    Wq = Phi.T / Nw                               # projection: coeffs = Wq @ values
    b = np.exp(-((w - 0.5) / 0.1) ** 2)           # actuator profile
    m = np.ones(Nw)                               # uniform inertial load
    return dict(n=n, lam=lam, E=Ediag, om2=om2, w=w, Phi=Phi, Wq=Wq,
                b_n=Wq @ b, m_n=Wq @ m, b=b, m=m)

def weights(M, h, al):
    """DFF predictor B and corrector A weights for order al (1<al<2)."""
    k = np.arange(M + 1)
    B = np.zeros((M + 1, M + 1)); A = np.zeros((M + 1, M + 1)); Adiag = np.zeros(M + 1)
    c = h ** al / (al * (al + 1))
    for kk in range(M):  # step kk -> kk+1
        j = np.arange(kk + 1)
        B[kk + 1, :kk + 1] = h ** al / al * ((kk + 1 - j) ** al - (kk - j) ** al)
        a = np.zeros(kk + 1)
        a[0] = kk ** (al + 1) - (kk - al) * (kk + 1) ** al
        jj = np.arange(1, kk + 1)
        a[1:] = (kk - jj + 2) ** (al + 1) + (kk - jj) ** (al + 1) - 2 * (kk - jj + 1) ** (al + 1)
        A[kk + 1, :kk + 1] = c * a
        Adiag[kk + 1] = c
    return B / math.gamma(al), A / math.gamma(al), Adiag / math.gamma(al)

class Solver:
    def __init__(self, p):
        self.p = p; self.S = setup(p)
        self.M = int(round(p["Theta"] / p["h"])); self.t = np.arange(self.M + 1) * p["h"]
        B, A, Ad = weights(self.M, p["h"], p["rho"])
        T = lambda x: torch.tensor(x)
        self.B, self.A, self.Ad = T(B), T(A), T(Ad)
        S = self.S
        self.E, self.om2 = T(S["E"]), T(S["om2"])
        self.Phi, self.Wq = T(S["Phi"]), T(S["Wq"])
        self.b_n, self.m_n = T(S["b_n"]), T(S["m_n"])
        self.dtau = int(round(p["tau"] / p["h"])); self.dkap = int(round(p["kappa"] / p["h"]))
        self.i1 = int(round(p["t1"] / p["h"]))
        phi = p["A0"] * np.sin(np.pi * S["w"])        # constant history shape
        self.phi_n = T(S["Wq"] @ phi)

    def rhs(self, y, ydel, u, v):
        # modal right-hand side of D^rho y = -om2 y + F_n / E_n
        xdel = self.Phi @ ydel
        J = -self.p["zeta"] * torch.tanh(xdel)
        F = self.Wq @ J + self.b_n * u - self.m_n * v
        return -self.om2 * y + F / self.E

    def solve_given_g(self, g_n, u, v):
        """u,v: tensors of length M+1 (u already delayed-consistent: u[k]=mu(t_k-kappa))."""
        y0 = self.phi_n + g_n
        Y = [y0]; Fs = [self.rhs(y0, y0, u[0], v[0])]
        for k in range(self.M):
            Fst = torch.stack(Fs)
            kd = k + 1 - self.dtau
            ydel = Y[kd] if kd >= 0 else y0
            pred = y0 + self.B[k + 1, :k + 1] @ Fst
            fpred = self.rhs(pred, ydel, u[k + 1], v[k + 1])
            corr = y0 + self.A[k + 1, :k + 1] @ Fst + self.Ad[k + 1] * fpred
            Y.append(corr); Fs.append(self.rhs(corr, ydel, u[k + 1], v[k + 1]))
        return torch.stack(Y)          # (M+1, N)

    def solve(self, mu, varpi, psi=None, iters=30, tol=1e-12):
        """mu: control on grid t_k (length M+1); delayed by kappa using history psi (default 0)."""
        dk = self.dkap
        hist = torch.zeros(dk) if psi is None else psi
        u = torch.cat([hist, mu[:self.M + 1 - dk]]) if dk > 0 else mu
        g = torch.zeros_like(self.phi_n)
        for _ in range(iters):
            Y = self.solve_given_g(g, u, varpi)
            g_new = self.p["c_g"] * Y[self.i1] / self.E      # g = c_g E^{-1} xi(t1)
            if torch.max(torch.abs(g_new - g)) < tol: g = g_new; break
            g = g_new.detach() if not g_new.requires_grad else g_new
        Y = self.solve_given_g(g, u, varpi)
        return Y, g

    def l2norm_t(self, Y):  # ||xi(t)||_{L2} = Euclidean norm of modal coefficients
        return torch.sqrt(torch.sum(Y ** 2, dim=1))

    def supnorm(self, Y, g):
        hist = torch.sqrt(torch.sum((self.phi_n + g) ** 2))
        return torch.maximum(self.l2norm_t(Y).max(), hist)

    def cost(self, Y, mu):
        h = self.p["h"]
        return h * torch.sum(self.l2norm_t(Y) ** 2 + self.p["rho_u"] * mu ** 2)

# ---------------- synthetic earthquake (Kanai-Tajimi + envelope) ----------------
def kanai_tajimi(t, seed, wg=3.0, zg=0.4, scale=1.0):
    """Filtered white noise ground acceleration with a Jennings-type envelope.
    Time t is non-dimensional; wg, zg are ground frequency and damping in these units."""
    rng = np.random.default_rng(seed)
    dt = t[1] - t[0]; nsub = 20; dts = dt / nsub
    n = len(t) * nsub
    wn = rng.standard_normal(n) / np.sqrt(dts)
    x = v = 0.0; acc = np.zeros(n)
    for i in range(n):
        a_rel = -2 * zg * wg * v - wg ** 2 * x - wn[i]
        v += a_rel * dts; x += v * dts
        acc[i] = -(2 * zg * wg * v + wg ** 2 * x)
    acc = acc[::nsub][:len(t)]
    T = t[-1]; t1e, t2e = 0.1 * T, 0.45 * T
    env = np.where(t < t1e, (t / t1e) ** 2, np.where(t < t2e, 1.0, np.exp(-2.5 * (t - t2e) / T)))
    g = acc * env
    return scale * g / np.max(np.abs(g))

def l2t(x, h):  # L2(0,Theta) norm of a grid function (trapezoid)
    x = np.asarray(x); return math.sqrt(h * (np.sum(x ** 2) - 0.5 * (x[0] ** 2 + x[-1] ** 2)))

# ---------------- theoretical constants ----------------
def theory_constants(p, S):
    vs = p["rho"] / 2
    Einv = 1 / S["E"].min()
    Ups = 1.0
    UE = Ups * Einv; UP = UE / math.gamma(2 * vs)
    fac = 2 * UE * p["Theta"] ** (2 * vs) / math.gamma(2 * vs + 1)
    lamJ = etaJ = p["zeta"]; lamg = eta1 = abs(p["c_g"])
    g1 = UE * eta1 + etaJ * fac
    g2 = UE * lamg + lamJ * fac
    Dnorm = math.sqrt(np.mean(S["m"] ** 2))
    Lam_w = UP * Dnorm * p["Theta"] ** (2 * vs - 0.5) / (1 - g2)
    return dict(varsigma=vs, Einv=Einv, UpsE=UE, UpsP=UP, gamma1=g1, gamma2=g2,
                Dnorm=Dnorm, Lambda_varpi=Lam_w)

# ---------------- plotting style ----------------
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "figure.dpi": 150})
C = dict(blue="#005AB4", red="#B3261E", green="#2E7D32", amber="#B8860B", grey="#555555", purple="#6A3D9A")

results = {}
t0 = time.time()
sol = Solver(P); S = sol.S; M = sol.M; tg = sol.t; h = P["h"]
TC = theory_constants(P, S); results["theory"] = TC
print("theory", TC)

# ================= Example 1: smooth disturbance, ABM vs fPINN =================
v_s = torch.tensor(0.5 * np.sin(2.0 * tg))
mu0 = torch.zeros(M + 1)
with torch.no_grad():
    Yref, gref = sol.solve(mu0, v_s)
# convergence check of ABM: halve the step
P_f = dict(P); P_f["h"] = P["h"] / 2
sol_f = Solver(P_f)
with torch.no_grad():
    Yf, _ = sol_f.solve(torch.zeros(sol_f.M + 1), torch.tensor(0.5 * np.sin(2.0 * sol_f.t)))
err_h = float(torch.max(torch.abs(Yf[::2] - Yref)))
results["abm_step_halving_error"] = err_h

# fPINN in integral form (hard initial constraints, residual of the Volterra equation,
# Fourier-feature input layer, Adam followed by L-BFGS)
class Net(torch.nn.Module):
    def __init__(s, N, width=64, nf=8):
        super().__init__()
        s.k = torch.arange(1, nf + 1).double() * 0.8
        s.f = torch.nn.Sequential(torch.nn.Linear(2 * nf + 1, width), torch.nn.Tanh(),
                                  torch.nn.Linear(width, width), torch.nn.Tanh(),
                                  torch.nn.Linear(width, width), torch.nn.Tanh(),
                                  torch.nn.Linear(width, N))
    def forward(s, t):
        z = t * P["Theta"]
        return s.f(torch.cat([t, torch.sin(s.k * z), torch.cos(s.k * z)], 1))

torch.manual_seed(1)
tt = torch.tensor(tg).unsqueeze(1)
net = Net(P["N"]); g_par = torch.zeros(P["N"], requires_grad=True)
Aful = sol.A.clone(); Aful[torch.arange(M + 1), torch.arange(M + 1)] = sol.Ad  # product-trapezoidal weights
SC = 1e4
def pinn_loss():
    y0 = sol.phi_n + g_par
    Y = y0 + (tt / P["Theta"]) ** 2 * net(tt / P["Theta"])      # y(0)=y0 and y'(0)=0 built in
    idx = torch.arange(M + 1) - sol.dtau
    Ydel = torch.where((idx >= 0).unsqueeze(1), Y[idx.clamp(min=0)], y0)
    Jn = (-P["zeta"] * torch.tanh(Ydel @ sol.Phi.T)) @ sol.Wq.T
    F = -sol.om2 * Y + (Jn - sol.m_n * v_s.unsqueeze(1)) / sol.E
    R = Y - y0 - Aful @ F
    return torch.mean(R ** 2) + torch.sum((g_par - P["c_g"] * Y[sol.i1] / sol.E) ** 2), Y
opt = torch.optim.Adam(list(net.parameters()) + [g_par], lr=2e-3)
hist_loss = []
for it in range(20000):
    opt.zero_grad(); l, _ = pinn_loss(); (SC * l).backward(); opt.step()
    if it == 14000:
        for gr in opt.param_groups: gr["lr"] = 5e-4
    if it % 100 == 0: hist_loss.append((it, float(l.detach())))
optL = torch.optim.LBFGS(list(net.parameters()) + [g_par], max_iter=2000, tolerance_grad=1e-12,
                         tolerance_change=1e-15, history_size=50, line_search_fn="strong_wolfe")
def closure():
    optL.zero_grad(); l, _ = pinn_loss(); (SC * l).backward(); return SC * l
for k in range(6):
    optL.step(closure)
    with torch.no_grad(): hist_loss.append((20000 + 2000 * (k + 1), float(pinn_loss()[0])))
with torch.no_grad():
    lval, Ypinn = pinn_loss()
xref = (Yref @ sol.Phi.T).numpy(); xpinn = (Ypinn @ sol.Phi.T).numpy()
err_abs = np.abs(xref - xpinn)
rel_l2 = float(np.linalg.norm(xref - xpinn) / np.linalg.norm(xref))
results["fpinn"] = dict(max_abs=float(err_abs.max()), rel_l2=rel_l2, final_loss=float(lval), peak=float(np.abs(xref).max()),
                        g_err=float(torch.max(torch.abs(g_par - gref))))
print("fPINN", results["fpinn"], "time", time.time() - t0)

# Figure 1: surface of reference + error
fig = plt.figure(figsize=(7.2, 2.9))
ax = fig.add_subplot(1, 2, 1, projection="3d")
WW, TT = np.meshgrid(S["w"], tg)
ax.plot_surface(TT, WW, xref, cmap="viridis", linewidth=0, antialiased=True)
ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$w$"); ax.set_zlabel(r"$\xi(t,w)$"); ax.set_title("(a) ABM reference solution", fontsize=9)
ax.view_init(28, -125)
ax2 = fig.add_subplot(1, 2, 2)
cs = ax2.contourf(TT, WW, err_abs, 30, cmap="magma"); fig.colorbar(cs, ax=ax2, fraction=0.046)
ax2.set_xlabel(r"$t$"); ax2.set_ylabel(r"$w$"); ax2.set_title(r"(b) $|\xi_{\mathrm{ABM}}-\xi_{\mathrm{fPINN}}|$", fontsize=9)
ax2.grid(False)
fig.tight_layout(); fig.savefig(OUT + "fig1_solution_error.pdf"); fig.savefig(OUT + "fig1_solution_error.png"); plt.close(fig)

# Figure 2: midpoint comparison + training loss
iw = np.argmin(np.abs(S["w"] - 0.5))
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7))
axs[0].plot(tg, xref[:, iw], color=C["blue"], lw=1.6, label="ABM (classical)")
axs[0].plot(tg[::6], xpinn[::6, iw], "o", ms=3.2, mfc="none", color=C["red"], label="fPINN")
axs[0].set_xlabel(r"$t$"); axs[0].set_ylabel(r"$\xi(t,1/2)$"); axs[0].legend(); axs[0].set_title("(a) midpoint displacement", fontsize=9)
hl = np.array(hist_loss)
axs[1].semilogy(hl[:, 0], hl[:, 1], color=C["green"], lw=1.4)
axs[1].set_xlabel("iteration"); axs[1].set_ylabel("loss"); axs[1].set_title("(b) fPINN training loss", fontsize=9)
fig.tight_layout(); fig.savefig(OUT + "fig2_fpinn.pdf"); fig.savefig(OUT + "fig2_fpinn.png"); plt.close(fig)

# ================= Effect of the fractional order =================
fig, ax = plt.subplots(figsize=(3.6, 2.7))
ord_res = {}
for rho_, col in zip([1.2, 1.5, 1.8], [C["amber"], C["blue"], C["red"]]):
    Pr = dict(P); Pr["rho"] = rho_; sr = Solver(Pr)
    with torch.no_grad():
        Yr, _ = sr.solve(torch.zeros(sr.M + 1), torch.tensor(0.5 * np.sin(2.0 * sr.t)))
    xm = (Yr @ sr.Phi.T).numpy()[:, iw]
    ax.plot(sr.t, xm, color=col, lw=1.4, label=rf"$\varrho={rho_}$")
    ord_res[str(rho_)] = float(np.max(np.abs(xm)))
ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$\xi(t,1/2)$"); ax.legend()
fig.tight_layout(); fig.savefig(OUT + "fig3_order.pdf"); fig.savefig(OUT + "fig3_order.png"); plt.close(fig)
results["order_peak"] = ord_res

# ================= Example 2: seismic disturbance =================
eq1 = kanai_tajimi(tg, seed=11, scale=P["eqscale"])
eq2 = kanai_tajimi(tg, seed=29, scale=P["eqscale"])
v1 = torch.tensor(eq1)

# --- robustness test: perturbations of eq1 ---
rob = []
with torch.no_grad():
    Yb, gb = sol.solve(mu0, v1)
    for eps in [0.02, 0.05, 0.1, 0.2, 0.4, 0.8]:
        vp = torch.tensor(eq1 + eps * (eq2 - eq1))
        Yp, gp = sol.solve(mu0, vp)
        dstate = float(torch.max(torch.cat([sol.l2norm_t(Yp - Yb), torch.sqrt(torch.sum((gp - gb) ** 2)).unsqueeze(0)])))
        dw = l2t(eps * (eq2 - eq1), h)
        rob.append((eps, dw, dstate))
rob = np.array(rob)
emp_gain = float(np.max(rob[:, 2] / rob[:, 1]))
results["robustness"] = dict(rows=rob.tolist(), empirical_gain=emp_gain, theoretical_gain=TC["Lambda_varpi"])

# --- optimal control: NN controller vs classical direct method ---
def train_nn(solver, vdist, iters=500, lr=1e-2, seed=0):
    torch.manual_seed(seed)
    ctrl = torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.Tanh(), torch.nn.Linear(32, 32),
                               torch.nn.Tanh(), torch.nn.Linear(32, 1))
    tt_ = torch.tensor(solver.t).unsqueeze(1) / solver.p["Theta"]
    opt_ = torch.optim.Adam(ctrl.parameters(), lr=lr); hist = []
    for it in range(iters):
        opt_.zero_grad()
        mu = P["umax"] * torch.tanh(ctrl(tt_).squeeze(1))          # |mu| <= 3 : admissible set
        Y, g = solver.solve(mu, vdist, iters=2)
        Jc = solver.cost(Y, mu); Jc.backward(); opt_.step(); hist.append(float(Jc))
    with torch.no_grad():
        mu = P["umax"] * torch.tanh(ctrl(tt_).squeeze(1)); Y, g = solver.solve(mu, vdist)
        return mu, Y, g, float(solver.cost(Y, mu)), hist

def train_direct(solver, vdist, iters=500, lr=5e-2):
    z = torch.zeros(solver.M + 1, requires_grad=True)
    opt_ = torch.optim.Adam([z], lr=lr); hist = []
    for it in range(iters):
        opt_.zero_grad()
        mu = P["umax"] * torch.tanh(z)
        Y, g = solver.solve(mu, vdist, iters=2)
        Jc = solver.cost(Y, mu); Jc.backward(); opt_.step(); hist.append(float(Jc))
    with torch.no_grad():
        mu = P["umax"] * torch.tanh(z); Y, g = solver.solve(mu, vdist)
        return mu, Y, g, float(solver.cost(Y, mu)), hist

with torch.no_grad():
    J_unc = float(sol.cost(Yb, mu0))
t1_ = time.time()
mu_nn, Y_nn, g_nn, J_nn, h_nn = train_nn(sol, v1)
t_nn = time.time() - t1_; t1_ = time.time()
mu_d, Y_d, g_d, J_d, h_d = train_direct(sol, v1)
t_d = time.time() - t1_
results["oc"] = dict(J_uncontrolled=J_unc, J_nn=J_nn, J_direct=J_d, time_nn=t_nn, time_direct=t_d,
                     peak_unc=float((Yb @ sol.Phi.T).abs().max()), peak_nn=float((Y_nn @ sol.Phi.T).abs().max()),
                     peak_direct=float((Y_d @ sol.Phi.T).abs().max()),
                     rms_unc=l2t(sol.l2norm_t(Yb).numpy(), h) / math.sqrt(P["Theta"]),
                     rms_nn=l2t(sol.l2norm_t(Y_nn).numpy(), h) / math.sqrt(P["Theta"]),
                     rms_direct=l2t(sol.l2norm_t(Y_d).numpy(), h) / math.sqrt(P["Theta"]))
print("OC", results["oc"], time.time() - t0)

# Figure 4: earthquake + controlled/uncontrolled midpoint + control
fig, axs = plt.subplots(3, 1, figsize=(7.0, 5.6), sharex=True)
axs[0].plot(tg, eq1, color=C["grey"], lw=0.9); axs[0].set_ylabel(r"$\varpi(t)$"); axs[0].set_title("(a) synthetic ground acceleration (record 1)", fontsize=9)
xb = (Yb @ sol.Phi.T).numpy()[:, iw]; xn = (Y_nn @ sol.Phi.T).numpy()[:, iw]; xd = (Y_d @ sol.Phi.T).numpy()[:, iw]
axs[1].plot(tg, xb, color=C["red"], lw=1.3, label="uncontrolled")
axs[1].plot(tg, xd, color=C["amber"], lw=1.3, ls="--", label="direct method")
axs[1].plot(tg, xn, color=C["blue"], lw=1.5, label="NN controller")
axs[1].set_ylabel(r"$\xi(t,1/2)$"); axs[1].legend(ncol=3, fontsize=8); axs[1].set_title("(b) midpoint displacement", fontsize=9)
axs[2].plot(tg, mu_d.numpy(), color=C["amber"], lw=1.2, ls="--", label="direct method")
axs[2].plot(tg, mu_nn.numpy(), color=C["blue"], lw=1.5, label="NN controller")
axs[2].axvspan(0, P["kappa"], color=C["grey"], alpha=0.15, lw=0)
axs[2].set_ylabel(r"$\mu(t)$"); axs[2].set_xlabel(r"$t$"); axs[2].legend(fontsize=8); axs[2].set_title(r"(c) optimal control (shaded: $[0,\kappa]$)", fontsize=9)
fig.tight_layout(); fig.savefig(OUT + "fig4_seismic_control.pdf"); fig.savefig(OUT + "fig4_seismic_control.png"); plt.close(fig)

# Figure 5: training histories + robustness
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
axs[0].semilogy(h_nn, color=C["blue"], lw=1.4, label="NN controller")
axs[0].semilogy(h_d, color=C["amber"], lw=1.4, ls="--", label="direct method")
axs[0].axhline(J_unc, color=C["red"], lw=1.0, ls=":", label="uncontrolled")
axs[0].set_xlabel("iteration"); axs[0].set_ylabel(r"$\mathcal{J}(\mu)$"); axs[0].legend(fontsize=8); axs[0].set_title("(a) cost during training", fontsize=9)
axs[1].loglog(rob[:, 1], rob[:, 2], "o-", color=C["blue"], ms=4, lw=1.3, label=r"computed $\|\xi^{\varpi^1}-\xi^{\varpi^2}\|_\ast$")
xx = np.array([rob[:, 1].min(), rob[:, 1].max()])
axs[1].loglog(xx, TC["Lambda_varpi"] * xx, color=C["red"], lw=1.3, ls="--", label=r"bound $\Lambda_\varpi\|\varpi^1-\varpi^2\|$")
axs[1].set_xlabel(r"$\|\varpi^1-\varpi^2\|_{L^2}$"); axs[1].legend(fontsize=8); axs[1].set_title("(b) robustness estimate (4.6)", fontsize=9)
fig.tight_layout(); fig.savefig(OUT + "fig5_training_robustness.pdf"); fig.savefig(OUT + "fig5_training_robustness.png"); plt.close(fig)

# ================= Effect of the control delay =================
kap_res = {}
for kap in [0.0, 0.1, 0.3]:
    Pk = dict(P); Pk["kappa"] = kap; sk = Solver(Pk)
    _, Yk, _, Jk, _ = train_nn(sk, v1, iters=500)
    kap_res[str(kap)] = dict(J=Jk, peak=float((Yk @ sk.Phi.T).abs().max()))
results["delay"] = kap_res
print("delay", kap_res, time.time() - t0)

# ================= Robustness of the optimal cost (Corollary 5.4) =================
v2 = torch.tensor(eq1 + 0.2 * (eq2 - eq1))
_, Y2, _, J2, _ = train_nn(sol, v2, iters=500)
with torch.no_grad():
    Rmax = float(max(sol.supnorm(Yb, gb), sol.supnorm(Y2, torch.zeros(P["N"]))))
results["value"] = dict(V1=J_nn, V2=J2, diff=abs(J_nn - J2), dw=l2t(0.2 * (eq2 - eq1), h))
results["time_total"] = time.time() - t0
json.dump(results, open(OUT + "results.json", "w"), indent=2)
print(json.dumps(results, indent=2))
