"""Additional numerical checks requested in the review (El Centro example).

1. Grid convergence of the ABM scheme in time (h) and in space (number of modes N).
2. Accuracy of the time-marching fPINN against a grid-converged reference solution.
3. Full-time-domain fPINN loss (Eq. (6.4)) of the assembled fPINN solution.
4. Effect of the nonlocal condition on the comparison of controlled and uncontrolled responses.
5. Time needed by the NN control solver and by the direct method to reach a target cost.

Run after numerics_elcentro.py and fpinn_elcentro.py. Output: ../results/revision_checks.json
Set RUN_LOCAL_IC = False to skip the extra optimization with c_g = 0 (about 6 minutes on a CPU).
"""
import json, math, os, time, csv
import numpy as np, torch

here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "numerics_elcentro.py")).read()
__file__ = os.path.join(here, "numerics_elcentro.py")
exec(src[:src.index("v1 = torch.tensor(eq1)")])      # P2, sol2, record, tscale, train_direct, ...
ROOT = os.path.abspath(os.path.join(here, ".."))
RUN_LOCAL_IC = True
out = {}

def forcing(solver):
    """El Centro record on the time grid of a given solver (same scaling as in the paper)."""
    return torch.tensor(PGA_ND / pga * np.interp(solver.t * tscale, tr, ar, left=0.0, right=0.0))

def run(h=0.05, N=8, Nw=64, **kw):
    p = dict(P2); p.update(h=h, N=N, Nw=Nw); p.update(kw)
    s = Solver(p)
    with torch.no_grad():
        Y, g = s.solve(torch.zeros(s.M + 1), forcing(s), iters=60, tol=1e-13)
    return s, Y.numpy(), g.numpy()

# ------------------------------------------------------------------ 1a. convergence in time
t0 = time.time()
hs = [0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125]
sols = {h: run(h=h) for h in hs}
href = hs[-1]; _, Yref, _ = sols[href]
step = lambda h: int(round(0.1 / h))                      # common time points t = 0.1 k
Yr = Yref[::step(href)]
time_rows = []
for h in hs[:-1]:
    Y = sols[h][1][::step(h)]
    e_sup = np.max(np.linalg.norm(Y - Yr, axis=1)) / np.max(np.linalg.norm(Yr, axis=1))
    e_l2 = np.linalg.norm(Y - Yr) / np.linalg.norm(Yr)
    time_rows.append([h, h * tscale, float(e_sup), float(e_l2)])
for i in range(1, len(time_rows)):
    time_rows[i].append(math.log2(time_rows[i - 1][2] / time_rows[i][2]))
time_rows[0].append(float("nan"))
out["time_convergence"] = dict(columns=["h", "h_seconds", "rel_sup_error", "rel_L2_error", "observed_order"],
                               reference_h=href, rows=time_rows)
print("time convergence", time_rows, round(time.time() - t0, 1), flush=True)

# ------------------------------------------------------------------ 1b. convergence in space
Ns = [4, 8, 16, 32]
hsp = 0.0125
spc = {N: run(h=hsp, N=N, Nw=256) for N in Ns}
Yn_ref = spc[Ns[-1]][1]
space_rows = []
for N in Ns[:-1]:
    Y = np.zeros_like(Yn_ref); Y[:, :N] = spc[N][1]
    e_sup = np.max(np.linalg.norm(Y - Yn_ref, axis=1)) / np.max(np.linalg.norm(Yn_ref, axis=1))
    space_rows.append([N, float(e_sup)])
out["space_convergence"] = dict(columns=["N", "rel_sup_error_vs_N32"], h=hsp, rows=space_rows)
print("space convergence", space_rows, flush=True)

# ------------------------------------------------------------------ 2. fPINN against the converged reference
d = np.load(os.path.join(ROOT, "results", "fpinn_elcentro_data.npz"))
xp = d["xp"]                                             # fPINN on the h = 0.05 grid, 64 points
Wq = sol2.S["Wq"]
Yp = xp @ Wq.T                                           # modal coefficients (N = 8)
Yabm = sols[0.05][1]                                     # ABM with h = 0.05
Yfine = Yref[::int(round(0.05 / href))]                  # ABM with h = 0.003125, sampled at the same times
Yfine32 = np.zeros((Yfine.shape[0], 32))
_, Y32, _ = run(h=0.003125, N=32, Nw=256)
Yfine32 = Y32[::int(round(0.05 / 0.003125))]
pad = lambda Y: np.hstack([Y, np.zeros((Y.shape[0], 32 - Y.shape[1]))])
rel = lambda A, B: float(np.linalg.norm(A - B) / np.linalg.norm(B))
out["fpinn_accuracy"] = dict(
    fpinn_vs_abm_h005=rel(Yp, Yabm),
    fpinn_vs_abm_fine=rel(Yp, Yfine),
    abm_h005_vs_abm_fine=rel(Yabm, Yfine),
    fpinn_vs_converged_N32=rel(pad(Yp), Yfine32),
    abm_h005_vs_converged_N32=rel(pad(Yabm), Yfine32))
print("fPINN accuracy", out["fpinn_accuracy"], flush=True)

# ------------------------------------------------------------------ 3. full-time-domain fPINN loss
M = sol2.M
Aful = sol2.A.clone(); Aful[torch.arange(M + 1), torch.arange(M + 1)] = sol2.Ad
Ypt = torch.tensor(Yp); v1 = forcing(sol2)
y0 = Ypt[0]; g_p = y0 - sol2.phi_n
idx = torch.arange(M + 1) - sol2.dtau
Ydel = torch.where((idx >= 0).unsqueeze(1), Ypt[idx.clamp(min=0)], y0)
Jn = (-P2["zeta"] * torch.tanh(Ydel @ sol2.Phi.T)) @ sol2.Wq.T
F = -sol2.om2 * Ypt + (Jn - sol2.m_n * v1.unsqueeze(1)) / sol2.E
R = Ypt - y0 - Aful @ F
res_part = float(torch.mean(R ** 2)); nl_part = float(torch.sum((g_p - P2["c_g"] * Ypt[sol2.i1] / sol2.E) ** 2))
kw_ = M // 5
win = [float(torch.mean(R[j * kw_:(j + 1) * kw_ + 1] ** 2)) for j in range(5)]
out["fpinn_full_loss"] = dict(full_domain_loss=res_part + nl_part, residual_part=res_part, nonlocal_part=nl_part,
                              window_residuals=win, last_window_loss_reported=2.7893656025926794e-09)
print("full loss", out["fpinn_full_loss"], flush=True)

# ------------------------------------------------------------------ 4. nonlocal condition and control
_cz = os.path.join(ROOT, "results", "elcentro_controls.npz")
if os.path.exists(_cz):                                   # written by numerics_elcentro.py
    _c = np.load(_cz); mus = {"mu_NN": torch.tensor(_c["mu_nn"]), "mu_direct": torch.tensor(_c["mu_direct"])}
else:                                                     # data shipped with the repository
    rows = list(csv.DictReader(open(os.path.join(ROOT, "data", "processed", "elcentro_control_timeseries.csv"))))
    mus = {k: torch.tensor([float(r[k]) for r in rows]) for k in ["mu_NN", "mu_direct"]}
v1 = torch.tensor(eq1)
with torch.no_grad():
    Yb, gb = sol2.solve(torch.zeros(M + 1), v1)
    Jb = float(sol2.cost(Yb, torch.zeros(M + 1)))
    hist_norm = float(torch.linalg.norm(sol2.phi_n + gb))
    nl = dict(J_uncontrolled=Jb, history_norm_uncontrolled=hist_norm)
    for k, mu in mus.items():
        Y, g = sol2.solve(mu, v1)
        dk = sol2.dkap; u = torch.cat([torch.zeros(dk), mu[:M + 1 - dk]])
        Yfix_ = sol2.solve_given_g(gb, u, v1)              # same initial history as the uncontrolled case
        nl[k] = dict(J=float(sol2.cost(Y, mu)), J_same_history=float(sol2.cost(Yfix_, mu)),
                     rel_change_history=float(torch.linalg.norm(g - gb) / hist_norm),
                     reduction=1 - float(sol2.cost(Y, mu)) / Jb,
                     reduction_same_history=1 - float(sol2.cost(Yfix_, mu)) / Jb)
out["nonlocal_effect"] = nl
print("nonlocal", nl, flush=True)

if RUN_LOCAL_IC:
    p0 = dict(P2); p0["c_g"] = 0.0
    s0 = Solver(p0)
    with torch.no_grad():
        Y0, _ = s0.solve(torch.zeros(M + 1), v1); J0 = float(s0.cost(Y0, torch.zeros(M + 1)))
    t1_ = time.time()
    mu0, Yc0, gc0, Jc0, hc0 = train_direct(s0, v1)
    out["local_initial_condition"] = dict(c_g=0.0, J_uncontrolled=J0, J_direct=Jc0, reduction=1 - Jc0 / J0,
                                          peak_unc=float(np.abs(Y0.numpy() @ s0.S["Phi"].T).max()),
                                          peak_direct=float(np.abs(Yc0.numpy() @ s0.S["Phi"].T).max()),
                                          time=time.time() - t1_)
    print("c_g = 0", out["local_initial_condition"], flush=True)

# ------------------------------------------------------------------ 5. time to reach a target cost
if os.path.exists(_cz):
    hn, hd = _c["hist_nn"], _c["hist_direct"]
else:
    tc = list(csv.DictReader(open(os.path.join(ROOT, "data", "processed", "elcentro_control_training_cost.csv"))))
    hn = np.array([float(r["J_NN"]) for r in tc if r["J_NN"] != ""])
    hd = np.array([float(r["J_direct"]) for r in tc if r["J_direct"] != ""])
R0 = json.load(open(os.path.join(ROOT, "results", "results_elcentro.json")))["oc"]
spi_nn, spi_d = R0["time_nn"] / len(hn), R0["time_direct"] / len(hd)
Jbest = min(hn.min(), hd.min())
tt = []
for f in [1.10, 1.05, 1.02, 1.01]:
    target = f * Jbest
    i_nn = int(np.argmax(hn <= target)) + 1 if np.any(hn <= target) else None
    i_d = int(np.argmax(hd <= target)) + 1 if np.any(hd <= target) else None
    tt.append(dict(target_factor=f, target=target, iters_nn=i_nn, iters_direct=i_d,
                   time_nn=None if i_nn is None else i_nn * spi_nn, time_direct=None if i_d is None else i_d * spi_d))
out["time_to_target"] = dict(J_best=float(Jbest), sec_per_iter_nn=spi_nn, sec_per_iter_direct=spi_d, rows=tt)
print("time to target", out["time_to_target"], flush=True)

# ------------------------------------------------------------------ 6. computed controls on finer time grids
fine = {}
for hf in [0.05, 0.025, 0.0125]:
    pf = dict(P2); pf["h"] = hf; sf = Solver(pf); vf = forcing(sf)
    with torch.no_grad():
        Yu, _ = sf.solve(torch.zeros(sf.M + 1), vf); Ju = float(sf.cost(Yu, torch.zeros(sf.M + 1)))
        row = dict(J_uncontrolled=Ju, peak_unc=float(np.abs(Yu.numpy() @ sf.S["Phi"].T).max()))
        for k, mu in mus.items():
            muf = torch.tensor(np.interp(sf.t, sol2.t, mu.numpy()))
            Yc, _ = sf.solve(muf, vf)
            Jc = float(sf.cost(Yc, muf))
            row[k] = dict(J=Jc, reduction=1 - Jc / Ju, peak=float(np.abs(Yc.numpy() @ sf.S["Phi"].T).max()))
    fine[str(hf)] = row
out["controls_on_finer_grids"] = fine
print("finer grids", fine, flush=True)

json.dump(out, open(os.path.join(ROOT, "results", "revision_checks.json"), "w"), indent=2)
print("done")
