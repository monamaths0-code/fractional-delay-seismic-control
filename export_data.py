"""Export all numerical data behind the tables and figures of the paper as CSV files.

Usage
    python export_data.py                 # tables + El Centro input + uncontrolled response + fPINN data (fast)
    python export_data.py --with-control  # also retrains the NN and direct controllers (about 10 min on a CPU)
                                          # and writes the controlled time histories

Output folder: ../data/processed/
"""
import os, sys, json, csv
import numpy as np, torch

here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(here, ".."))
RES = os.path.join(ROOT, "results")
DST = os.path.join(ROOT, "data", "processed"); os.makedirs(DST, exist_ok=True)

def write(name, header, rows):
    with open(os.path.join(DST, name), "w", newline="") as f:
        w = csv.writer(f); w.writerow(header)
        for r in rows: w.writerow([f"{x:.10g}" if isinstance(x, (float, np.floating)) else x for x in r])
    print("wrote", name)

# ------------------------------------------------------------------ 1. tables from the JSON results
R1 = json.load(open(os.path.join(RES, "results.json")))              # smooth / synthetic study
R2 = json.load(open(os.path.join(RES, "results_elcentro.json")))     # El Centro study
R3 = json.load(open(os.path.join(RES, "results_fpinn_elcentro.json")))

rows = []
for k in ["varsigma", "Einv", "UpsE", "UpsP", "gamma1", "gamma2", "Dnorm", "Lambda_varpi"]:
    rows.append([k, R1["theory"][k], R2["theory"][k]])
write("table_theory_constants.csv", ["constant", "main_example_Theta4_zeta0.1", "elcentro_Theta15_zeta0.01"], rows)

f1 = R1["fpinn"]
write("table_fpinn_accuracy.csv",
      ["case", "rel_L2_error", "max_abs_error", "g_error", "final_loss", "peak_displacement"],
      [["smooth_forcing_single_network", f1["rel_l2"], f1["max_abs"], f1["g_err"], f1["final_loss"], f1["peak"]],
       ["elcentro_time_marching_5_windows", R3["rel_l2"], R3["max_abs"], R3["g_err"], R3["final_loss"], R3["peak"]]])

# table_order_peaks.csv and order_midpoint_curves.csv are written by fig3_order.py

oc = R2["oc"]
write("table_optimal_control_elcentro.csv", ["quantity", "uncontrolled", "NN_controller", "direct_method"],
      [["cost_J", oc["J_uncontrolled"], oc["J_nn"], oc["J_direct"]],
       ["peak_displacement", oc["peak_unc"], oc["peak_nn"], oc["peak_direct"]],
       ["rms_L2_norm", oc["rms_unc"], oc["rms_nn"], oc["rms_direct"]],
       ["max_abs_control", 0.0, oc["umax_nn"], oc["umax_direct"]],
       ["cpu_time_s", "", oc["time_nn"], oc["time_direct"]]])

write("table_control_delay_elcentro.csv", ["kappa", "optimal_cost_J", "peak_displacement"],
      [[float(k), v["J"], v["peak"]] for k, v in sorted(R2["delay"].items(), key=lambda z: float(z[0]))])

write("table_robustness_elcentro.csv",
      ["epsilon", "disturbance_gap_L2", "state_gap_sup", "ratio", "theoretical_bound_Lambda_times_gap"],
      [[e, d, s, s / d, R2["theory"]["Lambda_varpi"] * d] for e, d, s in R2["robustness"]["rows"]])

v = R2["value"]
write("table_value_function_elcentro.csv", ["V1", "V2", "abs_difference", "disturbance_gap_L2"],
      [[v["V1"], v["V2"], v["diff"], v["dw"]]])

# ------------------------------------------------------------------ 2. fPINN field data (El Centro)
d = np.load(os.path.join(RES, "fpinn_elcentro_data.npz"))
xr, xp, hist = d["xr"], d["xp"], d["hist"]

# ------------------------------------------------------------------ 3. El Centro setting (fast part)
src = open(os.path.join(here, "numerics_elcentro.py")).read()
__file__ = os.path.join(here, "numerics_elcentro.py")
exec(src[:src.index("v1 = torch.tensor(eq1)")])
ts = tg2 * tscale; wgrid = S2["w"]; iw = int(np.argmin(np.abs(wgrid - 0.5)))

write("elcentro_scaled_input.csv", ["t_model", "t_seconds", "ag_recorded_g", "varpi_model_units"],
      [[tg2[k], ts[k], eq1[k] * pga / PGA_ND, eq1[k]] for k in range(len(tg2))])

rows = []
for k in range(xr.shape[0]):
    for i in range(xr.shape[1]):
        rows.append([tg2[k], ts[k], wgrid[i], xr[k, i], xp[k, i], abs(xr[k, i] - xp[k, i])])
write("fpinn_elcentro_field.csv", ["t_model", "t_seconds", "w", "xi_ABM", "xi_fPINN", "abs_error"], rows)
write("fpinn_elcentro_training_loss.csv", ["iteration", "loss"], hist.tolist())

v1 = torch.tensor(eq1); mu0 = torch.zeros(M2 + 1)
with torch.no_grad():
    Yb, gb = sol2.solve(mu0, v1)
X = lambda Y: (Y @ sol2.Phi.T).numpy()
xb = X(Yb); nb = sol2.l2norm_t(Yb).numpy()

write("elcentro_uncontrolled_timeseries.csv", ["t_model", "t_seconds", "varpi", "xi_mid", "xi_L2norm"],
      [[tg2[k], ts[k], eq1[k], xb[k, iw], nb[k]] for k in range(len(tg2))])
if "--with-control" not in sys.argv:
    sys.exit(0)

# ------------------------------------------------------------------ 4. controlled responses (slow)
mu_nn, Y_nn, _, J_nn, h_nn = train_nn(sol2, v1)
mu_d, Y_d, _, J_d, h_d = train_direct(sol2, v1)
xn, xd = X(Y_nn), X(Y_d)
nn_, nd = sol2.l2norm_t(Y_nn).numpy(), sol2.l2norm_t(Y_d).numpy()
write("elcentro_control_timeseries.csv",
      ["t_model", "t_seconds", "varpi", "xi_mid_uncontrolled", "xi_mid_NN", "xi_mid_direct",
       "L2norm_uncontrolled", "L2norm_NN", "L2norm_direct", "mu_NN", "mu_direct"],
      [[tg2[k], ts[k], eq1[k], xb[k, iw], xn[k, iw], xd[k, iw], nb[k], nn_[k], nd[k],
        float(mu_nn[k]), float(mu_d[k])] for k in range(len(tg2))])
L = max(len(h_nn), len(h_d))
write("elcentro_control_training_cost.csv", ["iteration", "J_NN", "J_direct"],
      [[i + 1, h_nn[i] if i < len(h_nn) else "", h_d[i] if i < len(h_d) else ""] for i in range(L)])
print("J_NN", J_nn, "J_direct", J_d)
