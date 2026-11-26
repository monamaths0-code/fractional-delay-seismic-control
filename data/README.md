# Data description

All files are plain CSV with a header row. Values are dimensionless model units unless the column name says otherwise (`t_seconds`, `_g`).

## raw/

| File | Content |
|---|---|
| `elcentro_NS.txt` | 1940 El Centro NS ground acceleration. Column 1: time [s]; column 2: acceleration [g]. 2688 rows, Δt = 0.02 s. Source: Vibrationdata (T. Irvine), http://www.vibrationdata.com/elcentro.htm. |

## processed/

**Time scaling.** One model time unit equals 0.532 s. This follows from a first natural period T₁ = 1.5 s and ω₁ = 2.229. The model interval [0, 15] therefore covers the first 7.98 s of the record. The record is scaled so that its peak equals 4 model units.

### Tables of the article

| File | Article | Columns |
|---|---|---|
| `table_theory_constants.csv` | Section 6.2 | ς, ‖E⁻¹‖, Υ_E, Υ_P, γ₁, γ₂, ‖D‖, Λ_ϖ for the two settings |
| `table_fpinn_accuracy.csv` | Table 3 | relative L² error, max error, error in g, final loss, peak |
| `table_optimal_control_elcentro.csv` | Table 4 | cost J, peak displacement over the whole structure, RMS norm, max control, CPU time |
| `table_control_delay_elcentro.csv` | Table 7 | control delay κ, optimal cost, peak |
| `table_robustness_elcentro.csv` | Table 9 | ε, ‖ϖ¹ − ϖ²‖, state gap, ratio, bound Λ_ϖ‖ϖ¹ − ϖ²‖ |
| `table_value_function_elcentro.csv` | Corollary 5.5 test | V(ϖ¹), V(ϖ²), difference, disturbance gap |
| `table_order_peaks.csv` | Figure 4(b) | ϱ (17 values in [1.1, 1.9]), peak midpoint displacement, L² energy norm |

### Curves of the figures

| File | Article | Columns |
|---|---|---|
| `order_midpoint_curves.csv` | Figure 4(a), (c) | t, ξ(t, 1/2) for each of the 17 orders |
| `elcentro_scaled_input.csv` | Figure 6(a) | model time, seconds, recorded a_g [g], scaled ϖ |
| `elcentro_uncontrolled_timeseries.csv` | Figures 5(a), 6(b) | time, ϖ, midpoint displacement, ‖ξ(t)‖_{L²} |
| `elcentro_control_timeseries.csv` | Figure 6(b), (c) | time, ϖ, midpoint displacement and L² norm (uncontrolled / NN / direct), controls μ_NN and μ_direct |
| `elcentro_control_training_cost.csv` | Figure 7(a) | iteration, cost J for the NN open-loop solver and the direct method |
| `fpinn_elcentro_field.csv` | Figure 5(b) | t, w, ξ_ABM, ξ_fPINN, absolute error on the 301 × 64 grid |
| `fpinn_elcentro_training_loss.csv` | Figure 5(c) | iteration, fPINN loss |

The midpoint is w = 0.492 (the grid point closest to 1/2 on the 64-point midpoint grid).

### Revision checks

| File | Article | Content |
|---|---|---|
| `../results/revision_checks.json` | Section 6.6, Tables 5, 6, 8 | grid convergence in time and space, fPINN error against the converged reference, full-domain fPINN loss, effect of the nonlocal condition, time to reach a target cost |
