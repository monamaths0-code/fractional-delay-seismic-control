# Delayed Sobolev-type fractional control under seismic disturbance: code and data

This repository contains the Python code and the numerical data for the article

> M. Siddiq and A. U. K. Niazi, *Optimal Control of Delayed Sobolev-Type Fractional Evolution Systems of Order (1,2) under Seismic Disturbances with Neural Network Validation* (submitted, 2026).

Every figure and table in Section 6 of the article can be reproduced with these scripts.

Repository: https://github.com/monamaths0-code/fractional-delay-seismic-control

## The model problem

The state ξ(t, w) models the displacement of a one-dimensional viscoelastic structure, w ∈ (0, 1), with fixed ends:

```
ᶜD^ϱ_t (I − a ∂_ww) ξ = c² ∂_ww ξ − ζ tanh(ξ(t − τ)) + b(w) μ(t − κ) − m(w) ϖ(t),    1 < ϱ < 2
ξ(θ) = φ(θ) + g(ξ),  θ ∈ [−τ, 0],      g(ξ) = c_g E⁻¹ ξ(t₁),      ξ_t(0) = 0
```

| Symbol | Meaning | Value |
|---|---|---|
| ϱ | order of the Caputo derivative | 1.6 |
| a, c | Sobolev and wave parameters | 0.1, 1.0 |
| ζ | nonlinear damping | 0.1 (El Centro: 0.01) |
| τ, κ | state delay, control delay | 0.2, 0.1 |
| c_g, t₁ | nonlocal constant and time | 0.1, 2.0 |
| Θ | final time | 4 (El Centro: 15) |
| ϖ(t) | ground acceleration (disturbance) | 0.5 sin 2t, or El Centro 1940 NS |
| μ(t) | actuator force (control) | computed |

Numerical methods:

- **Space:** spectral Galerkin method with 8 sine modes.
- **Classical time scheme:** Adams–Bashforth–Moulton (ABM) predictor–corrector.
- **fPINN:** fractional physics-informed neural network in integral form. The El Centro case uses 5 time windows (time marching).
- **Optimal control:** a neural network controller μ = ū tanh(N(t/Θ)), compared with a direct (grid-based) method.

## Folder layout

```
code/                      Python scripts
  numerics.py              solver, smooth-forcing study, fPINN (Figures 2, 3; Table 3 first column)
  fig0_model.py            Figure 1 (model sketch, profiles, natural frequencies)
  fig3_order.py            Figure 4 (effect of the order ϱ) + its CSV data
  numerics_elcentro.py     El Centro study: optimal control, delay, robustness (Figures 6, 7; Tables 4, 5, 6)
  fpinn_elcentro.py        time-marching fPINN on El Centro (Figure 5; Table 3 second column)
  export_data.py           writes all CSV files in data/processed/
data/raw/                  recorded El Centro 1940 NS ground acceleration
data/processed/            CSV data behind every figure and table (see data/README.md)
results/                   figures (PDF, PNG) and raw results (JSON, NPZ)
```

## How to run

```bash
git clone https://github.com/monamaths0-code/fractional-delay-seismic-control.git
cd fractional-delay-seismic-control
pip install -r requirements.txt
cd code
python numerics.py            # about 18 min on a CPU
python fig0_model.py          # a few seconds
python fig3_order.py          # about 10 s
python numerics_elcentro.py   # about 30 min
python fpinn_elcentro.py      # about 4 min
python export_data.py --with-control   # about 10 min; without the flag, about 10 s
```

**Order matters.** `numerics.py` also draws early versions of Figures 4, 6 and 7. They are replaced by the final versions from `fig3_order.py` and `numerics_elcentro.py`. So run the scripts in the order above.

All runs use double precision and fixed random seeds. The results may differ in the last digits on other hardware or with other PyTorch versions.

## Main results

| Quantity | Value |
|---|---|
| fPINN relative L² error, smooth forcing | 0.83 % |
| fPINN relative L² error, El Centro (5 windows) | 0.51 % |
| Cost reduction with optimal control (El Centro) | 82.5 % |
| Peak displacement over the structure: uncontrolled → NN control | 0.196 → 0.069 |
| Cost for control delay κ = 0, 0.1, 0.3 | 0.01102, 0.01105, 0.01114 |
| Largest robustness ratio vs. theoretical gain Λ_ϖ | 0.049 < 26.8 |

## Data source

The El Centro record (`data/raw/elcentro_NS.txt`) is the north–south component of the 1940 Imperial Valley earthquake, recorded at El Centro. It was obtained from Vibrationdata (T. Irvine), http://www.vibrationdata.com/elcentro.htm. It contains 2688 samples at Δt = 0.02 s, with a PGA of 0.349 g at 2.12 s.

## License and citation

The code is released under the MIT License (see `LICENSE`). If you use it, please cite the article (see `CITATION.cff`).

## Contact

Mamoona Siddiq, Department of Mathematics and Statistics, The University of Lahore, Sargodha, Pakistan. Email: monamaths0@gmail.com, ORCID: 0009-0001-9940-6144.
