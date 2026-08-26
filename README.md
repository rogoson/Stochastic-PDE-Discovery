# Stochastic-PDE-Discovery
Dissertation code for "Collinearity as a Barrier to Sparse Discovery of Stochastic PDEs: Mechanisms, Mitigations, and Limits".
This repository extends and corrects the implementation of Mathpati et al. (2024) — see **Attribution** below.

## Attribution

This project builds on the implementation accompanying:

> Mathpati, Y.C., Tripura, T., Nayek, R., and Chakraborty, S.,
> "Discovering stochastic partial differential equations from limited data
> using variational Bayes inference," arXiv:2306.15873 (2023).

The following components are adapted or corrected from their original code:
- `SPDE.py`: the variational Bayes core (`Variational_Bayes_Code`, `run_VB2`,
  `build_linear_system`, `FiniteDiff`) — retained with corrections documented
  in the accompanying dissertation. The original authors' attribution header is
  preserved at the top of this file.

The following are original contributions:
- `DataGenerator.py` / `PredictionDataGenerator.py`: SDE ensemble generation
  via DifferentialEquations.jl for all five problems (Heat, Allen-Cahn, Nagumo,
  KdV, Burgers), including the multiplicative-noise Burgers formulation.
- `MasterNotebook.ipynb`: the full identification and prediction pipeline,
  sweep infrastructure, and result logging. Some code adapts from 1D_{problem}.ipynb 
  files in the code of Mathpati et al. (2024).
- `ormerodAlgorithm2` in `SPDE.py`: two-phase coordinate-ascent model
  selection wrapper, including parallelised ELBO evaluation.
- `SolverDiagnostics.py`: KM diffusion convergence study.
- All figures in `worthwhileImages/` and result CSVs in `results/`.

The original repository did not carry a signed license file; attribution is
provided above in accordance with academic practice.
