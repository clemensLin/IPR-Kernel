# IPR Kernel

Code and data for *A Basis-Dependent Kernel for Learning Quantum Phases of Matter*. The repository regenerates every dataset, balanced-accuracy table and figure of the manuscript.

The pre-revision state of the repository, including the 2D TFIM data, the rotated-basis IPRs and the original notebooks, is preserved on the `old` branch.

## Layout

```text
data/
  raw/                 committed generator output
    tfim_1d/           TFIM ground states, L=<L>x1/{hs,ipr_x,ipr_z}.csv
    xxz/               XXZ ground states, L=<L>/
    xxz_field/         XXZ chain in a longitudinal field, L=14/
    tfim_thermal/      TFIM Gibbs states, ipr/ and hs/
  processed/           dense resampled data (gitignored, from notebooks/process_data.ipynb)
data_generation/       one generator per raw dataset (Python and Julia)
src/ipr_kernel/        analysis package: dataset access, Gram matrices, finite-shot noise, SVM experiments
notebooks/
  process_data.ipynb   raw -> processed (smoothing splines on log10 IPR, dense grids)
  tfim_svm.ipynb       Tables II and III
  xxz_svm.ipynb        Table IV
  thermal_svm.ipynb    Table V and the cached thermal SVM sweep
figures/
  make_figures.py      every figure of the manuscript
  tables/              balanced-accuracy tables written by the SVM notebooks
tests/                 consistency checks for the generators, mirroring data_generation/
```

## Pipeline

```text
data_generation/  ->  data/raw/  ->  notebooks/process_data  ->  data/processed/
                                                                     |
                      figures/tables/*.json  <-  notebooks/*_svm  <--+
                                |                                    |
                                +---------->  figures/make_figures.py
```

Each stage can run on its own: `data/raw/` and `figures/tables/` are committed, so the figures only need `data/processed/`, which `process_data.ipynb` rebuilds from the raw data.

## Installation

Two environments are pinned, one per language.

```bash
pip install -r requirements.txt          # installs src/ipr_kernel in editable mode
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

`Manifest.toml` pins the full transitive closure, so `instantiate` reproduces the package set the data was generated with (Julia 1.12.0, ITensors 0.6.23, ITensorMPS 0.2.6). `ITensors` and `ITensorMPS` are pinned exactly in `[compat]` because they determine the numerics; `CairoMakie` affects only plots and is left on a range.

## Reproducing the paper

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/process_data.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/tfim_svm.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/xxz_svm.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/thermal_svm.ipynb   # about three hours
python figures/make_figures.py                                                   # writes figures/output/
```

`python figures/make_figures.py --only fig_tfim fig_xxz_field` draws a subset. Without the cached thermal sweep (`data/processed/thermal_svm_results.pkl`), the two thermal SVM figures run the sweep themselves.

## Data generation

| Raw folder | Script | Method |
| --- | --- | --- |
| `tfim_1d/L=<L>x1/` | `generate_data_tfim.jl` | DMRG with perfect sampling (ITensors) |
| `xxz/L=<L>/` | `generate_data_heisenberg.py` | Exact diagonalization (netket) |
| `xxz_field/L=14/` | `generate_data_heisenberg_field.py` | Exact diagonalization in two bases |
| `tfim_thermal/ipr/` | `generate_data_tfim_thermal_ipr_kernel.jl` | Trotterized MPO Gibbs states, MPO-diagonal sampling |
| `tfim_thermal/hs/` | `generate_data_tfim_thermal_hs_kernel.jl` | Trotterized MPO Gibbs states, exact MPO inner products |

Every script is self-contained, writes CSV, accepts `--sizes` and `--outdir`, and defaults to writing into the corresponding `data/raw/` folder. Cost grows steeply with system size, so start with one small size and verify before scaling up. The generators also write a few outputs the paper does not use (overview plots, correlators, magnetizations, rotated-basis IPRs, 2D TFIM via `--geometry 2d`); these are not committed.

### Parameters

All datasets use `J = 1` and periodic boundary conditions.

**TFIM ground states**, `H = -J sum_i Z_i Z_{i+1} - h sum_i X_i`, `L = 3 ... 100`, `h = 0.0, 0.2, ..., 4.0`. DMRG uses a random initial MPS with `linkdims = 10`, 10 sweeps, `maxdim = 400`, `cutoff = 1e-10`, `noise = 1e-8`; IPRs come from 10 000 perfect samples.

**XXZ**, `H = sum_i [X_i X_{i+1} + Y_i Y_{i+1} + Delta Z_i Z_{i+1}]`, zero field, `L = 6, 8, ..., 24`, `Delta = 0.0, 0.1, ..., 5.0`. `IPR Z` is the ground state in the computational basis, `IPR X` after a global Hadamard, `IPR XY` in the momentum-occupation basis of the XX model. `IPR XY = 1` at `Delta = 0` because that basis is then exact. `energies_old.csv` and `energies_new.csv` hold the ground-state energies of the two diagonalizations (Pauli-string and Fourier form), which agree.

**XXZ in a field**, `L = 14`, with `Delta` and `h` each spanning `-5` to `5` in 51 steps. The ground state is found twice per grid point: from the Pauli-string Hamiltonian, and from its Jordan-Wigner/Fourier form restricted to the particle-number sector implied by the measured magnetization. The two spectra must agree, and the script verifies this at every point.

**TFIM thermal**, both kernels with `L = 10, 20, ..., 100`, `h = 0.0, 0.1, ..., 4.0`, and imaginary-time step `dtau = 0.01`. The IPR kernel covers `beta = 0.0, 0.1, ..., 6.0` and uses `maxdim = 400`, `cutoff = 1e-8` and 2000 Monte Carlo samples per point. The HS kernel covers `beta = 0.0, 0.1, ..., 2.0` (every downstream analysis uses `beta <= 2`) and uses `maxdim = 200`, `cutoff = 1e-12`; the largest bond dimension reached is 143 (`L = 50`, `beta = 2`). Kernel entries are exact MPO inner products. Run it with one worker process per field, e.g. `--workers 41`.

### Conventions that are easy to misread

All of the following are deliberate, and each was confirmed against independent exact diagonalization.

- **Both thermal datasets are periodic.** The HS kernel was originally generated with open boundaries (`N-1` bonds) and has been regenerated with periodic ones so that both kernels describe the same states. At `beta = 2` the two boundary conditions differ by up to 0.17 at `L = 10` and 0.04 at `L = 100`.
- **The field sweep is written with Pauli operators, not spin-1/2 operators.** Its field is therefore twice the Bethe-ansatz field, `h_data = 2 h_BA`, so saturation sits at `2 (1 + Delta)` instead of `1 + Delta`. `figures/make_figures.py` draws the Bethe-ansatz lines in the same convention.
- **The HS kernel's `beta = 0` slice carries `Tr[Id^2] = 2^L`, not 1.** Every `beta > 0` slice is normalized to `Tr[rho^2] = 1`, making the kernel diagonal exactly one there. The offset at `beta = 0` is reproduced deliberately.
- **The field sweep bypasses the Fourier diagonalization in three regions**, reusing the real-space state so that the basis contrast is exactly zero: an empty fermion sector, a full one, and the line `h = 0` with `Delta < -1`, where the ferromagnetic ground state is degenerate.

## Processed data

`notebooks/process_data.ipynb` fits a smoothing spline to `log10(IPR)` for every size and basis, evaluates it on a dense grid (4000 points in `h` for the TFIM, 5000 in `Delta` for the XXZ chain), and exponentiates back. For the thermal data it resamples the IPRs of `L = 20, 50, 100` and `beta = 0.1, ..., 2.0`, and interpolates each raw HS Gram matrix bilinearly onto the same 4000-point grid. The HS output is about 3.6 GB of `float32` `.npy` files, which is why `data/processed/` is not committed.

When bit-level fidelity matters, read the raw CSVs with `pandas.read_csv(..., float_precision="round_trip")`. The default parser is off by one unit in the last place on most values; the files themselves are exact.

## Tests

`tests/data_generation/` holds one consistency suite per generator. Each compares the generator against an independent dense Kronecker-product construction of the same model, diagonalized exactly, rather than against itself. Run them all with:

```bash
bash tests/run_tests.sh
```

Monte Carlo quantities are judged against the exact standard error of the estimator rather than a fixed tolerance. Perfect sampling draws `x` from `p(x)` and averages `p(x)`, so `Var = sum_x p^3 - IPR^2`.

Several checks exist specifically to catch conventions that would otherwise fail silently. The thermal IPR and HS-kernel suites both require their data to match a periodic reference and to disagree with an open one. The zero-field suite builds the Jordan-Wigner momentum basis from scratch, and separately verifies that the Fourier operator reproduces the exact spectrum in every particle-number sector, not only at half filling.

That last check matters because on a ring the Jordan-Wigner string gives the wrapping bond a factor `(-1)^N_f`, so the fermions are periodic or antiperiodic according to the parity of the fermion *number*. The half-integer momentum offset implements the antiperiodic case, and both Fourier operators select it from `n_fermions % 2`.

The following reproduce the committed data to machine precision:

| Dataset | Agreement |
| --- | --- |
| XXZ, zero field | `6e-15` relative, full grid at `L = 6` |
| XXZ in a field | `2e-15`, representative points at `L = 14` |
| TFIM thermal, HS kernel | `2.2e-5` against periodic exact diagonalization over the full grid at `L = 10` (Trotter error) |
| TFIM thermal, IPR kernel | two ulps at fixed seed |

**The TFIM ground-state IPRs are the one dataset that does not reproduce exactly.** There are two causes, both understood:

1. Monte Carlo noise. IPRs come from 10 000 samples, giving 1 to 3 per cent scatter. The seed of the original runs was not recorded. `--seed` is exposed for new runs, and with a fixed seed the pipeline is bit-for-bit deterministic.
2. Z2 degeneracy in the ordered phase. The symmetric and symmetry-broken ground states are nearly degenerate at small `h`, and which one DMRG finds depends on the random initial state. This affects `h = 0.4` and `h = 0.6`. The energy and `<sum_i X_i>` are unaffected, so the effect is invisible in the observables one would normally use to check convergence.

This attribution was tested, not assumed: across 168 comparisons at small sizes and in both bases, every committed value lies inside the range reachable by `cos(theta) psi_0 + sin(theta) psi_1` within the near-degenerate doublet, judged against the exact standard error of the sampling estimator.

One further known artifact: in the `L = 14` field sweep, 24 of 2601 grid points are not symmetric under `h -> -h`, although spin-flip symmetry requires it. Those points are exactly the saturation line `h = +/- 2 (1 + Delta)`, where the two lowest levels cross and the solver returns an arbitrary member of the degenerate pair. One grid step away the gap is of order `0.2` and the data is symmetric.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
