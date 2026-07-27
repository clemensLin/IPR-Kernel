# IPR Kernel

Research code and notebooks for analyzing inverse participation ratio (IPR) based kernels and related kernel-SVM workflows for quantum spin models.

## Repository Layout

- `data_generation/`: scripts that produce every dataset under `Data/Raw/`.
- `scripts/`: reusable Python helpers for data loading, plotting, Gram-matrix construction, finite-shot noise, and SVM experiments.
- `Paper Analysis/`: notebooks and generated figures/tables used in the paper.
- `Data/`: raw datasets and processed datasets.
- `tests/`: consistency checks for the generation scripts, mirroring `data_generation/`.

## Pipeline

```text
data_generation/  ->  Data/Raw/  ->  Data/Processed/  ->  Paper Analysis/
   generators         committed      Data_Processing      figures, tables
                                     .ipynb
```

The stages are independent. `Data/Raw/` is committed, so the analysis notebooks run without regenerating anything.

## Installation

Two environments are pinned, one per language.

```bash
pip install -r requirements.txt
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

`Manifest.toml` pins the full transitive closure, so `instantiate` reproduces the package set the data was generated with (Julia 1.12.0, ITensors 0.6.23, ITensorMPS 0.2.6). `ITensors` and `ITensorMPS` are pinned exactly in `[compat]` because they determine the numerics; `CairoMakie` affects only plots and is left on a range.

Notebooks should be opened from the repository root so relative paths resolve correctly.

## Data generation

| Raw folder | Script | Method |
| --- | --- | --- |
| `Data Heisenberg/L=<L>/` | `generate_data_heisenberg.py` | Exact diagonalization (netket) |
| `Data Heisenberg/L=14_with_field/` | `generate_data_heisenberg_field.py` | Exact diagonalization in two bases |
| `Data TFIM/Data TFIM {1D,2D}/` | `generate_data_tfim.jl` | DMRG with perfect sampling (ITensors) |
| `Data TFIM Thermal/Data IPR Kernel/` | `generate_data_tfim_thermal_ipr_kernel.jl` | Trotterized MPO Gibbs states, MPO-diagonal sampling |
| `Data TFIM Thermal/Data HS Kernel/` | `generate_data_tfim_thermal_hs_kernel.jl` | Trotterized MPO Gibbs states, exact MPO trace |

Every script is self-contained, writes only CSV, accepts `--sizes` and `--outdir`, and defaults to writing into the corresponding `Data/Raw/` folder. Cost grows steeply with system size, so start with one small size and verify before scaling up.

### Parameters

All datasets use `J = 1`.

**Heisenberg**, periodic, `H = sum_i [X_i X_{i+1} + Y_i Y_{i+1} + Delta Z_i Z_{i+1}]`, zero field, `L = 6, 8, ..., 24`, `Delta = 0.0, 0.1, ..., 5.0`. `IPR Z` is the ground state in the computational basis, `IPR X` after a global Hadamard, `IPR XY` in the momentum-occupation basis of the XX model. `IPR XY = 1` at `Delta = 0` because that basis is then exact.

**Heisenberg in a field**, `L = 14`, with `Delta` and `h` each spanning `-5` to `5` in 51 steps. The ground state is found twice per grid point: from the Pauli-string Hamiltonian, and from its Jordan-Wigner/Fourier form restricted to the particle-number sector implied by the measured magnetization. The two spectra must agree, and the script verifies this at every point.

**TFIM ground states**, `H = -J sum_<ij> Z_i Z_j - h sum_i X_i`, periodic in every direction of extent above two. 1D `L = 3 ... 100`; 2D `L = 3 ... 10` on a torus with snake-ordered MPS. `h = 0.0, 0.2, ..., 4.0`, except 2D `8x8` and `10x10`, which use a 41-point grid and carry no `new_iprs.csv`. DMRG uses a random initial MPS with `linkdims = 10`, 10 sweeps, `maxdim = 400`, `cutoff = 1e-10`, `noise = 1e-8`; IPRs come from 10 000 perfect samples. `new_iprs.csv` holds the IPR after `Ry(theta)` applied to every site for `theta = range(0, pi/2, length=5)`, one row per angle and one column per field.

**TFIM thermal**, both kernels on `L = 10, 20, ..., 100`, with `h = 0.0, 0.1, ..., 4.0`, `beta = 0.0, 0.1, ..., 6.0`, and imaginary-time step `dtau = 0.01`. The IPR kernel uses `maxdim = 400`, `cutoff = 1e-8` and 2000 Monte Carlo samples per point. The HS kernel uses `maxdim = 200`, `cutoff = 1e-12`, with kernel traces at `maxdim = 800`.

### Conventions that are easy to misread

All of the following are deliberate, and each was confirmed against independent exact diagonalization.

- **The two thermal datasets use different boundary conditions.** The IPR kernel is periodic (`N` bonds); the HS kernel is open (`N-1` bonds). Checked at `L = 10`: each matches its own boundary condition and is badly wrong under the other.
- **The field sweep is written with Pauli operators, not spin-1/2 operators.** Its field is therefore twice the Bethe-ansatz field, `h_data = 2 h_BA`, so saturation sits at `2 (1 + Delta)` instead of `1 + Delta`. `Field_phase_diagram.ipynb` applies the factor explicitly as `BA_FIELD_SCALE`.
- **`all_correlations.csv` stores `4 <Z_0 Z_{L/2}>`**, not the bare correlator. Consequently 374 of its 510 values lie outside `[-1, 1]`, saturating at `+/- 4` under strong Neel order. Quote it as `4 <Z_0 Z_{L/2}>`, or divide by four. The `zz_correlations` file of the field sweep is unscaled, so the same physical quantity appears in two normalizations in this repository.
- **The HS kernel's `beta = 0` slice carries `Tr[Id^2] = 2^L`, not 1.** Every `beta > 0` slice is normalized to `Tr[rho^2] = 1`, making the kernel diagonal exactly one there. The offset at `beta = 0` is reproduced deliberately.
- **The field sweep bypasses the Fourier diagonalization in three regions**, reusing the real-space state so that the basis contrast is exactly zero: an empty fermion sector, a full one, and the line `h = 0` with `Delta < -1`, where the ferromagnetic ground state is degenerate.

## Reproducibility

`tests/data_generation/` holds one consistency suite per generator. Each compares the generator against an independent dense Kronecker-product construction of the same model, diagonalized exactly, rather than against itself. Run them all with:

```bash
bash tests/run_tests.sh
```

Monte Carlo quantities are judged against the exact standard error of the estimator rather than a fixed tolerance. Perfect sampling draws `x` from `p(x)` and averages `p(x)`, so `Var = sum_x p^3 - IPR^2`.

Several checks exist specifically to catch conventions that would otherwise fail silently. The thermal IPR suite requires its data to match a periodic reference and to disagree with an open one; the HS-kernel suite requires the opposite. The zero-field suite builds the Jordan-Wigner momentum basis from scratch, and separately verifies that the Fourier operator reproduces the exact spectrum in every particle-number sector, not only at half filling.

That last check matters because on a ring the Jordan-Wigner string gives the wrapping bond a factor `(-1)^N_f`, so the fermions are periodic or antiperiodic according to the parity of the fermion *number*. The half-integer momentum offset implements the antiperiodic case, and both Fourier operators now select it from `n_fermions % 2`. An earlier version of the zero-field operator derived the offset from `L` alone, which is that parity evaluated at half filling: correct where it was used, since the zero-field ground state is always at half filling, but wrong by order one in sectors of the other parity.

The following reproduce the committed data to machine precision:

| Dataset | Agreement |
| --- | --- |
| Heisenberg, zero field | `6e-15` relative, full grid at `L = 6` |
| Heisenberg in a field | `2e-15`, representative points at `L = 14` |
| TFIM thermal, HS kernel | `1.9e-15` over all overlapping rows at `L = 10` |
| TFIM thermal, IPR kernel | two ulps at fixed seed |

The two-basis energies of the field sweep agree to `4e-14` across representative points, and the netket Pauli-string Hamiltonian equals a from-scratch dense construction exactly.

**The TFIM ground-state IPRs are the one dataset that does not reproduce exactly.** There are two causes, both understood:

1. Monte Carlo noise. IPRs come from 10 000 samples, giving 1 to 3 per cent scatter. The seed of the original runs was not recorded. `--seed` is exposed for new runs, and with a fixed seed the pipeline is bit-for-bit deterministic.
2. Z2 degeneracy in the ordered phase. The symmetric and symmetry-broken ground states are nearly degenerate at small `h`, and which one DMRG finds depends on the random initial state. In 1D this affects `h = 0.4` and `h = 0.6`; in 2D it affects roughly `h <= 1.6`. The energy and `<sum_i X_i>` are unaffected, so the effect is invisible in the observables one would normally use to check convergence.

This attribution was tested, not assumed. Across 168 comparisons at 1D `L = 10, 12` and 2D `3x3, 4x4`, in both bases, every committed value lies inside the range reachable by `cos(theta) psi_0 + sin(theta) psi_1` within the near-degenerate doublet, judged against the exact standard error of the sampling estimator. No unexplained residual remains: every deviation is sampling noise or the choice of member within a degenerate pair.

One further known artifact: in the `L = 14` field sweep, 24 of 2601 grid points are not symmetric under `h -> -h`, although spin-flip symmetry requires it. Those points are exactly the saturation line `h = +/- 2 (1 + Delta)`, where the two lowest levels cross and the solver returns an arbitrary member of the degenerate pair. One grid step away the gap is of order `0.2` and the data is symmetric.

## Data

`Data/Raw/` is committed. `Data/Processed/` is generated by `Data/Data_Processing.ipynb`, which interpolates the raw curves onto dense grids, and is gitignored because of its size. The largest processed output is the thermal HS-kernel data:

```text
Data/Processed/DataSyn TFIM Thermal/DataSyn HS Kernel/  ~3.58 GB
```

Expected local layout:

```text
Data/
  Raw/
  Processed/  # generated by Data/Data_Processing.ipynb
```

The processing step consumes only a subset of the raw columns: `hs.csv`, `ipr_x.csv` and `ipr_z.csv` for TFIM, and `IPR X.csv`, `IPR XY.csv`, `IPR Z.csv` for Heisenberg. It also narrows the thermal data to `L = 20, 50, 100` and `beta <= 2.0`, so most of the thermal sweep is archived rather than consumed downstream. `Phase_diagram.ipynb` reads the raw thermal files directly and does use the full `beta` range.

When bit-level fidelity matters, read these CSVs with `pandas.read_csv(..., float_precision="round_trip")`. The default parser is off by one unit in the last place on most values; the files themselves are exact.

## Usage

The reusable code is in `scripts/`, with dataset paths and column conventions centralized in `scripts/data_config.py`. The notebooks in `Paper Analysis/` run the full workflows and produce the paper figures and tables.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
