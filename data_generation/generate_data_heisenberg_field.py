"""Generates Data/Raw/Data Heisenberg/L=<L>_with_field/

Exact diagonalization of the periodic spin-1/2 XXZ chain in a longitudinal field
on a two-dimensional (Delta, h) grid,

    H = sum_i [X_i X_{i+1} + Y_i Y_{i+1} + Delta Z_i Z_{i+1}] + h sum_i Z_i,

written with Pauli operators. Note this is twice the Bethe-ansatz field
normalization: h_data = 2 h_BA, so the saturation field sits at 2(1 + Delta).

At each grid point the ground state is found twice, in two unitarily equivalent
representations:

  * the Pauli-string form above, whose ground state lives in the computational
    (Z) basis;
  * the Jordan-Wigner / Fourier form, whose ground state lives in the
    momentum-occupation basis of the XX model, restricted to the particle-number
    sector implied by the measured magnetization.

Their spectra must agree, and the script checks that at every point. The two IPRs
then measure delocalization of the same physical state in two different bases,
which is the point of the dataset.

Three cases skip the Fourier diagonalization and reuse the real-space state, so
the basis contrast is exactly zero there: an empty or full fermion sector (the
state is a single polarized product state, identical in both bases), and the
ferromagnetic line h = 0 with Delta < -1, where the ground state is degenerate.

Outputs per size, all CSV:
    ipr_field_sweep_L=<L>.csv   the artifact the plots read:
                                delta,h,ipr_z,ipr_xy,structure_factor_0,structure_factor_pi
    zz_correlations_L=<L>.csv   <Z_i Z_j> for i < j, one column per site pair. The
                                matrix is symmetric with a diagonal of exactly 1, so
                                the upper triangle is lossless.
    energies_L=<L>.csv          ground-state energy in each basis and the
                                magnetization used to pick the fermion sector

Usage:
    python generate_data_heisenberg_field.py                       # L=14, full grid
    python generate_data_heisenberg_field.py --sizes 6 8 --points 11
    python generate_data_heisenberg_field.py --sizes 6 --check-only

Cost grows as 2^L per grid point, times two diagonalizations, times
`points`^2 points. The committed L=14 sweep is 51 x 51 = 2601 points in a
2^14-dimensional space. Start small.

Provenance: the quick_IPR_test.py draft, whose numerics are preserved exactly; this
version adds argument handling, validation, progress reporting and CSV output. The
draft has since been removed from the repository.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

from pathlib import Path

import numpy as np
import pandas as pd

import netket as nk
from netket.operator.fermion import create as cdag
from netket.operator.fermion import destroy as c
from netket.operator.fermion import number as nc

# Parameters reproducing the committed sweep.
DEFAULT_SIZES = (14,)
GRID_MIN = -5.0
GRID_MAX = 5.0
GRID_POINTS = 51
J = 1.0

# Tolerance on the agreement between the two representations' ground-state energies.
ENERGY_ATOL = 1e-6
ENERGY_RTOL = 1e-6


def get_Hamiltonian(L: int, delta: float, h: float):
    """XXZ chain in a longitudinal field as a sum of Pauli strings."""
    graph = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
    hilbert = nk.hilbert.Spin(s=0.5, N=L)
    identity = "I" * L

    operators: list[str] = []
    weights: list[float] = []
    for edge in graph.edges():
        i, j = int(edge[0]), int(edge[1])
        operators.append(identity[:i] + "X" + identity[i + 1:j] + "X" + identity[j + 1:])
        weights.append(1.0)
        operators.append(identity[:i] + "Y" + identity[i + 1:j] + "Y" + identity[j + 1:])
        weights.append(1.0)
        operators.append(identity[:i] + "Z" + identity[i + 1:j] + "Z" + identity[j + 1:])
        weights.append(delta)
    for i in range(graph.n_nodes):
        operators.append(identity[:i] + "Z" + identity[i + 1:])
        weights.append(h)

    return nk.operator.PauliStringsJax(hilbert, operators, weights)


def build_heisenberg_fourier_operator(L: int, J: float = 1.0, delta: float = 1.0,
                                      h: float = 0.0, n_fermions: int | None = None):
    """The same Hamiltonian after Jordan-Wigner and a Fourier transform of the modes.

    Restricted to the `n_fermions` sector. The half-integer momentum offset for an
    even fermion number accounts for the boundary phase the Jordan-Wigner string
    picks up on a ring; the field enters as h (2 n_i - 1) since n_i = (1 + Z_i) / 2.
    """
    if n_fermions is None:
        n_fermions = L // 2
    hilbert = nk.hilbert.SpinOrbitalFermions(L, s=None, n_fermions=n_fermions)
    offset = 0.5 if (n_fermions % 2 == 0) else 0.0

    H = None
    for i in range(L):
        dispersion = -4 * J * np.cos(2 * np.pi * (i + offset) / L) - 4 * delta
        if H is None:
            H = dispersion * nc(hilbert, i, dtype=np.complex128)
        else:
            H = H + dispersion * nc(hilbert, i)
        for j in range(L):
            for q in range(L):
                H = H + (4 * delta * np.exp(-2j * np.pi * q / L) / L
                         * (cdag(hilbert, (i + q) % L) @ c(hilbert, i)
                            @ cdag(hilbert, (j - q) % L) @ c(hilbert, j % L)))
        H = H + h * (2 * nc(hilbert, i) - 1)
    return H + delta * L


def ground_state(operator) -> tuple[float, np.ndarray]:
    """Lowest eigenpair, returned as a scalar energy and a flat state vector."""
    energy, states = nk.exact.lanczos_ed(operator, k=1, compute_eigenvectors=True)
    return float(np.ravel(energy)[0]), np.asarray(states)[:, 0]


def correlation_matrices(spin_hilbert, states: np.ndarray) -> np.ndarray:
    """Per-state matrices of <Z_i Z_j>, shape (n_states, L, L).

    `Z_i Z_j` is diagonal in the computational basis, so this needs only the
    (dim, L) table of spin configurations and the state probabilities:

        <Z_i Z_j> = sum_x |psi(x)|^2 s_i(x) s_j(x).

    The draft built one dense `2^L x 2^L` operator per site pair, which at L=14 is
    4.3 GB each for 91 pairs. This form is numerically the same -- verified against
    the dense path in the tests -- and the diagonal comes out exactly 1 because the
    probabilities are normalized and s_i^2 = 1.
    """
    spins = np.asarray(spin_hilbert.all_states(), dtype=float)
    if spins.shape[0] != states.shape[1]:
        raise ValueError(
            f"basis size {spins.shape[0]} does not match state dimension {states.shape[1]}")

    probabilities = np.abs(states) ** 2
    out = np.empty((states.shape[0], spins.shape[1], spins.shape[1]))
    for index, probability in enumerate(probabilities):
        out[index] = spins.T @ (spins * probability[:, None])
    return out


def skips_fourier(L: int, delta: float, h: float, n_fermions: int) -> bool:
    """Whether the Fourier diagonalization is bypassed at this grid point.

    Empty and full sectors are single polarized product states, identical in both
    bases. The line h = 0, Delta < -1 is the degenerate ferromagnetic manifold,
    where the two representations need not pick the same member.
    """
    if n_fermions in (0, L):
        return True
    return abs(h) < 1e-6 and delta < -1 + 1e-6


def sweep_size(L: int, delta_array: np.ndarray, h_array: np.ndarray,
               verbose: bool = True) -> dict[str, np.ndarray]:
    """Diagonalize on the whole (Delta, h) grid and assemble every observable.

    The grid is traversed with Delta in the outer loop and h in the inner loop, so
    the flat ordering is `index = i_delta * len(h_array) + i_h`, matching a C-order
    reshape to (len(delta_array), len(h_array)).
    """
    identity = "I" * L
    spin_hilbert = nk.hilbert.Spin(s=0.5, N=L)
    total_magnetization = nk.operator.PauliStringsJax(
        spin_hilbert,
        [identity[:i] + "Z" + identity[i + 1:] for i in range(L)],
        [1.0] * L,
    )

    real_states: list[np.ndarray] = []
    fourier_states: list[np.ndarray] = []
    energies_real: list[float] = []
    energies_fourier: list[float] = []
    magnetizations: list[float] = []
    fourier_skipped: list[bool] = []
    mismatches: list[dict[str, float]] = []

    total = delta_array.size * h_array.size
    for i_delta, delta in enumerate(delta_array):
        for h in h_array:
            delta, h = float(delta), float(h)

            energy, psi = ground_state(get_Hamiltonian(L, delta, h))
            real_states.append(psi)
            energies_real.append(energy)

            magnetization = float(np.real(psi.conj() @ (total_magnetization @ psi)))
            magnetizations.append(magnetization)
            n_fermions = (int(np.round(magnetization)) + L) // 2

            if skips_fourier(L, delta, h, n_fermions):
                fourier_states.append(psi)
                energies_fourier.append(energy)
                fourier_skipped.append(True)
                continue

            fourier_operator = (
                nk.experimental.operator.ParticleNumberConservingFermioperator2nd
                .from_fermionoperator2nd(
                    build_heisenberg_fourier_operator(L, J=J, delta=delta, h=h,
                                                      n_fermions=n_fermions)
                )
            )
            energy_fourier, psi_fourier = ground_state(fourier_operator)
            fourier_states.append(psi_fourier)
            energies_fourier.append(energy_fourier)
            fourier_skipped.append(False)

            if not np.allclose(energy, energy_fourier, atol=ENERGY_ATOL, rtol=ENERGY_RTOL):
                mismatches.append({
                    "delta": delta, "h": h, "n_fermions": n_fermions,
                    "energy_z": energy, "energy_xy": energy_fourier,
                    "abs_diff": abs(energy - energy_fourier),
                })

        if verbose:
            done = (i_delta + 1) * h_array.size
            print(f"  L={L}: {done}/{total} grid points", flush=True)

    if mismatches:
        worst = max(mismatches, key=lambda m: m["abs_diff"])
        print(f"  WARNING: {len(mismatches)}/{total} points disagree between bases "
              f"beyond atol={ENERGY_ATOL}. Worst: Delta={worst['delta']:.2f}, "
              f"h={worst['h']:.2f}, n_fermions={worst['n_fermions']}, "
              f"|dE|={worst['abs_diff']:.3e}")

    real_array = np.asarray(real_states)
    ipr_z = np.sum(np.abs(real_array) ** 4, axis=1)
    ipr_xy = np.array([float(np.sum(np.abs(state) ** 4)) for state in fourier_states])

    correlation_matrix = correlation_matrices(spin_hilbert, real_array)

    signs = np.array([[(-1) ** (i + j) for j in range(L)] for i in range(L)])
    structure_factor_0 = np.mean(correlation_matrix, axis=(1, 2))
    structure_factor_pi = np.mean(correlation_matrix * signs, axis=(1, 2))

    return {
        "ipr_z": ipr_z,
        "ipr_xy": ipr_xy,
        "structure_factor_0": structure_factor_0,
        "structure_factor_pi": structure_factor_pi,
        "correlation_matrix": correlation_matrix,
        "energy_z": np.asarray(energies_real),
        "energy_xy": np.asarray(energies_fourier),
        "magnetization": np.asarray(magnetizations),
        "fourier_skipped": np.asarray(fourier_skipped),
        "mismatches": mismatches,
    }


def write_outputs(L: int, delta_array: np.ndarray, h_array: np.ndarray,
                  results: dict[str, np.ndarray], folder: Path) -> list[Path]:
    """Write every result as CSV: observables, ZZ correlations and energies."""
    folder.mkdir(parents=True, exist_ok=True)

    delta_column = np.repeat(delta_array, h_array.size)
    h_column = np.tile(h_array, delta_array.size)
    coordinates = {"delta": delta_column, "h": h_column}

    # 17 significant digits round-trips float64 exactly, so the grid coordinates read
    # back bit-identical to the linspace that produced them.
    def write(frame: pd.DataFrame, name: str) -> Path:
        path = folder / name
        frame.to_csv(path, index=False, float_format="%.17g")
        return path

    paths = [write(pd.DataFrame({
        **coordinates,
        "ipr_z": results["ipr_z"],
        "ipr_xy": results["ipr_xy"],
        "structure_factor_0": results["structure_factor_0"],
        "structure_factor_pi": results["structure_factor_pi"],
    }), f"ipr_field_sweep_L={L}.csv")]

    # Upper triangle only: the matrix is symmetric and its diagonal is exactly 1.
    correlation = results["correlation_matrix"]
    pairs = {f"c_{i}_{j}": correlation[:, i, j]
             for i in range(L) for j in range(i + 1, L)}
    paths.append(write(pd.DataFrame({**coordinates, **pairs}),
                       f"zz_correlations_L={L}.csv"))

    paths.append(write(pd.DataFrame({
        **coordinates,
        "energy_z": results["energy_z"],
        "energy_xy": results["energy_xy"],
        "magnetization": results["magnetization"],
        "fourier_skipped": results["fourier_skipped"].astype(int),
    }), f"energies_L={L}.csv"))

    return paths


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Exact-diagonalization (Delta, h) sweep of the periodic XXZ chain.")
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES),
                        help="chain lengths (default: 14, the committed size)")
    parser.add_argument("--points", type=int, default=GRID_POINTS,
                        help=f"grid points per axis (default {GRID_POINTS})")
    parser.add_argument("--grid-min", type=float, default=GRID_MIN)
    parser.add_argument("--grid-max", type=float, default=GRID_MAX)
    parser.add_argument("--outdir", type=Path,
                        default=repo_root / "Data" / "Raw" / "Data Heisenberg")
    parser.add_argument("--check-only", action="store_true",
                        help="report basis-energy agreement without writing any files")
    args = parser.parse_args()

    # Rounded to 10 decimals so each coordinate is the nearest double to a clean
    # decimal. The shift from
    # the raw linspace is ~4e-16 and physically irrelevant.
    delta_array = np.round(np.linspace(args.grid_min, args.grid_max, args.points), 10)
    h_array = np.round(np.linspace(args.grid_min, args.grid_max, args.points), 10)

    for L in args.sizes:
        if L % 2:
            raise ValueError(f"L must be even for a half-filled ring, got {L}")
        print("=" * 70)
        print(f"XXZ chain in a field, L = {L}, "
              f"{args.points} x {args.points} grid on [{args.grid_min}, {args.grid_max}]")
        print("=" * 70)

        results = sweep_size(L, delta_array, h_array)

        n_skipped = int(results["fourier_skipped"].sum())
        print(f"  Fourier diagonalization bypassed at {n_skipped} points "
              f"(polarized sectors and the degenerate h=0, Delta<-1 line)")
        print(f"  basis-energy agreement: "
              f"{len(results['mismatches'])} mismatches beyond atol={ENERGY_ATOL}")

        if args.check_only:
            continue

        folder = args.outdir / f"L={L}_with_field"
        paths = write_outputs(L, delta_array, h_array, results, folder)
        print(f"  wrote {', '.join(path.name for path in paths)} in {folder}")


if __name__ == "__main__":
    main()
