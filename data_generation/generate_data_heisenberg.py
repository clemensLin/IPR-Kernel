"""Generates Data/Raw/Data Heisenberg/L=<L>/

Exact diagonalization of the periodic spin-1/2 XXZ chain

    H = sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Delta * Z_i Z_{i+1}) + h sum_i Z_i

for a sweep in the anisotropy Delta. Two representations of the same Hamiltonian
are diagonalized:

  * the Pauli-string form above, whose ground state ``V`` lives in the product
    (Z) basis;
  * the Jordan-Wigner / Fourier form ``build_heisenberg_fourier_operator``, whose
    ground state ``Vnew`` lives in the momentum-occupation basis of the XX model.

Because the two operators are unitarily equivalent, their spectra agree and the
IPRs of the two ground states measure delocalization of the same physical state
in two different bases. This is the point of the dataset: at Delta = 0 the
momentum basis is the exact eigenbasis, so IPR XY = 1, while the Z-basis IPR is
small.

Outputs per size, matching the committed raw data (single-column CSV with a
``value`` header):
    Delta.csv            the anisotropy grid the other columns are indexed by
    energies_old.csv     ground-state energy from the Pauli-string operator
    energies_new.csv     ground-state energy from the Fourier operator
    all_correlations.csv 4 * <Z_0 Z_{L/2}>
    IPR Z.csv            IPR of V in the computational basis
    IPR X.csv            IPR of V after a global Hadamard rotation
    IPR XY.csv           IPR of Vnew in the momentum basis
    IPR_vs_Delta.png     overview plot
    Correlation_vs_Delta.png

Usage:
    python generate_data_heisenberg.py                 # all committed sizes
    python generate_data_heisenberg.py --sizes 6 8 10

Cost grows as 2^L: the committed L = 24 run needs a 2^24-dimensional sparse
diagonalization for both operators and roughly 100 GB-hours. Start small.

Provenance: assembled from the XXZ_momentum_ED.py draft, since removed from the
repository.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

from pathlib import Path

import jax.numpy as jnp
import matplotlib
import numpy as np
import pandas as pd
import scipy.sparse.linalg

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netket as nk
from netket.operator.fermion import create as cdag
from netket.operator.fermion import destroy as c
from netket.operator.fermion import number as nc
from tqdm import tqdm

# Parameters reproducing the committed raw data.
SIZES = list(range(6, 26, 2))
# linspace, not arange: this is the exact grid Data/Data_Processing.ipynb assumes
# as `source_delta` when it interpolates these columns.
DELTA_ARRAY = np.linspace(0.0, 5.0, 51)
FIELD = 0.0
J = 1.0
# The stored correlations are 4 * <sigma^z_0 sigma^z_{L/2}>, matching the committed
# data. This is a normalization convention, so the Delta- and L-dependence is
# unaffected, but note the values are NOT confined to [-1, 1] as a bare Pauli
# correlator would be: they saturate at +/-4 for strong Neel order. Label plots
# accordingly, or divide by 4.
CORRELATION_WEIGHT = 4.0


def robust_lanczos_ed(operator, k: int = 1, compute_eigenvectors: bool = False,
                      max_retries: int = 5):
    """Lanczos ground state, retrying from random start vectors on ARPACK stalls."""
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            if attempt == 0:
                return nk.exact.lanczos_ed(operator, k=k,
                                           compute_eigenvectors=compute_eigenvectors)
            print(f"ARPACK did not converge on attempt {attempt}; retrying with random v0.")
            n_states = operator.hilbert.n_states
            v0 = np.random.rand(n_states) - 0.5
            v0 /= np.linalg.norm(v0)
            return nk.exact.lanczos_ed(
                operator, k=k, compute_eigenvectors=compute_eigenvectors,
                scipy_args={"v0": v0, "maxiter": 100000},
            )
        except scipy.sparse.linalg.ArpackNoConvergence as exc:
            last_error = exc
    raise RuntimeError(
        f"ARPACK failed to converge after {max_retries} attempts"
    ) from last_error


def get_XXZ_model(L: int, delta: float, h: float = 0.0):
    """Periodic XXZ chain as a sum of Pauli strings."""
    graph = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
    hi = nk.hilbert.Spin(s=0.5, N=graph.n_nodes)

    identity = "I" * graph.n_nodes
    operators, weights = [], []
    for edge in graph.edges():
        i, j = int(edge[0]), int(edge[1])
        operators.append(identity[:i] + "X" + identity[i + 1:j] + "X" + identity[j + 1:])
        weights.append(J)
        operators.append(identity[:i] + "Y" + identity[i + 1:j] + "Y" + identity[j + 1:])
        weights.append(J)
        operators.append(identity[:i] + "Z" + identity[i + 1:j] + "Z" + identity[j + 1:])
        weights.append(delta)
    for i in range(graph.n_nodes):
        operators.append(identity[:i] + "Z" + identity[i + 1:])
        weights.append(h)

    return nk.operator.PauliStringsJax(hi, operators=operators, weights=weights)


def build_heisenberg_fourier_operator(L: int, J: float = 1.0, delta: float = 1.0,
                                      h: float = 0.0, n_fermions: int | None = None):
    """Same Hamiltonian after Jordan-Wigner and a Fourier transform of the modes.

    Restricted to the ``n_fermions`` sector, matching
    ``generate_data_heisenberg_field.build_heisenberg_fourier_operator``.

    On a ring the Jordan-Wigner string gives the wrapping bond a factor
    ``(-1)^N_f``, so the fermions are periodic or antiperiodic according to the
    parity of the fermion *number*, and the half-integer momentum offset implements
    the antiperiodic case. The criterion is therefore ``n_fermions % 2``, not a
    function of ``L`` alone: a fixed offset applied across an unconstrained Fock
    space is correct only in sectors whose parity happens to match. Verified against
    exact diagonalization sector by sector.

    The field enters as ``h (2 n_i - 1)`` since ``n_i = (1 + Z_i) / 2``.
    """
    if n_fermions is None:
        n_fermions = L // 2
    hi = nk.hilbert.SpinOrbitalFermions(L, s=None, n_fermions=n_fermions)
    offset = 0.5 if (n_fermions % 2 == 0) else 0.0

    H = None
    for i in range(L):
        dispersion = -4 * J * np.cos(2 * np.pi * (i + offset) / L) - 4 * delta
        if H is None:
            H = dispersion * nc(hi, i, dtype=jnp.complex128)
        else:
            H = H + dispersion * nc(hi, i)
        for j in range(L):
            for q in range(L):
                H = H + (4 * delta * np.exp(-2j * np.pi * q / L) / L
                         * (cdag(hi, (i + q) % L) @ c(hi, i)
                            @ cdag(hi, (j - q) % L) @ c(hi, j % L)))
        H = H + h * (2 * nc(hi, i) - 1)
    return H + delta * L


def hadamard_ipr(state: np.ndarray, L: int) -> float:
    """IPR of ``state`` after the global Hadamard rotation into the X basis."""
    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    psi = np.asarray(state).reshape((2,) * L)
    for axis in range(L):
        psi = np.tensordot(hadamard, psi, axes=([1], [axis]))
        psi = np.moveaxis(psi, 0, axis)
    return float(np.sum(np.abs(psi) ** 4))


def write_column(path: Path, values) -> None:
    """Write a one-column CSV with a ``value`` header, as in the raw data."""
    pd.Series(np.real_if_close(np.asarray(values)).astype(float),
              name="value").to_csv(path, index=False)


def run_size(L: int, deltas: np.ndarray, h: float, out_root: Path) -> Path:
    folder = out_root / f"L={L}"
    folder.mkdir(parents=True, exist_ok=True)

    energies_old, energies_new = [], []
    ipr_z, ipr_x, ipr_xy = [], [], []
    correlations = []

    identity = "I" * L
    total_magnetization = nk.operator.PauliStringsJax(
        nk.hilbert.Spin(s=0.5, N=L),
        [identity[:i] + "Z" + identity[i + 1:] for i in range(L)],
        [1.0] * L,
    )

    for delta in tqdm(deltas, desc=f"L={L}"):
        delta = float(delta)

        ha = get_XXZ_model(L, delta=delta, h=h)
        E, V = robust_lanczos_ed(ha, k=1, compute_eigenvectors=True)
        V = np.asarray(V).flatten()

        neel = nk.operator.PauliStringsJax(
            ha.hilbert,
            operators=["Z" + identity[1:L // 2] + "Z" + identity[L // 2 + 1:]],
            weights=[CORRELATION_WEIGHT],
        )
        correlation = V.conj() @ (neel.to_sparse() @ V)

        # The Fourier form is restricted to one particle-number sector, so pick the
        # sector the real-space ground state actually occupies. On the committed grid
        # (Delta >= 0, h = 0) this is always half filling, but reading it off the
        # magnetization keeps the operator's momentum offset correct for any point.
        magnetization = float(np.real(V.conj() @ (total_magnetization.to_sparse() @ V)))
        n_fermions = (int(np.round(magnetization)) + L) // 2
        if n_fermions in (0, L):
            raise ValueError(
                f"L={L} Delta={delta}: the ground state is fully polarized "
                f"(n_fermions={n_fermions}), where the momentum basis is a single "
                "state and the Fourier comparison is meaningless.")

        H_fourier = nk.experimental.operator.ParticleNumberConservingFermioperator2nd.from_fermionoperator2nd(
            build_heisenberg_fourier_operator(L=L, J=J, delta=delta, h=h,
                                              n_fermions=n_fermions)
        )
        Enew, Vnew = robust_lanczos_ed(H_fourier, k=1, compute_eigenvectors=True)
        Vnew = np.asarray(Vnew).flatten()

        if not np.allclose(E, Enew, atol=1e-6, rtol=1e-6):
            print(f"  warning: L={L} Delta={delta:.2f} spectra disagree, "
                  f"E={float(np.ravel(E)[0]):.8f} vs {float(np.ravel(Enew)[0]):.8f}")

        energies_old.append(float(np.ravel(E)[0]))
        energies_new.append(float(np.ravel(Enew)[0]))
        correlations.append(np.real(correlation))
        ipr_z.append(float(np.sum(np.abs(V) ** 4)))
        ipr_xy.append(float(np.sum(np.abs(Vnew) ** 4)))
        ipr_x.append(hadamard_ipr(V, L))

    write_column(folder / "Delta.csv", np.round(deltas, 10))
    write_column(folder / "energies_old.csv", energies_old)
    write_column(folder / "energies_new.csv", energies_new)
    write_column(folder / "all_correlations.csv", correlations)
    write_column(folder / "IPR Z.csv", ipr_z)
    write_column(folder / "IPR X.csv", ipr_x)
    write_column(folder / "IPR XY.csv", ipr_xy)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(deltas, ipr_z, label="IPR Z")
    ax.semilogy(deltas, ipr_x, label="IPR X")
    ax.semilogy(deltas, ipr_xy, label="IPR XY")
    ax.set_xlabel(r"$\Delta$")
    ax.set_ylabel("IPR")
    ax.set_title(f"XXZ chain, L = {L}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "IPR_vs_Delta.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(deltas, correlations)
    ax.set_xlabel(r"$\Delta$")
    ax.set_ylabel(rf"$4\langle \sigma^z_0 \sigma^z_{{{L // 2}}}\rangle$")
    ax.set_title(f"XXZ chain, L = {L}")
    fig.tight_layout()
    fig.savefig(folder / "Correlation_vs_Delta.png", dpi=150)
    plt.close(fig)

    print(f"Wrote {folder}")
    return folder


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Exact diagonalization sweep of the periodic XXZ chain.")
    parser.add_argument("--sizes", type=int, nargs="+", default=SIZES,
                        help="chain lengths to run (default: all committed sizes)")
    parser.add_argument("--field", type=float, default=FIELD,
                        help="longitudinal field h (default 0.0, as committed)")
    parser.add_argument("--outdir", type=Path,
                        default=repo_root / "Data" / "Raw" / "Data Heisenberg")
    args = parser.parse_args()

    for L in args.sizes:
        if L % 2:
            raise ValueError(f"L must be even for a half-filled ring, got {L}")
        print("=" * 70)
        print(f"XXZ chain, L = {L}, h = {args.field}, "
              f"{len(DELTA_ARRAY)} Delta points")
        print("=" * 70)
        run_size(L, DELTA_ARRAY, args.field, args.outdir)


if __name__ == "__main__":
    main()
