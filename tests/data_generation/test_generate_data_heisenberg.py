"""Consistency checks for data_generation/generate_data_heisenberg.py.

The reference is an independent dense Kronecker-product construction of the periodic
XXZ chain, diagonalized exactly. Six things are checked:

1. The netket Pauli-string Hamiltonian equals the explicit dense construction.
2. Ground-state energies match exact diagonalization at representative anisotropies.
3. ``IPR Z`` matches the dense ground state and ``IPR X`` matches it after a global
   Hadamard rotation.
4. ``IPR XY`` matches an independent Jordan-Wigner momentum-basis construction, which
   pins the momentum offset convention.
5. The Fourier operator reproduces the exact spectrum in *every* particle-number
   sector, not only at half filling. This is what fixes the momentum offset to the
   fermion-number parity.
6. ``all_correlations`` is ``4 <Z_0 Z_{L/2}>`` (written by the generator, not committed),
   and the generator reproduces the committed IPRs and energies.

Runs standalone or under pytest.
"""

from __future__ import annotations

import importlib.util
import sys
from functools import reduce
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = REPO_ROOT / "data_generation" / "generate_data_heisenberg.py"
COMMITTED = REPO_ROOT / "data" / "raw" / "xxz"


def _load_generator():
    spec = importlib.util.spec_from_file_location("heis", GENERATOR)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["heis"] = module
    spec.loader.exec_module(module)
    return module


gen = _load_generator()

I2 = np.eye(2)
X = np.array([[0.0, 1.0], [1.0, 0.0]])
Y = np.array([[0.0, -1.0j], [1.0j, 0.0]])
Z = np.diag([1.0, -1.0])
SM = np.array([[0.0, 0.0], [1.0, 0.0]])  # sigma^-, annihilates the up state

# Representative anisotropies: the free-fermion point, the isotropic point, and both
# sides of the Delta = 1 transition.
DELTAS = [0.0, 0.5, 1.0, 2.0, 3.5]


def kron_op(L: int, ops: dict[int, np.ndarray]) -> np.ndarray:
    return reduce(np.kron, [ops.get(i, I2) for i in range(L)])


def dense_xxz(L: int, delta: float, h: float = 0.0) -> np.ndarray:
    """H = sum_i [XX + YY + Delta ZZ] + h sum_i Z, periodic, built from scratch."""
    H = np.zeros((2**L, 2**L), dtype=complex)
    for i in range(L):
        j = (i + 1) % L
        H += kron_op(L, {i: X, j: X})
        H += kron_op(L, {i: Y, j: Y})
        H += delta * kron_op(L, {i: Z, j: Z})
    for i in range(L):
        H += h * kron_op(L, {i: Z})
    return H


def ipr(state: np.ndarray) -> float:
    return float(np.sum(np.abs(state) ** 4))


def hadamard_rotate(state: np.ndarray, L: int) -> np.ndarray:
    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    return reduce(np.kron, [hadamard] * L) @ state


def momentum_basis_ipr(state: np.ndarray, L: int, n_fermions: int | None = None) -> float:
    """IPR in the Jordan-Wigner momentum-occupation basis of the XX model.

    Built independently of the generator: Jordan-Wigner fermions, Fourier-transformed,
    then every occupation state of the given filling is constructed explicitly and
    overlapped with ``state``. The offset follows the fermion-number parity, since the
    wrapping bond carries a factor ``(-1)^N_f``.
    """
    if n_fermions is None:
        n_fermions = L // 2
    shift = 0.5 if (n_fermions % 2 == 0) else 0.0
    c_ops = [kron_op(L, {**{m: Z for m in range(j)}, j: SM}) for j in range(L)]
    ck_dag = [
        sum(np.exp(-2j * np.pi * j * (k + shift) / L) * c_ops[j] for j in range(L)).conj().T
        / np.sqrt(L)
        for k in range(L)
    ]
    vacuum = np.zeros(2**L, dtype=complex)
    vacuum[-1] = 1.0  # all spins down

    total = 0.0
    for occupation in combinations(range(L), n_fermions):
        basis = vacuum
        for k in occupation:
            basis = ck_dag[k] @ basis
        norm = np.linalg.norm(basis)
        if norm < 1e-10:
            continue
        total += abs((basis / norm).conj() @ state) ** 4
    return float(total)


def test_pauli_hamiltonian_matches_dense(L: int = 6) -> None:
    worst = 0.0
    for delta in DELTAS:
        netket_dense = np.asarray(gen.get_XXZ_model(L, delta=delta, h=0.0).to_dense())
        worst = max(worst, float(np.max(np.abs(netket_dense - dense_xxz(L, delta)))))
    print(f"  [1] Pauli-string vs dense Kronecker, L={L}: max |diff| = {worst:.3e}")
    assert worst < 1e-10, f"Hamiltonian mismatch {worst:.3e}"


def test_energies_and_iprs_match_dense(L: int = 6) -> None:
    print(f"  [2] energies and IPRs vs exact diagonalization, L={L}:")
    print(f"      {'Delta':>6} {'E':>15} {'dE':>10} {'d IPR_Z':>10} {'d IPR_X':>10}")
    worst = 0.0
    for delta in DELTAS:
        energy, state = gen.robust_lanczos_ed(
            gen.get_XXZ_model(L, delta=delta, h=0.0), k=1, compute_eigenvectors=True)
        state = np.asarray(state).flatten()

        w, v = np.linalg.eigh(dense_xxz(L, delta))
        if abs(w[1] - w[0]) < 1e-9:
            print(f"      {delta:6.1f}   degenerate ground state, skipped")
            continue
        reference = v[:, 0]

        d_e = abs(float(np.ravel(energy)[0]) - w[0])
        d_z = abs(ipr(state) - ipr(reference))
        d_x = abs(ipr(hadamard_rotate(state, L)) - ipr(hadamard_rotate(reference, L)))
        print(f"      {delta:6.1f} {w[0]:15.9f} {d_e:10.2e} {d_z:10.2e} {d_x:10.2e}")
        worst = max(worst, d_e, d_z, d_x)
    print(f"      worst deviation: {worst:.3e}")
    assert worst < 1e-8, f"observables disagree by {worst:.3e}"


def test_momentum_basis_ipr(L: int = 6) -> None:
    """IPR XY matches an independent momentum-basis construction."""
    print(f"  [3] IPR XY vs independent Jordan-Wigner momentum basis, L={L}:")
    print(f"      {'Delta':>6} {'generator':>12} {'independent':>13} {'diff':>10}")
    worst = 0.0
    for delta in DELTAS:
        fourier = gen.build_heisenberg_fourier_operator(L, J=1.0, delta=delta, h=0.0)
        _, state_new = gen.robust_lanczos_ed(fourier, k=1, compute_eigenvectors=True)
        got = ipr(np.asarray(state_new).flatten())

        w, v = np.linalg.eigh(dense_xxz(L, delta))
        if abs(w[1] - w[0]) < 1e-9:
            print(f"      {delta:6.1f}   degenerate ground state, skipped")
            continue
        want = momentum_basis_ipr(v[:, 0], L)
        print(f"      {delta:6.1f} {got:12.8f} {want:13.8f} {abs(got - want):10.2e}")
        worst = max(worst, abs(got - want))
    print(f"      worst deviation: {worst:.3e}")
    # At Delta = 0 this must be exactly 1: the momentum basis is then the eigenbasis.
    assert worst < 1e-8, f"momentum-basis IPR disagrees by {worst:.3e}"


def sector_spectrum(L: int, delta: float, h: float, n_fermions: int) -> np.ndarray:
    operator = gen.build_heisenberg_fourier_operator(
        L, J=1.0, delta=delta, h=h, n_fermions=n_fermions)
    return np.sort(np.real(np.linalg.eigvalsh(np.asarray(operator.to_dense()))))


def spin_sector_spectrum(L: int, delta: float, h: float, n_up: int) -> np.ndarray:
    """Exact XXZ spectrum restricted to the sector with n_up spins up."""
    H = dense_xxz(L, delta, h)
    bits = (np.arange(2**L)[:, None] >> np.arange(L)[None, ::-1]) & 1
    idx = np.flatnonzero((bits == 0).sum(axis=1) == n_up)
    return np.sort(np.linalg.eigvalsh(H[np.ix_(idx, idx)]))


def test_fourier_operator_correct_in_every_sector(L: int = 6) -> None:
    """The momentum offset follows the fermion-number parity, in every sector.

    On a ring the Jordan-Wigner string gives the wrapping bond a factor (-1)^N_f, so
    the offset must be chosen from n_fermions % 2. An earlier version derived it from
    L alone, which is that parity evaluated at half filling: correct where it is used
    but wrong by order one in sectors of the other parity. This check would fail if
    that regressed, including at non-zero field where the sweep never runs.
    """
    print(f"  [4] Fourier operator vs exact ED in every sector, L={L}:")
    print(f"      {'Delta':>6} {'h':>5} {'N_f':>4} {'parity':>7} {'max |dE|':>10}")
    worst = 0.0
    for delta, h in [(1.0, 0.0), (2.0, 0.0), (1.0, 0.7)]:
        for n_fermions in range(1, L):
            exact = spin_sector_spectrum(L, delta, h, n_fermions)
            deviation = float(np.max(np.abs(sector_spectrum(L, delta, h, n_fermions) - exact)))
            worst = max(worst, deviation)
            parity = "even" if n_fermions % 2 == 0 else "odd"
            print(f"      {delta:6.1f} {h:5.1f} {n_fermions:4d} {parity:>7} "
                  f"{deviation:10.2e}")
    print(f"      worst deviation: {worst:.3e}")
    assert worst < 1e-8, f"Fourier operator wrong in some sector, by {worst:.3e}"


def test_correlation_normalization(L: int = 6) -> None:
    """all_correlations is 4 <Z_0 Z_{L/2}>, and the weight constant says so."""
    assert gen.CORRELATION_WEIGHT == 4.0
    operator = kron_op(L, {0: Z, L // 2: Z})
    print(f"  [5] all_correlations / <Z_0 Z_{{{L // 2}}}>, L={L}:")
    ratios = []
    for delta in DELTAS:
        _, state = gen.robust_lanczos_ed(
            gen.get_XXZ_model(L, delta=delta, h=0.0), k=1, compute_eigenvectors=True)
        state = np.asarray(state).flatten()
        bare = float(np.real(state.conj() @ (operator @ state)))
        if abs(bare) < 1e-9:
            continue
        neel = gen.nk.operator.PauliStringsJax(
            gen.get_XXZ_model(L, delta=delta, h=0.0).hilbert,
            operators=["Z" + "I" * (L // 2 - 1) + "Z" + "I" * (L - L // 2 - 1)],
            weights=[gen.CORRELATION_WEIGHT],
        )
        scaled = float(np.real(state.conj() @ (neel.to_sparse() @ state)))
        ratios.append(scaled / bare)
        print(f"      Delta={delta:5.1f}  ratio = {scaled / bare:.10f}")
    assert ratios, "no non-zero correlations to compare"
    assert max(abs(r - 4.0) for r in ratios) < 1e-9, f"ratios {ratios}"


def test_matches_committed_data(L: int = 6) -> None:
    """Regenerating representative anisotropies reproduces the committed CSVs."""
    folder = COMMITTED / f"L={L}"
    if not folder.is_dir():
        print(f"  [6] skipped: {folder} not present")
        return

    read = lambda name: pd.read_csv(folder / name, float_precision="round_trip")["value"].to_numpy()
    grid = read("Delta.csv")
    # The committed grid must be exactly the one the generator sweeps. This is a plain
    # linspace, unlike the field sweep, which rounds to ten decimals.
    assert np.array_equal(grid, gen.DELTA_ARRAY), (
        f"committed Delta grid differs from the generator's DELTA_ARRAY; "
        f"max |diff| = {np.max(np.abs(grid - gen.DELTA_ARRAY)):.3e}")

    columns = {
        "energies_old.csv": "energy_z", "energies_new.csv": "energy_xy",
        "IPR Z.csv": "ipr_z", "IPR X.csv": "ipr_x", "IPR XY.csv": "ipr_xy",
    }
    committed = {name: read(name) for name in columns}

    print(f"  [6] regenerated vs committed, L={L}:")
    print(f"      {'Delta':>6} " + " ".join(f"{c:>11}" for c in columns.values()))
    worst = 0.0
    for delta in [0.0, 1.0, 2.5]:
        k = int(np.argmin(np.abs(grid - delta)))
        model = gen.get_XXZ_model(L, delta=float(grid[k]), h=0.0)
        energy, state = gen.robust_lanczos_ed(model, k=1, compute_eigenvectors=True)
        state = np.asarray(state).flatten()
        fourier = gen.build_heisenberg_fourier_operator(L, J=1.0, delta=float(grid[k]), h=0.0)
        energy_xy, state_xy = gen.robust_lanczos_ed(fourier, k=1, compute_eigenvectors=True)

        values = {
            "energies_old.csv": float(np.ravel(energy)[0]),
            "energies_new.csv": float(np.ravel(energy_xy)[0]),
            "IPR Z.csv": ipr(state),
            "IPR X.csv": ipr(hadamard_rotate(state, L)),
            "IPR XY.csv": ipr(np.asarray(state_xy).flatten()),
        }
        diffs = [abs(values[n] - committed[n][k]) for n in columns]
        print(f"      {grid[k]:6.1f} " + " ".join(f"{d:11.2e}" for d in diffs))
        worst = max(worst, max(diffs))
    print(f"      worst deviation: {worst:.3e}")
    assert worst < 1e-8, f"regenerated data differs by {worst:.3e}"


def main() -> None:
    print("=" * 78)
    print("Consistency checks: zero-field XXZ generator")
    print("=" * 78)
    for check in (test_pauli_hamiltonian_matches_dense,
                  test_energies_and_iprs_match_dense,
                  test_momentum_basis_ipr,
                  test_fourier_operator_correct_in_every_sector,
                  test_correlation_normalization,
                  test_matches_committed_data):
        check()
        print()
    print("All checks passed.")


if __name__ == "__main__":
    main()
