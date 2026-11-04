"""Consistency checks for data_generation/generate_data_heisenberg_field.py.

Runs at representative (Delta, h) points rather than over the full grid, and checks
four independent things:

1. The two representations agree on the ground-state energy. This is the check the
   generator itself performs; here it is exercised on a spread of points including
   the awkward ones (zero field, saturation boundary, both signs of Delta).
2. The Pauli-string Hamiltonian matches a from-scratch dense Kronecker-product
   construction, so the netket operator really is the intended XXZ chain.
3. The Z-basis IPR, the q=0 and q=pi structure factors and the magnetization match
   direct dense-matrix evaluation.
4. The generator reproduces the committed L=14 sweep at representative points.

Runs standalone (`python tests/data_generation/test_...py`) or under pytest.
"""

from __future__ import annotations

import importlib.util
import sys
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = REPO_ROOT / "data_generation" / "generate_data_heisenberg_field.py"
COMMITTED_CSV = REPO_ROOT / "data" / "raw" / "xxz_field" / "L=14" / "ipr_field_sweep_L=14.csv"


def _load_generator():
    spec = importlib.util.spec_from_file_location("heis_field", GENERATOR)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["heis_field"] = module
    spec.loader.exec_module(module)
    return module


gen = _load_generator()

I2 = np.eye(2)
X = np.array([[0.0, 1.0], [1.0, 0.0]])
Y = np.array([[0.0, -1.0j], [1.0j, 0.0]])
Z = np.diag([1.0, -1.0])

# Representative points: zero field, both signs of Delta, the isotropic point, near
# and beyond the saturation field h_sat = 2(1 + Delta), and the gapped Neel side.
REPRESENTATIVE = [
    (0.0, 0.0), (1.0, 0.0), (-2.0, 0.0), (2.0, 0.0),
    (1.0, 1.0), (1.0, 2.0), (1.0, 3.0),
    (2.0, 1.0), (2.0, 4.0), (2.0, 6.2),
    (-0.5, 0.6), (-2.0, 1.0), (3.0, 2.0), (0.5, -1.4), (-1.0, -2.0),
]


def kron_op(L: int, ops: dict[int, np.ndarray]) -> np.ndarray:
    return reduce(np.kron, [ops.get(i, I2) for i in range(L)])


def dense_xxz(L: int, delta: float, h: float) -> np.ndarray:
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


def test_pauli_hamiltonian_matches_dense(L: int = 6) -> None:
    """The netket Pauli-string operator equals the explicit Kronecker construction."""
    worst = 0.0
    for delta, h in [(0.0, 0.0), (1.0, 0.7), (-2.0, 1.3), (2.5, -0.9)]:
        netket_dense = np.asarray(gen.get_Hamiltonian(L, delta, h).to_dense())
        worst = max(worst, float(np.max(np.abs(netket_dense - dense_xxz(L, delta, h)))))
    print(f"  [1] Pauli-string vs dense Kronecker, L={L}: max |diff| = {worst:.3e}")
    assert worst < 1e-10, f"Hamiltonian mismatch {worst:.3e}"


def test_basis_energies_agree(L: int = 6) -> None:
    """Real-space and Fourier-space ground-state energies agree where both are used."""
    rows = []
    for delta, h in REPRESENTATIVE:
        energy_z, psi = gen.ground_state(gen.get_Hamiltonian(L, delta, h))
        magnetization = float(np.real(psi.conj() @ (kron_sum_z(L) @ psi)))
        n_fermions = (int(np.round(magnetization)) + L) // 2
        skipped = gen.skips_fourier(L, delta, h, n_fermions)
        if skipped:
            rows.append((delta, h, n_fermions, energy_z, np.nan, np.nan, True))
            continue
        operator = _fourier_operator(L, delta, h, n_fermions)
        energy_xy, _ = gen.ground_state(operator)
        rows.append((delta, h, n_fermions, energy_z, energy_xy,
                     abs(energy_z - energy_xy), False))

    print(f"  [2] basis energies at {len(rows)} representative points, L={L}:")
    print(f"      {'Delta':>6} {'h':>6} {'n_f':>4} {'E (Z basis)':>14} "
          f"{'E (XY basis)':>14} {'|dE|':>10}")
    worst = 0.0
    for delta, h, n_f, e_z, e_xy, diff, skipped in rows:
        tag = "  (Fourier skipped)" if skipped else ""
        e_xy_s = "-" if skipped else f"{e_xy:14.9f}"
        diff_s = "-" if skipped else f"{diff:10.2e}"
        print(f"      {delta:6.1f} {h:6.1f} {n_f:4d} {e_z:14.9f} {e_xy_s} {diff_s}{tag}")
        if not skipped:
            worst = max(worst, diff)
    print(f"      worst |dE| over non-skipped points: {worst:.3e}")
    assert worst < 1e-6, f"basis energies disagree by {worst:.3e}"


def kron_sum_z(L: int) -> np.ndarray:
    return sum(kron_op(L, {i: Z}) for i in range(L))


def _fourier_operator(L: int, delta: float, h: float, n_fermions: int):
    import netket as nk
    return (nk.experimental.operator.ParticleNumberConservingFermioperator2nd
            .from_fermionoperator2nd(
                gen.build_heisenberg_fourier_operator(L, J=1.0, delta=delta, h=h,
                                                      n_fermions=n_fermions)))


def test_observables_match_dense(L: int = 6) -> None:
    """IPR_Z, magnetization and both structure factors match dense evaluation."""
    signs = np.array([[(-1) ** (i + j) for j in range(L)] for i in range(L)])
    print(f"  [3] observables vs dense evaluation, L={L}:")
    print(f"      {'Delta':>6} {'h':>6} {'dIPR_Z':>10} {'dS(0)':>10} "
          f"{'dS(pi)':>10} {'dMz':>10}")
    worst = 0.0
    for delta, h in [(0.0, 0.0), (1.0, 0.0), (2.0, 1.0), (-1.5, 0.6), (2.5, 3.0)]:
        _, psi = gen.ground_state(gen.get_Hamiltonian(L, delta, h))

        w, v = np.linalg.eigh(dense_xxz(L, delta, h))
        # Compare only where the ground state is unique, otherwise the two solvers may
        # pick different members of a degenerate space and nothing is expected to match.
        if abs(w[1] - w[0]) < 1e-8:
            print(f"      {delta:6.1f} {h:6.1f}   degenerate ground state, skipped")
            continue
        ref = v[:, 0]

        corr = np.eye(L)
        for i in range(L):
            for j in range(i + 1, L):
                corr[i, j] = np.real(ref.conj() @ (kron_op(L, {i: Z, j: Z}) @ ref))
                corr[j, i] = corr[i, j]

        d_ipr = abs(float(np.sum(np.abs(psi) ** 4)) - float(np.sum(np.abs(ref) ** 4)))
        gen_corr = _generator_correlation(gen, L, psi)
        d_s0 = abs(gen_corr.mean() - corr.mean())
        d_spi = abs((gen_corr * signs).mean() - (corr * signs).mean())
        d_mz = abs(float(np.real(psi.conj() @ (kron_sum_z(L) @ psi)))
                   - float(np.real(ref.conj() @ (kron_sum_z(L) @ ref))))
        print(f"      {delta:6.1f} {h:6.1f} {d_ipr:10.2e} {d_s0:10.2e} "
              f"{d_spi:10.2e} {d_mz:10.2e}")
        worst = max(worst, d_ipr, d_s0, d_spi, d_mz)
    print(f"      worst deviation: {worst:.3e}")
    assert worst < 1e-8, f"observables disagree by {worst:.3e}"


def _generator_correlation(module, L: int, psi: np.ndarray) -> np.ndarray:
    """Correlation matrix from the generator's fast diagonal path."""
    import netket as nk
    return module.correlation_matrices(nk.hilbert.Spin(s=0.5, N=L),
                                      psi.reshape(1, -1))[0]


def _dense_correlation(L: int, psi: np.ndarray) -> np.ndarray:
    """Correlation matrix from explicit dense Z_i Z_j operators (the draft's path)."""
    import netket as nk
    identity = "I" * L
    hilbert = nk.hilbert.Spin(s=0.5, N=L)
    corr = np.eye(L)
    for i in range(L):
        for j in range(i + 1, L):
            operator = nk.operator.PauliStringsJax(
                hilbert,
                [identity[:i] + "Z" + identity[i + 1:j] + "Z" + identity[j + 1:]],
                [1.0],
            ).to_dense()
            corr[i, j] = float(np.real(psi.conj() @ (operator @ psi)))
            corr[j, i] = corr[i, j]
    return corr


def test_fast_correlation_matches_dense_operators(L: int = 6) -> None:
    """The diagonal shortcut equals building dense Z_i Z_j operators.

    The generator uses the shortcut because at L=14 the dense route needs 91
    operators of 4.3 GB each. This pins the two together on a size where the dense
    route is still cheap.
    """
    worst = 0.0
    for delta, h in [(0.0, 0.0), (1.0, 0.8), (2.0, 1.0), (-1.5, 0.6)]:
        _, psi = gen.ground_state(gen.get_Hamiltonian(L, delta, h))
        fast = _generator_correlation(gen, L, psi)
        dense = _dense_correlation(L, psi)
        worst = max(worst, float(np.max(np.abs(fast - dense))))
    print(f"  [3a] fast vs dense correlation matrix, L={L}: max |diff| = {worst:.3e}")
    assert worst < 1e-10, f"correlation matrices disagree by {worst:.3e}"


def test_matches_committed_sweep() -> None:
    """Regenerating representative L=14 points reproduces the committed CSV."""
    if not COMMITTED_CSV.is_file():
        print(f"  [4] skipped: {COMMITTED_CSV.name} not present")
        return

    committed = pd.read_csv(COMMITTED_CSV, float_precision="round_trip")
    L = 14
    signs = np.array([[(-1) ** (i + j) for j in range(L)] for i in range(L)])
    points = [(0.0, 0.0), (1.0, 0.0), (2.0, 2.0), (-2.0, 1.0), (3.0, 4.0), (1.0, -2.0)]

    print(f"  [4] regenerated vs committed L={L} sweep:")
    print(f"      {'Delta':>6} {'h':>6} {'d ipr_z':>11} {'d ipr_xy':>11} "
          f"{'d S(0)':>11} {'d S(pi)':>11}")
    worst = 0.0
    for delta, h in points:
        row = committed[np.isclose(committed.delta, delta) & np.isclose(committed.h, h)]
        if row.empty:
            print(f"      {delta:6.1f} {h:6.1f}   not on the committed grid")
            continue
        row = row.iloc[0]

        energy_z, psi = gen.ground_state(gen.get_Hamiltonian(L, delta, h))
        magnetization = float(np.real(psi.conj() @ (_sum_z_netket(L) @ psi)))
        n_fermions = (int(np.round(magnetization)) + L) // 2
        if gen.skips_fourier(L, delta, h, n_fermions):
            psi_xy = psi
        else:
            _, psi_xy = gen.ground_state(_fourier_operator(L, delta, h, n_fermions))

        corr = _generator_correlation(gen, L, psi)
        d = [abs(float(np.sum(np.abs(psi) ** 4)) - row.ipr_z),
             abs(float(np.sum(np.abs(psi_xy) ** 4)) - row.ipr_xy),
             abs(corr.mean() - row.structure_factor_0),
             abs((corr * signs).mean() - row.structure_factor_pi)]
        print(f"      {delta:6.1f} {h:6.1f} " + " ".join(f"{x:11.2e}" for x in d))
        worst = max(worst, max(d))
    print(f"      worst deviation: {worst:.3e}")
    assert worst < 1e-6, f"regenerated data differs from committed by {worst:.3e}"


def _sum_z_netket(L: int):
    import netket as nk
    identity = "I" * L
    return nk.operator.PauliStringsJax(
        nk.hilbert.Spin(s=0.5, N=L),
        [identity[:i] + "Z" + identity[i + 1:] for i in range(L)],
        [1.0] * L,
    )


def main() -> None:
    print("=" * 78)
    print("Consistency checks: XXZ (Delta, h) sweep generator")
    print("=" * 78)
    test_pauli_hamiltonian_matches_dense()
    print()
    test_basis_energies_agree()
    print()
    test_fast_correlation_matches_dense_operators()
    print()
    test_observables_match_dense()
    print()
    test_matches_committed_sweep()
    print()
    print("All checks passed.")


if __name__ == "__main__":
    main()
