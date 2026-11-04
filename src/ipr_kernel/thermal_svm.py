from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC

from .config import available_thermal_betas, normalize_thermal_kind, normalize_thermal_source
from .gram import compute_thermal_gram_matrix
from .ipr_curves import set_ipr_plot_style
from .svm import (
    _balanced_global_test_indices,
    _balanced_local_test_indices,
    _mean_std,
    _phase_labels,
)


def _run_precomputed_svm_split_with_decision(
    K: np.ndarray,
    y: np.ndarray,
    test_idx: np.ndarray,
    C: float,
) -> tuple[float, np.ndarray]:
    """Train one precomputed-kernel SVM split and return score plus full-grid decision values."""
    train_mask = np.ones(len(y), dtype=bool)
    train_mask[test_idx] = False
    train_idx = np.where(train_mask)[0]

    K_train = K[np.ix_(train_idx, train_idx)]
    K_test = K[np.ix_(test_idx, train_idx)]
    K_full = K[:, train_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    clf = SVC(kernel="precomputed", class_weight="balanced", C=float(C))
    clf.fit(K_train, y_train)
    y_pred = clf.predict(K_test)
    decision_full = clf.decision_function(K_full)

    return float(balanced_accuracy_score(y_test, y_pred)), np.asarray(decision_full, dtype=float)


def _evaluate_svm_for_kernel_with_decision(
    x: np.ndarray,
    K: np.ndarray,
    phase_boundary: float,
    test_frac: float,
    local_delta: float,
    n_splits: int,
    rng_seed: int,
    C: float,
) -> dict:
    """Evaluate balanced accuracy and store full-grid decision curves for one Gram matrix."""
    x = np.asarray(x, dtype=float)
    y = _phase_labels(x, phase_boundary)
    if np.unique(y).size < 2:
        nan_curve = np.full(len(x), np.nan)
        return {
            "global_bal_acc_mean": np.nan,
            "global_bal_acc_std": np.nan,
            "local_bal_acc_mean": np.nan,
            "local_bal_acc_std": np.nan,
            "global_decision_function_mean": nan_curve,
            "global_decision_function_std": nan_curve,
            "local_decision_function_mean": nan_curve,
            "local_decision_function_std": nan_curve,
            "global_decision_function_by_split": [],
            "local_decision_function_by_split": [],
        }

    global_balacc_runs = []
    local_balacc_runs = []
    global_decision_runs = []
    local_decision_runs = []

    for split_id in range(int(n_splits)):
        rng = np.random.default_rng(int(rng_seed) + split_id)

        global_test_idx = _balanced_global_test_indices(y, test_frac, rng)
        bacc_global, decision_global = _run_precomputed_svm_split_with_decision(
            K, y, global_test_idx, C=C
        )
        global_balacc_runs.append(bacc_global)
        global_decision_runs.append(decision_global)

        local_test_idx = _balanced_local_test_indices(
            x=x,
            y=y,
            phase_boundary=float(phase_boundary),
            local_delta=float(local_delta),
            test_frac=float(test_frac),
            rng=rng,
        )
        if local_test_idx is None:
            local_balacc_runs.append(np.nan)
            local_decision_runs.append(np.full(len(x), np.nan))
        else:
            bacc_local, decision_local = _run_precomputed_svm_split_with_decision(
                K, y, local_test_idx, C=C
            )
            local_balacc_runs.append(bacc_local)
            local_decision_runs.append(decision_local)

    global_balacc_mean, global_balacc_std = _mean_std(global_balacc_runs)
    local_balacc_mean, local_balacc_std = _mean_std(local_balacc_runs)
    global_decisions = np.asarray(global_decision_runs, dtype=float)
    local_decisions = np.asarray(local_decision_runs, dtype=float)

    return {
        "global_bal_acc_mean": global_balacc_mean,
        "global_bal_acc_std": global_balacc_std,
        "local_bal_acc_mean": local_balacc_mean,
        "local_bal_acc_std": local_balacc_std,
        "global_decision_function_mean": np.nanmean(global_decisions, axis=0),
        "global_decision_function_std": np.nanstd(
            global_decisions, axis=0, ddof=1 if len(global_decision_runs) > 1 else 0
        ),
        "local_decision_function_mean": np.nanmean(local_decisions, axis=0),
        "local_decision_function_std": np.nanstd(
            local_decisions, axis=0, ddof=1 if len(local_decision_runs) > 1 else 0
        ),
        "global_decision_function_by_split": global_decision_runs,
        "local_decision_function_by_split": local_decision_runs,
    }


def _thermal_kernel_svm_beta_sweep_for_kind(
    kind: str,
    L: int,
    betas: list[float] | tuple[float, ...] | None = None,
    source: str = "processed",
    phase_boundary: float = 1.0,
    test_frac: float = 0.10,
    local_delta: float = 0.25,
    n_splits: int = 10,
    rng_seed: int = 12345,
    C: float = 1.0,
) -> dict:
    """Compute one thermal kernel-SVM beta sweep for a single kernel kind."""
    kind = normalize_thermal_kind(kind)
    source = normalize_thermal_source(source)
    if betas is None:
        betas = available_thermal_betas(kind=kind, L=int(L), source=source)
    betas = sorted(float(beta) for beta in betas)
    if not betas:
        raise ValueError("betas must contain at least one value.")
    normalize = True if kind == "ipr" else None

    global_balacc_mean, global_balacc_std = [], []
    local_balacc_mean, local_balacc_std = [], []
    gram_metadata = []

    for beta in betas:
        gram = compute_thermal_gram_matrix(
            kind=kind,
            L=int(L),
            beta=float(beta),
            source=source,
            normalize=normalize,
        )
        x = np.asarray(gram["x"], dtype=float)
        K = np.asarray(gram["K"], dtype=float)
        metrics = _evaluate_svm_for_kernel_with_decision(
            x=x,
            K=K,
            phase_boundary=float(phase_boundary),
            test_frac=float(test_frac),
            local_delta=float(local_delta),
            n_splits=int(n_splits),
            rng_seed=int(rng_seed),
            C=float(C),
        )

        global_balacc_mean.append(metrics["global_bal_acc_mean"])
        global_balacc_std.append(metrics["global_bal_acc_std"])
        local_balacc_mean.append(metrics["local_bal_acc_mean"])
        local_balacc_std.append(metrics["local_bal_acc_std"])
        decision_metadata = {
            "x": x,
            "global_mean": metrics["global_decision_function_mean"],
            "global_std": metrics["global_decision_function_std"],
            "local_mean": metrics["local_decision_function_mean"],
            "local_std": metrics["local_decision_function_std"],
            "global_by_split": metrics["global_decision_function_by_split"],
            "local_by_split": metrics["local_decision_function_by_split"],
        }
        gram_metadata.append(
            {
                "beta": float(beta),
                "matrix_shape": tuple(K.shape),
                "h_min": float(np.min(x)),
                "h_max": float(np.max(x)),
                "path": str(gram["path"]),
                "decision_function": decision_metadata,
            }
        )

    return {
        "kind": kind,
        "source": source,
        "L": int(L),
        "betas": betas,
        "phase_boundary": float(phase_boundary),
        "kernel_normalized": None if kind == "hs" else True,
        "n_splits": int(n_splits),
        "test_frac": float(test_frac),
        "local_delta": float(local_delta),
        "global_bal_acc_mean": global_balacc_mean,
        "global_bal_acc_std": global_balacc_std,
        "local_bal_acc_mean": local_balacc_mean,
        "local_bal_acc_std": local_balacc_std,
        "gram_metadata": gram_metadata,
    }


def run_thermal_kernel_svm_beta_sweep(
    L: int,
    kinds: list[str] | tuple[str, ...] = ("ipr", "hs"),
    source: str = "processed",
    test_frac: float = 0.10,
    local_delta: float = 0.25,
    n_splits: int = 10,
    rng_seed: int = 12345,
    C: float = 1.0,
    show_plot_title: bool = True,
):
    """Run thermal TFIM kernel-SVM beta sweeps for one L and plot all kernels together.

    The plot contains IPR and HS kernel results on the same axes with a reversed beta axis from 2 to 0.
    """
    source = normalize_thermal_source(source)
    kinds = [normalize_thermal_kind(kind) for kind in kinds]
    if not kinds:
        raise ValueError("kinds must contain at least one kernel kind.")
    if len(set(kinds)) != len(kinds):
        raise ValueError("kinds must not contain duplicates.")

    betas = available_thermal_betas(kind=kinds[0], L=int(L), source=source)
    if not betas:
        raise ValueError("No beta values found for the requested thermal data.")

    results_by_kind = {}
    for kind in kinds:
        results_by_kind[kind] = _thermal_kernel_svm_beta_sweep_for_kind(
            kind=kind,
            L=int(L),
            betas=betas,
            source=source,
            phase_boundary=1.0,
            test_frac=test_frac,
            local_delta=local_delta,
            n_splits=n_splits,
            rng_seed=rng_seed,
            C=C,
        )

    set_ipr_plot_style()
    fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)

    colors = {"ipr": "#1b9e77", "hs": "#d95f02"}
    markers = {"ipr": "s", "hs": "o"}
    for kind in kinds:
        res = results_by_kind[kind]
        color = colors.get(kind, None)
        marker = markers.get(kind, "o")
        label_prefix = kind.upper()
        ax.errorbar(
            betas,
            res["global_bal_acc_mean"],
            yerr=res["global_bal_acc_std"],
            marker=marker,
            lw=2.0,
            capsize=3,
            ls="-",
            color=color,
            label=f"{label_prefix} global",
        )
        ax.errorbar(
            betas,
            res["local_bal_acc_mean"],
            yerr=res["local_bal_acc_std"],
            marker=marker,
            lw=2.0,
            capsize=3,
            ls="--",
            color=color,
            label=fr"{label_prefix} local, $\delta={local_delta}$",
        )

    if show_plot_title:
        ax.set_title(f"Thermal TFIM kernel SVM (L={int(L)}, {source}, {n_splits} splits)")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(-0.1, 1.05)
    ax.set_xlim(2.0, 0.0)
    ax.set_xticks(np.arange(2.0, -1e-9, -0.2))
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.grid(which="major", alpha=0.18, lw=0.7)
    ax.grid(which="minor", alpha=0.07, lw=0.5)
    ax.minorticks_on()
    ax.legend(frameon=False, loc="best")

    results = {
        "source": source,
        "L": int(L),
        "kinds": kinds,
        "betas": betas,
        "phase_boundary": 1.0,
        "n_splits": int(n_splits),
        "test_frac": float(test_frac),
        "local_delta": float(local_delta),
        "beta_xlim": (2.0, 0.0),
        "by_kind": results_by_kind,
    }

    print(f"THERMAL experiment: L={int(L)}, source={source}, kinds={kinds}")
    for kind in kinds:
        res = results_by_kind[kind]
        print(f"\n{kind.upper()} kernel:")
        for i, beta in enumerate(betas):
            print(
                f"beta={beta:g}: global = {res['global_bal_acc_mean'][i]:.4f} "
                f"+/- {res['global_bal_acc_std'][i]:.4f}; "
                f"local = {res['local_bal_acc_mean'][i]:.4f} "
                f"+/- {res['local_bal_acc_std'][i]:.4f}"
            )

    return fig, ax, results


