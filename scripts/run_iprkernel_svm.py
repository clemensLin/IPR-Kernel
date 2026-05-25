from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC

from data_config import DATASETS, available_l_values, normalize_dataset
from finite_shot_noise import compute_ipr_gram_from_features, make_finite_shot_noisy_ipr_dataset
from plot_gram_matrices import compute_ipr_gram_matrix
from plot_ipr_curves import set_ipr_plot_style


def _phase_labels(x: np.ndarray, phase_boundary: float) -> np.ndarray:
    """Convert parameter values into binary phase labels."""
    return (np.asarray(x, dtype=float) >= float(phase_boundary)).astype(int)


def _balanced_global_test_indices(y: np.ndarray, test_frac: float, rng: np.random.Generator) -> np.ndarray:
    """Sample a balanced test set from the full parameter range."""
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    n_test = int(round(float(test_frac) * len(y)))
    n_each = max(1, n_test // 2)
    n_each = min(n_each, len(idx0), len(idx1))

    test0 = rng.choice(idx0, size=n_each, replace=False)
    test1 = rng.choice(idx1, size=n_each, replace=False)
    return np.concatenate([test0, test1])


def _balanced_local_test_indices(
    x: np.ndarray,
    y: np.ndarray,
    phase_boundary: float,
    local_delta: float,
    test_frac: float,
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Sample a balanced test set inside a window around the phase boundary."""
    x = np.asarray(x, dtype=float)
    in_window = (x >= phase_boundary - local_delta) & (x <= phase_boundary + local_delta)
    idx0 = np.where(in_window & (y == 0))[0]
    idx1 = np.where(in_window & (y == 1))[0]

    n_test = int(round(float(test_frac) * len(y)))
    n_each = max(1, n_test // 2)
    n_each = min(n_each, len(idx0), len(idx1))
    if n_each == 0:
        return None

    test0 = rng.choice(idx0, size=n_each, replace=False)
    test1 = rng.choice(idx1, size=n_each, replace=False)
    return np.concatenate([test0, test1])


def _run_precomputed_svm_split(
    K: np.ndarray,
    y: np.ndarray,
    test_idx: np.ndarray,
    C: float,
) -> float:
    """Train and evaluate one precomputed-kernel SVM split."""
    train_mask = np.ones(len(y), dtype=bool)
    train_mask[test_idx] = False
    train_idx = np.where(train_mask)[0]

    K_train = K[np.ix_(train_idx, train_idx)]
    K_test = K[np.ix_(test_idx, train_idx)]
    y_train = y[train_idx]
    y_test = y[test_idx]

    clf = SVC(kernel="precomputed", class_weight="balanced", C=float(C))
    clf.fit(K_train, y_train)
    y_pred = clf.predict(K_test)

    return float(balanced_accuracy_score(y_test, y_pred))


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Return nan-aware mean and standard deviation."""
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)), float(np.nanstd(arr, ddof=1 if np.sum(~np.isnan(arr)) > 1 else 0))


def _evaluate_svm_for_kernel(
    x: np.ndarray,
    K: np.ndarray,
    phase_boundary: float,
    test_frac: float,
    local_delta: float,
    n_splits: int,
    rng_seed: int,
    C: float,
) -> dict:
    """Evaluate global and local balanced accuracy for one Gram matrix."""
    y = _phase_labels(x, phase_boundary)
    if np.unique(y).size < 2:
        return {
            "global_bal_acc_mean": np.nan,
            "global_bal_acc_std": np.nan,
            "local_bal_acc_mean": np.nan,
            "local_bal_acc_std": np.nan,
        }

    global_balacc_runs = []
    local_balacc_runs = []

    for split_id in range(int(n_splits)):
        rng = np.random.default_rng(int(rng_seed) + split_id)

        global_test_idx = _balanced_global_test_indices(y, test_frac, rng)
        global_balacc_runs.append(_run_precomputed_svm_split(K, y, global_test_idx, C=C))

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
        else:
            local_balacc_runs.append(_run_precomputed_svm_split(K, y, local_test_idx, C=C))

    global_balacc_mean, global_balacc_std = _mean_std(global_balacc_runs)
    local_balacc_mean, local_balacc_std = _mean_std(local_balacc_runs)

    return {
        "global_bal_acc_mean": global_balacc_mean,
        "global_bal_acc_std": global_balacc_std,
        "local_bal_acc_mean": local_balacc_mean,
        "local_bal_acc_std": local_balacc_std,
    }


def run_iprkernel_svm_experiment(
    dataset: str,
    L_values: list[int] | None = None,
    basis: list[str] | tuple[str, ...] = ("IPR X", "IPR Z"),
    phase_boundary: float | None = None,
    test_frac: float = 0.20,
    local_delta: float = 0.25,
    n_splits: int = 10,
    rng_seed: int = 12345,
    use_normalized_kernel: bool = False,
    C: float = 1.0,
    make_plot: bool = True,
    show_plot_title: bool = True,
    print_report: bool = True,
):
    """Run global and local SVM tests using processed raw IPR-kernel Gram matrices.

    The global experiment samples balanced test points from the full parameter range.
    The local experiment samples balanced test points from a window around the phase
    boundary. Both experiments train an SVM with a precomputed IPR kernel.
    """
    dataset = normalize_dataset(dataset)
    cfg = DATASETS[dataset]
    if phase_boundary is None:
        phase_boundary = float(cfg["critical_point"])
    if L_values is None:
        L_values = available_l_values(dataset)

    used_L = []
    global_balacc_mean, global_balacc_std = [], []
    local_balacc_mean, local_balacc_std = [], []

    for L in L_values:
        gram = compute_ipr_gram_matrix(
            dataset=dataset,
            L=L,
            basis=basis,
            normalize=use_normalized_kernel,
        )
        x = np.asarray(gram["x"], dtype=float)
        K = np.asarray(gram["K"], dtype=float)
        used_L.append(int(L))
        metrics = _evaluate_svm_for_kernel(
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

    fig, ax = None, None
    if make_plot:
        set_ipr_plot_style()
        fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)

        color_global = "#1b9e77"
        color_local = "#d95f02"

        ax.errorbar(
            used_L,
            global_balacc_mean,
            yerr=global_balacc_std,
            marker="s",
            lw=2.0,
            capsize=3,
            ls="-",
            color=color_global,
            label="Balanced accuracy (global)",
        )
        ax.errorbar(
            used_L,
            local_balacc_mean,
            yerr=local_balacc_std,
            marker="o",
            lw=2.0,
            capsize=3,
            ls="--",
            color=color_local,
            label=fr"Balanced accuracy (local, $\delta={local_delta}$)",
        )

        if show_plot_title:
            norm_txt = "normalized" if use_normalized_kernel else "unnormalized"
            ax.set_title(f"{cfg['title_name']} IPR-kernel SVM ({basis}, {norm_txt}, {n_splits} splits)")
        ax.set_xlabel("L")
        ax.set_ylabel("Balanced accuracy")
        ax.set_ylim(-0.1, 1.05)
        ax.set_xticks(used_L)
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.grid(which="major", alpha=0.18, lw=0.7)
        ax.grid(which="minor", alpha=0.07, lw=0.5)
        ax.minorticks_on()
        ax.legend(frameon=False, loc="best")

    results = {
        "dataset": dataset,
        "basis": basis,
        "L": used_L,
        "phase_boundary": float(phase_boundary),
        "kernel_normalized": bool(use_normalized_kernel),
        "n_splits": int(n_splits),
        "test_frac": float(test_frac),
        "local_delta": float(local_delta),
        "global_bal_acc_mean": global_balacc_mean,
        "global_bal_acc_std": global_balacc_std,
        "local_bal_acc_mean": local_balacc_mean,
        "local_bal_acc_std": local_balacc_std,
    }

    if print_report:
        print("GLOBAL experiment:")
        for i, L in enumerate(used_L):
            print(
                f"L={L}: balanced accuracy = {global_balacc_mean[i]:.4f} +/- {global_balacc_std[i]:.4f}"
            )

        print("\nLOCAL experiment:")
        for i, L in enumerate(used_L):
            print(
                f"L={L}: balanced accuracy = {local_balacc_mean[i]:.4f} +/- {local_balacc_std[i]:.4f}"
            )

    return fig, ax, results



#-------------Noisy finite-shot IPR-kernel SVM experiments with multiple noise seeds and aggregation------------------

def _plot_svm_metric_summary(
    L_values: list[int],
    global_balacc_mean: list[float],
    global_balacc_std: list[float],
    local_balacc_mean: list[float],
    local_balacc_std: list[float],
    local_delta: float,
    title: str | None = None,
):
    """Plot global and local balanced accuracy versus system size."""
    set_ipr_plot_style()
    fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    color_global = "#1b9e77"
    color_local = "#d95f02"

    ax.errorbar(
        L_values,
        global_balacc_mean,
        yerr=global_balacc_std,
        marker="s",
        lw=2.0,
        capsize=3,
        ls="-",
        color=color_global,
        label="Balanced accuracy (global)",
    )
    ax.errorbar(
        L_values,
        local_balacc_mean,
        yerr=local_balacc_std,
        marker="o",
        lw=2.0,
        capsize=3,
        ls="--",
        color=color_local,
        label=fr"Balanced accuracy (local, $\delta={local_delta}$)",
    )

    if title:
        ax.set_title(title)
    ax.set_xlabel("L")
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(-0.1, 1.05)
    ax.set_xticks(L_values)
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.grid(which="major", alpha=0.18, lw=0.7)
    ax.grid(which="minor", alpha=0.07, lw=0.5)
    ax.minorticks_on()
    ax.legend(frameon=False, loc="best")
    return fig, ax


def run_noisy_iprkernel_svm_experiment(
    dataset: str,
    L_values: list[int] | None = None,
    basis: list[str] | tuple[str, ...] = ("IPR X", "IPR Z"),
    N_s: int = 10000,
    noise_seeds: list[int] | None = None,
    n_noise_seeds: int = 10,
    clip_eps: float | None = None,
    phase_boundary: float | None = None,
    test_frac: float = 0.10,
    local_delta: float = 0.25,
    n_splits: int = 10,
    rng_seed: int = 12345,
    use_normalized_kernel: bool = True,
    C: float = 1.0,
    make_plot: bool = True,
    show_plot_title: bool = True,
    print_report: bool = True,
):
    """Run finite-shot noisy IPR-kernel SVM experiments.

    For each noise seed, a noisy processed IPR dataset is generated with Gaussian
    noise of scale 1/sqrt(N_s). Balanced accuracy is averaged over SVM splits for each
    noise seed, then aggregated across noise seeds.
    """
    dataset = normalize_dataset(dataset)
    cfg = DATASETS[dataset]
    if phase_boundary is None:
        phase_boundary = float(cfg["critical_point"])
    if L_values is None:
        L_values = available_l_values(dataset)
    L_values = sorted(int(v) for v in L_values)
    if noise_seeds is None:
        noise_seeds = list(range(int(n_noise_seeds)))
    else:
        noise_seeds = [int(seed) for seed in noise_seeds]

    per_seed_results = []
    noise_stats_by_seed = {}

    for noise_seed in noise_seeds:
        noisy_data, noise_stats = make_finite_shot_noisy_ipr_dataset(
            dataset=dataset,
            L_values=L_values,
            basis=basis,
            N_s=int(N_s),
            rng_seed=int(noise_seed),
            clip_eps=clip_eps,
        )
        noise_stats_by_seed[int(noise_seed)] = noise_stats

        seed_metrics = {
            "noise_seed": int(noise_seed),
            "L": [],
            "global_bal_acc_mean": [],
            "local_bal_acc_mean": [],
        }

        for L in L_values:
            entry = noisy_data[L]
            K = compute_ipr_gram_from_features(entry["features"], normalize=use_normalized_kernel)
            metrics = _evaluate_svm_for_kernel(
                x=np.asarray(entry["x"], dtype=float),
                K=K,
                phase_boundary=float(phase_boundary),
                test_frac=float(test_frac),
                local_delta=float(local_delta),
                n_splits=int(n_splits),
                rng_seed=int(rng_seed),
                C=float(C),
            )
            seed_metrics["L"].append(int(L))
            seed_metrics["global_bal_acc_mean"].append(metrics["global_bal_acc_mean"])
            seed_metrics["local_bal_acc_mean"].append(metrics["local_bal_acc_mean"])

        per_seed_results.append(seed_metrics)

    def mean_over_noise(metric_key: str) -> list[float]:
        values = np.asarray([r[metric_key] for r in per_seed_results], dtype=float)
        return np.nanmean(values, axis=0).tolist()

    def std_over_noise(metric_key: str) -> list[float]:
        values = np.asarray([r[metric_key] for r in per_seed_results], dtype=float)
        return np.nanstd(values, axis=0, ddof=1 if len(per_seed_results) > 1 else 0).tolist()

    results = {
        "dataset": dataset,
        "basis": basis,
        "L": L_values,
        "N_s": int(N_s),
        "noise_seeds": noise_seeds,
        "n_noise_seeds": len(noise_seeds),
        "n_splits_per_noise_seed": int(n_splits),
        "svm_rng_seed": int(rng_seed),
        "phase_boundary": float(phase_boundary),
        "kernel_normalized": bool(use_normalized_kernel),
        "test_frac": float(test_frac),
        "local_delta": float(local_delta),
        "global_bal_acc_mean": mean_over_noise("global_bal_acc_mean"),
        "global_bal_acc_std_over_noise_seed": std_over_noise("global_bal_acc_mean"),
        "local_bal_acc_mean": mean_over_noise("local_bal_acc_mean"),
        "local_bal_acc_std_over_noise_seed": std_over_noise("local_bal_acc_mean"),
        "per_seed_results": per_seed_results,
        "noise_stats_by_seed": noise_stats_by_seed,
    }

    fig, ax = None, None
    if make_plot:
        title = None
        if show_plot_title:
            norm_txt = "normalized" if use_normalized_kernel else "unnormalized"
            title = (
                f"{cfg['title_name']} noisy IPR-kernel SVM "
                f"({basis}, N_s={N_s}, {len(noise_seeds)} noise seeds, {norm_txt})"
            )
        fig, ax = _plot_svm_metric_summary(
            L_values=L_values,
            global_balacc_mean=results["global_bal_acc_mean"],
            global_balacc_std=results["global_bal_acc_std_over_noise_seed"],
            local_balacc_mean=results["local_bal_acc_mean"],
            local_balacc_std=results["local_bal_acc_std_over_noise_seed"],
            local_delta=float(local_delta),
            title=title,
        )

    if print_report:
        print(f"NOISY experiment: N_s={int(N_s)}, noise seeds={noise_seeds}")
        for i, L in enumerate(L_values):
            print(
                f"L={L}: global balanced accuracy = {results['global_bal_acc_mean'][i]:.4f} "
                f"+/- {results['global_bal_acc_std_over_noise_seed'][i]:.4f}; "
                f"local balanced accuracy = {results['local_bal_acc_mean'][i]:.4f} "
                f"+/- {results['local_bal_acc_std_over_noise_seed'][i]:.4f}"
            )

    return fig, ax, results


def run_noisy_iprkernel_svm_ns_sweep(
    dataset: str,
    N_s_values: list[int],
    L_values: list[int],
    basis: list[str] | tuple[str, ...] = ("IPR X", "IPR Z"),
    noise_seeds: list[int] | None = None,
    n_noise_seeds: int = 10,
    phase_boundary: float | None = None,
    test_frac: float = 0.10,
    local_delta: float = 0.25,
    n_splits: int = 10,
    rng_seed: int = 12345,
    use_normalized_kernel: bool = True,
    C: float = 1.0,
    show_plot_title: bool = False,
    make_plot: bool = True,
):
    """Sweep finite-shot sample counts and plot balanced accuracy.

    Each `N_s` value is evaluated by averaging split-averaged balanced accuracy over
    independent noise seeds. The returned table contains numeric mean/std
    columns and formatted result columns.
    """
    dataset = normalize_dataset(dataset)
    global_mean_key = "global_bal_acc_mean"
    global_std_key = "global_bal_acc_std_over_noise_seed"
    local_mean_key = "local_bal_acc_mean"
    local_std_key = "local_bal_acc_std_over_noise_seed"

    if noise_seeds is None:
        noise_seeds = list(range(int(n_noise_seeds)))
    else:
        noise_seeds = [int(seed) for seed in noise_seeds]

    L_values = [int(L) for L in L_values]
    N_s_values = [int(N_s) for N_s in N_s_values]
    sweep_results = {}

    for N_s in N_s_values:
        _, _, res = run_noisy_iprkernel_svm_experiment(
            dataset=dataset,
            L_values=L_values,
            basis=basis,
            N_s=N_s,
            noise_seeds=noise_seeds,
            clip_eps=1.0 / float(N_s),
            phase_boundary=phase_boundary,
            test_frac=test_frac,
            local_delta=local_delta,
            n_splits=n_splits,
            rng_seed=rng_seed,
            use_normalized_kernel=use_normalized_kernel,
            C=C,
            make_plot=False,
            print_report=False,
        )
        sweep_results[N_s] = res

    fig, ax = None, None
    if make_plot:
        set_ipr_plot_style()
        fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)

        color_global = "#1b9e77"
        color_local = "#d95f02"
        global_handles, global_labels = [], []
        local_handles, local_labels = [], []
        sorted_ns = sorted(N_s_values)

        for rank, N_s in enumerate(sorted_ns):
            res = sweep_results[N_s]
            t_rank = 0.0 if len(sorted_ns) == 1 else rank / (len(sorted_ns) - 1)
            alpha = 0.12 + 0.83 * t_rank

            g = ax.errorbar(
                L_values,
                res[global_mean_key],
                yerr=res[global_std_key],
                marker="s",
                lw=1.9,
                capsize=3,
                ls="-",
                color=color_global,
                alpha=alpha,
            )
            l = ax.errorbar(
                L_values,
                res[local_mean_key],
                yerr=res[local_std_key],
                marker="o",
                lw=1.9,
                capsize=3,
                ls="-",
                color=color_local,
                alpha=alpha,
            )

            global_handles.append(g)
            global_labels.append(fr"Global, $N_s={N_s}$")
            local_handles.append(l)
            local_labels.append(fr"Local, $N_s={N_s}$")

        ax.set_xlabel("L")
        ax.set_ylabel("Balanced accuracy")
        ax.set_ylim(-0.1, 1.05)
        ax.set_xticks(L_values)
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.minorticks_on()
        ax.grid(which="major", alpha=0.18, lw=0.7)
        ax.grid(which="minor", alpha=0.07, lw=0.5)

        if show_plot_title:
            ax.set_title(
                f"Balanced accuracy: global and local finite-shot noise sweep "
                f"({len(noise_seeds)} noise seeds, {n_splits} splits)"
            )

        leg_left = ax.legend(
            global_handles,
            global_labels,
            frameon=False,
            loc="lower left",
            bbox_to_anchor=(0.01, 0.01),
            title="Global",
        )
        ax.add_artist(leg_left)
        ax.legend(
            local_handles,
            local_labels,
            frameon=False,
            loc="lower left",
            bbox_to_anchor=(0.24, 0.01),
            title="Local",
        )

    rows = []
    for N_s in sorted(N_s_values):
        res = sweep_results[N_s]
        for i, L in enumerate(L_values):
            g_mean = float(res[global_mean_key][i])
            g_std = float(res[global_std_key][i])
            l_mean = float(res[local_mean_key][i])
            l_std = float(res[local_std_key][i])
            rows.append(
                {
                    "L": int(L),
                    "N_s": int(N_s),
                    "global_mean": g_mean,
                    "global_std": g_std,
                    "local_mean": l_mean,
                    "local_std": l_std,
                    "global_mean_pm_std": f"{g_mean:.6f} +/- {g_std:.6f}",
                    "local_mean_pm_std": f"{l_mean:.6f} +/- {l_std:.6f}",
                    "n_noise_seeds": int(len(noise_seeds)),
                    "n_splits_per_noise_seed": int(n_splits),
                    "svm_rng_seed": int(rng_seed),
                    "kernel_normalized": bool(use_normalized_kernel),
                    "metric": "balanced_accuracy",
                }
            )

    table = pd.DataFrame(rows)
    return fig, ax, table, sweep_results
