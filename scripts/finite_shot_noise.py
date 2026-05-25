from __future__ import annotations

import numpy as np

from data_config import (
    DATASETS,
    available_l_values,
    curves_from_basis,
    folder_name,
    normalize_dataset,
    read_curve,
    read_vector,
)


FINITE_SHOT_DATASET = "tfim1d"


def _normalize_finite_shot_dataset(dataset: str) -> str:
    dataset = normalize_dataset(dataset)
    if dataset != FINITE_SHOT_DATASET:
        raise ValueError("finite-shot noisy IPR data is only supported for processed tfim1d.")
    return dataset


def _l_value(dataset: str, L: int | str) -> int:
    return int(folder_name(dataset, L).split("=", 1)[1].split("x", 1)[0])


def load_processed_ipr_features(
    dataset: str,
    L: int | str,
    basis: list[str] | tuple[str, ...] = ("IPR X", "IPR Z"),
):
    """Load processed TFIM 1D raw IPR features for one system size."""
    dataset = _normalize_finite_shot_dataset(dataset)
    cfg = DATASETS[dataset]
    curves = curves_from_basis(dataset, basis)
    folder = cfg["processed"] / folder_name(dataset, L)
    if not folder.exists():
        raise FileNotFoundError(f"Processed data folder not found: {folder}")

    x = read_vector(folder / cfg["x_file"])
    feature_columns = [read_curve(folder, cfg, curve, "raw") for curve in curves]
    features = np.column_stack(feature_columns)

    return {
        "dataset": dataset,
        "L": _l_value(dataset, L),
        "folder": folder_name(dataset, L),
        "x": x,
        "features": features,
        "curves": curves,
        "basis": basis,
    }


def add_finite_shot_noise_to_features(
    features: np.ndarray,
    N_s: int = 10000,
    rng: np.random.Generator | None = None,
    clip_eps: float | None = None,
):
    """Add independent Gaussian finite-shot noise to raw IPR features."""
    if int(N_s) <= 0:
        raise ValueError("N_s must be a positive integer.")
    if rng is None:
        rng = np.random.default_rng()
    if clip_eps is None:
        clip_eps = 1.0 / float(N_s)

    features = np.asarray(features, dtype=float)
    sigma = 1.0 / np.sqrt(float(N_s))
    noise = rng.normal(0.0, sigma, size=features.shape)
    noisy_features = features + noise

    if clip_eps is not None and clip_eps > 0.0:
        noisy_features = np.clip(noisy_features, float(clip_eps), 1.0)

    stats = {
        "sigma": float(sigma),
        "noise_mean": float(np.mean(noise)),
        "noise_std": float(np.std(noise)),
        "clip_eps": None if clip_eps is None else float(clip_eps),
    }
    return noisy_features, stats


def make_finite_shot_noisy_ipr_dataset(
    dataset: str,
    L_values: list[int] | None = None,
    basis: list[str] | tuple[str, ...] = ("IPR X", "IPR Z"),
    N_s: int = 10000,
    rng_seed: int | None = 12345,
    clip_eps: float | None = None,
):
    """Construct an in-memory noisy processed TFIM 1D IPR dataset.

    One RNG stream is used across all selected system sizes. Independent noise is
    added to every parameter point and every selected IPR basis feature.
    """
    dataset = _normalize_finite_shot_dataset(dataset)
    if L_values is None:
        L_values = available_l_values(dataset)

    rng = np.random.default_rng() if rng_seed is None else np.random.default_rng(int(rng_seed))
    noisy_data = {}
    noise_stats = {}

    for L in sorted(int(v) for v in L_values):
        exact = load_processed_ipr_features(dataset=dataset, L=L, basis=basis)
        noisy_features, stats = add_finite_shot_noise_to_features(
            exact["features"],
            N_s=N_s,
            rng=rng,
            clip_eps=clip_eps,
        )
        exact["features_exact"] = exact["features"]
        exact["features"] = noisy_features
        exact["N_s"] = int(N_s)
        noisy_data[L] = exact
        noise_stats[L] = stats

    return noisy_data, noise_stats


def compute_ipr_gram_from_features(
    features: np.ndarray,
    normalize: bool = False,
) -> np.ndarray:
    """Compute an IPR Gram matrix from raw IPR feature vectors."""
    features = np.asarray(features, dtype=float)
    K = features @ features.T

    if normalize:
        norms = np.linalg.norm(features, axis=1)
        denom = np.outer(norms, norms)
        K = np.divide(K, denom, out=np.zeros_like(K), where=denom > 0)

    return K


def compute_noisy_ipr_gram_matrix(
    dataset: str,
    L: int | str,
    basis: list[str] | tuple[str, ...] = ("IPR X", "IPR Z"),
    N_s: int = 10000,
    rng_seed: int | None = 12345,
    clip_eps: float | None = None,
    normalize: bool = False,
):
    """Compute one finite-shot noisy TFIM 1D IPR Gram matrix."""
    dataset = _normalize_finite_shot_dataset(dataset)
    data, stats = make_finite_shot_noisy_ipr_dataset(
        dataset=dataset,
        L_values=[_l_value(dataset, L)],
        basis=basis,
        N_s=N_s,
        rng_seed=rng_seed,
        clip_eps=clip_eps,
    )
    L_key = next(iter(data))
    entry = data[L_key]
    K = compute_ipr_gram_from_features(entry["features"], normalize=normalize)

    return {
        **entry,
        "K": K,
        "normalize": bool(normalize),
        "noise_stats": stats[L_key],
    }

