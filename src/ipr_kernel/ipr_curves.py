from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np



CURVE_STYLES = {
    "IPR X": {"color": "#1b9e77", "label": r"$I^{(X)}$", "linestyle": "-"},
    "IPR Z": {"color": "#d95f02", "label": r"$I^{(Z)}$", "linestyle": "--"},
    "IPR XY": {"color": "#1b9e77", "label": r"$I^{(XY)}$", "linestyle": "-"},
}


HEISENBERG_CURVE_STYLES = {
    "IPR XY": {"color": "#1b9e77", "label": r"$I^{(XY)}$", "linestyle": "-"},
    "IPR Z": {"color": "#d95f02", "label": r"$I^{(Z)}$", "linestyle": "--"},
    "IPR X": {"color": "#1b9e77", "label": r"$I^{(X)}$ (dotted)", "linestyle": ":"},
}


def set_ipr_plot_style() -> None:
    """Apply the shared publication-style matplotlib settings."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "mathtext.fontset": "dejavusans",
        "font.size": 15,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 14,
    })


def _nearest_crossing(x: np.ndarray, y_a: np.ndarray, y_b: np.ndarray, reference_x: float) -> tuple[float, float] | None:
    """Return the curve crossing closest to a reference x-value, if one exists."""
    x = np.asarray(x, dtype=float)
    y_a = np.asarray(y_a, dtype=float)
    y_b = np.asarray(y_b, dtype=float)

    order = np.argsort(x)
    x = x[order]
    y_a = y_a[order]
    y_b = y_b[order]
    diff = y_a - y_b

    exact = np.where(np.isclose(diff, 0.0, atol=1e-12, rtol=1e-10))[0]
    candidates = []
    for i in exact:
        candidates.append((float(x[i]), float(y_a[i])))

    brackets = np.where(diff[:-1] * diff[1:] < 0.0)[0]
    for i in brackets:
        x0, x1 = x[i], x[i + 1]
        d0, d1 = diff[i], diff[i + 1]
        t = -d0 / (d1 - d0)
        x_cross = x0 + t * (x1 - x0)
        y_cross = y_a[i] + t * (y_a[i + 1] - y_a[i])
        candidates.append((float(x_cross), float(y_cross)))

    if not candidates:
        return None
    return min(candidates, key=lambda p: abs(p[0] - reference_x))


