"""Per-target probability calibration (spec 05, Req 2.2).

Raw classifier scores are not comparable across targets: one target's 0.6 can be another's 0.2.
Calibration maps scores to probabilities on held-out validation compounds, so rankings across
targets mean something. The rule is fixed in advance rather than chosen on the data it is fit to:

- **identity** when validation has fewer than 10 of either class: too little data, and an
  unregularized fit on e.g. 4 positives separates them perfectly and saturates to 1.0 (seen in
  the first full run: a raw 0.42 became 1.000);
- **isotonic** regression when validation has ≥ 100 of each class (flexible, needs data);
- **regularized Platt** scaling otherwise: logistic recalibration of the log-odds,
  p = σ(a·logit(raw) + b), with an L2 penalty pulling (a, b) toward the identity (1, 0). Small
  validation sets stay close to the raw scores; large ones are driven by the data.

Isotonic maps can tie at their top bin, so rankings break ties by raw score (`ranking_score`).
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression

ISOTONIC_MIN_PER_CLASS = 100
CALIBRATION_MIN_PER_CLASS = 10
PLATT_L2 = 1.0
_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _fit_platt(z: np.ndarray, y: np.ndarray, l2: float) -> tuple[float, float]:
    def loss(theta):
        a, b = theta
        s = a * z + b
        # logistic log-loss in a numerically stable form, plus the pull toward identity
        nll = np.sum(np.logaddexp(0, s) - y * s)
        return nll + l2 * ((a - 1) ** 2 + b ** 2)

    res = minimize(loss, x0=np.array([1.0, 0.0]), method="L-BFGS-B")
    return float(res.x[0]), float(res.x[1])


def fit_calibrator(raw: np.ndarray, y: np.ndarray, l2: float = PLATT_L2) -> dict:
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if min(n_pos, n_neg) < CALIBRATION_MIN_PER_CLASS:
        return {"method": "identity"}
    if min(n_pos, n_neg) >= ISOTONIC_MIN_PER_CLASS:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, y)
        return {"method": "isotonic", "x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}
    a, b = _fit_platt(_logit(raw), y.astype(np.float64), l2)
    return {"method": "platt", "a": a, "b": b}


def apply_calibrator(cal: dict, raw: np.ndarray) -> np.ndarray:
    if cal["method"] == "identity":
        return raw
    if cal["method"] == "isotonic":
        return np.interp(raw, cal["x"], cal["y"])
    return expit(cal["a"] * _logit(raw) + cal["b"])


def ranking_score(calibrated: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """Calibrated probability with the raw score as a tie-breaker (isotonic maps plateau)."""
    return calibrated + 1e-6 * raw
