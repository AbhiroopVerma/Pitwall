"""Module 2 — tyre-degradation and pace estimation from public lap times.

Fuel-corrected lap time = stint baseline + D(age)
    D(age) = b1*a + b2*a^2, with a = age * compound factor / 10, b1, b2 >= 0

Why this shape:
- In a first stint tyre age and fuel load move in perfect lockstep, so lap times
  alone cannot separate them (spec §5). Fuel is therefore a stated assumption
  (FUEL_S_PER_LAP), not a fitted term.
- A baseline per stint stops a compound change being read as tyre wear.
- The non-negative coefficients rule out tyres that get faster with age.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .state import LapRecord

# Assumption: lap-time gain per lap of fuel burned. Published estimates for F1
# run from about 0.03 to 0.06 s/lap.
FUEL_S_PER_LAP = 0.04
# Assumption: relative wear rate, because a short stint cannot measure its own.
COMPOUND_WEAR = {"SOFT": 1.4, "MEDIUM": 1.0, "HARD": 0.7}
MIN_CLEAN_LAPS = 6
OUTLIER_S = 3.0  # traffic, mistakes and cool-down laps inside a stint
MIN_STD = 0.1


def wear_factor(compound: str | None) -> float:
    return COMPOUND_WEAR.get(compound or "", 1.0)


def fuel_effect(lap: int, laps_total: int) -> float:
    return FUEL_S_PER_LAP * (laps_total - lap)


def clean_laps(history: list[LapRecord], laps_total: int) -> list[tuple[LapRecord, int]]:
    """(record, stint index) for laps that represent normal racing pace."""
    tagged, stint = [], 0
    for r in history:
        if (not r.pitted and not r.pit_out and not r.safety_car and r.lap > 1
                and np.isfinite(r.lap_time) and r.compound not in ("UNKNOWN",)):
            tagged.append((r, stint))
        if r.pitted:
            stint += 1
    keep = []
    for s in {t[1] for t in tagged}:
        members = [t for t in tagged if t[1] == s]
        times = [t[0].lap_time - fuel_effect(t[0].lap, laps_total) for t in members]
        mid = float(np.median(times))
        keep += [t for t, v in zip(members, times) if abs(v - mid) <= OUTLIER_S]
    return sorted(keep, key=lambda t: t[0].lap)


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: np.ndarray) -> np.ndarray:
    return np.linalg.solve(X.T @ X + np.diag(lam), X.T @ y)


def _curve_columns(ages: np.ndarray, factors: np.ndarray) -> np.ndarray:
    a = ages * factors / 10.0
    return np.column_stack([a, a ** 2])


def _fit_curve(curve: np.ndarray, base: np.ndarray, y: np.ndarray,
               ridge: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Least squares with b1, b2 >= 0 via their small active set.

    Returns (b1 b2, intercepts, posterior precision inverse for all columns).
    """
    nb = base.shape[1]
    best = None
    for mask in ((True, True), (True, False), (False, True), (False, False)):
        cols = [i for i, on in enumerate(mask) if on]
        X = np.column_stack([curve[:, cols], base]) if nb else curve[:, cols]
        lam = np.array([ridge] * len(cols) + [1e-6] * nb)
        if X.shape[1] == 0:
            coef = np.zeros(0)
        else:
            coef = ridge_fit(X, y, lam)
        if len(cols) and np.any(coef[:len(cols)] < 0):
            continue
        rss = float(np.sum((y - X @ coef) ** 2)) if X.shape[1] else float(np.sum(y ** 2))
        if best is None or rss < best[0] - 1e-12:
            full = np.zeros(2)
            full[cols] = coef[:len(cols)]
            best = (rss, full, coef[len(cols):])
    X = np.column_stack([curve, base]) if nb else curve
    cov = np.linalg.inv(X.T @ X + np.diag([ridge, ridge] + [1e-6] * nb))
    return best[1], best[2], cov


@dataclass
class DegradationEstimate:
    penalty: float        # D(age_now), seconds vs fresh
    slope: float          # dD/d_age at current age (s/lap)
    forecast: list[float]  # expected clean lap times for next k laps (stay-out)
    std: float            # next-lap forecast std, includes coefficient uncertainty
    max_extra_laps: int   # laps until penalty exceeds `limit` (default 2.0 s)
    forecast_std: list[float] = field(default_factory=list)
    slope_se: float = 0.0


class DegradationEstimator:
    def __init__(self, ridge: float = 0.1, horizon: int = 5):
        self.ridge = ridge
        self.horizon = horizon
        self.coef_: np.ndarray | None = None  # [b1, b2]
        self.baseline_: float = 0.0           # fuel-free pace of the latest stint
        self.resid_std_: float = 0.25
        self.laps_total_: int = 0
        self.compound_: str | None = None
        self._cov: np.ndarray | None = None
        self._last_stint_col: int | None = None

    # -- fitting -----------------------------------------------------------------
    def _curve_target(self, recs: list[LapRecord], y: np.ndarray,
                      curve: np.ndarray, base: np.ndarray):
        return _fit_curve(curve, base, y, self.ridge)

    def fit(self, history: list[LapRecord], laps_total: int) -> "DegradationEstimator":
        kept = clean_laps(history, laps_total)
        if len(kept) < MIN_CLEAN_LAPS:
            raise ValueError("not enough clean laps to fit degradation model")
        recs = [t[0] for t in kept]
        stints = sorted({t[1] for t in kept})
        idx = {s: i for i, s in enumerate(stints)}
        base = np.zeros((len(kept), len(stints)))
        for row, (_, s) in enumerate(kept):
            base[row, idx[s]] = 1.0
        y = np.array([r.lap_time - fuel_effect(r.lap, laps_total) for r in recs])
        ages = np.array([r.tyre_age for r in recs], float)
        factors = np.array([wear_factor(r.compound) for r in recs])
        curve = _curve_columns(ages, factors)
        coef, intercepts, cov = self._curve_target(recs, y, curve, base)
        self.coef_ = coef
        self.baseline_ = float(intercepts[-1])
        resid = y - curve @ coef - base @ intercepts
        dof = max(len(y) - len(stints) - 2, 1)
        self.resid_std_ = max(float(np.sqrt(np.sum(resid ** 2) / dof)), MIN_STD)
        self.laps_total_ = laps_total
        self.compound_ = history[-1].compound
        self._cov = cov * self.resid_std_ ** 2
        self._last_stint_col = 2 + len(stints) - 1
        return self

    # -- curve -------------------------------------------------------------------
    def penalty_at(self, age: float, compound: str | None = None) -> float:
        a = age * wear_factor(compound) / 10.0
        return float(self.coef_[0] * a + self.coef_[1] * a ** 2)

    def slope_at(self, age: float, compound: str | None = None) -> float:
        f = wear_factor(compound)
        a = age * f / 10.0
        return float((self.coef_[0] + 2 * self.coef_[1] * a) * f / 10.0)

    def pace(self, age: float, lap: int, compound: str | None = None) -> float:
        """Expected clean lap time on this stint's baseline."""
        return (self.baseline_ + self.penalty_at(age, compound)
                + fuel_effect(lap, self.laps_total_))

    def _forecast_std(self, age: float, compound: str | None) -> float:
        a = age * wear_factor(compound) / 10.0
        x = np.zeros(self._cov.shape[0])
        x[0], x[1], x[self._last_stint_col] = a, a ** 2, 1.0
        return float(np.sqrt(self.resid_std_ ** 2 + x @ self._cov @ x))

    # -- outputs -----------------------------------------------------------------
    def estimate(self, history: list[LapRecord], laps_total: int,
                 current_lap: int, penalty_limit: float = 2.0) -> DegradationEstimate:
        assert self.coef_ is not None, "call fit() first"
        cur = history[-1]
        comp = cur.compound
        age0 = 0 if cur.pitted else cur.tyre_age  # a stop resets the next lap
        ages = [age0 + k for k in range(1, self.horizon + 1)]
        forecast = [self.pace(a, current_lap + k + 1, comp) for k, a in enumerate(ages)]
        stds = [self._forecast_std(a, comp) for a in ages]
        extra = 0
        while self.penalty_at(age0 + extra + 1, comp) < penalty_limit and extra < 15:
            extra += 1
        f = wear_factor(comp) / 10.0
        slope_var = (self._cov[0, 0] + 4 * (age0 * f) * self._cov[0, 1]
                     + 4 * (age0 * f) ** 2 * self._cov[1, 1]) * f ** 2
        return DegradationEstimate(self.penalty_at(cur.tyre_age, comp),
                                   self.slope_at(cur.tyre_age, comp),
                                   forecast, stds[0], extra, stds,
                                   float(np.sqrt(max(slope_var, 0.0))))
