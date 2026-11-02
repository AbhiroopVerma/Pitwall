"""Module 3 — opponent pit-stop hazard model (discrete-time logistic, numpy only).

h_t = P(pit at end of lap t | survived to t, X_t) = sigmoid(b . X_t)
Features (all known BEFORE the lap-t decision, no leakage):
  [bias, tyre_age, lap_fraction, deg_penalty, gap_behind(+opp trails), stops_done]
Multi-lap: P(pit within k) = 1 - prod(1 - h_{t+j}), ages projected forward.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .state import RaceSnapshot, Replay
from .degradation import DegradationEstimator


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


@dataclass
class PitForecast:
    hazards: list[float]       # conditional h for laps t+1 .. t+k
    p_within: dict[int, float]  # k -> cumulative P(pit within k)
    pit_dist: list[float]       # proper distribution over first-pit lap (sums to ~1 w/ tail)


class PitHazardModel:
    def __init__(self, l2: float = 1.0, lr: float = 0.5, iters: int = 2000):
        self.l2, self.lr, self.iters = l2, lr, iters
        self.beta: np.ndarray | None = None

    # -- features -----------------------------------------------------------------
    @staticmethod
    def feats(tyre_age: int, lap: int, laps_total: int, deg_penalty: float,
              gap_opp_behind_own: float, stops_done: int) -> np.ndarray:
        return np.array([1.0, tyre_age / 20.0, lap / laps_total,
                         deg_penalty, gap_opp_behind_own / 10.0,
                         stops_done / 3.0], float)

    def _design(self, replay: Replay, deg: DegradationEstimator,
                upto_lap: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Build supervised car-lap rows for the OPPONENT using only history<=lap per row."""
        rows, labels = [], []
        laps = replay.available_laps()
        if upto_lap is not None:
            laps = [l for l in laps if l <= upto_lap]
        for lap in laps:
            snap = replay.state_at(lap)
            hist = list(snap.history_opp)
            if len([r for r in hist if not r.pitted and not r.safety_car]) < 6:
                continue
            try:
                local = DegradationEstimator().fit(hist, replay.laps_total)
                est = local.estimate(hist, replay.laps_total, lap)
            except ValueError:
                continue
            opp = snap.opp
            rows.append(self.feats(opp.tyre_age, lap, replay.laps_total,
                                   est.penalty, -snap.gap_own_behind_opp, opp.stops_done))
            # label: did opp pit at END of this lap?
            rec = next(r for r in hist if r.lap == lap)
            labels.append(1.0 if rec.pitted else 0.0)
        return np.asarray(rows), np.asarray(labels)

    # -- training -----------------------------------------------------------------
    def fit(self, replays: list[Replay], pos_weight: float = 8.0) -> "PitHazardModel":
        """Fit with upweighted positives: stops are rare (~5% of laps), so an
        unweighted fit learns 'never pit'. pos_weight rebalances the loss."""
        Xs, ys = [], []
        for rp in replays:
            deg = DegradationEstimator()
            X, y = self._design(rp, deg)
            if len(y):
                Xs.append(X)
                ys.append(y)
        X = np.vstack(Xs)
        y = np.concatenate(ys)
        # standardise non-bias cols for stable GD
        self.mu_ = X[:, 1:].mean(0)
        self.sd_ = X[:, 1:].std(0) + 1e-9
        Xs_ = np.column_stack([X[:, [0]], (X[:, 1:] - self.mu_) / self.sd_])
        b = np.zeros(Xs_.shape[1])
        w = np.where(y == 1.0, pos_weight, 1.0)
        wsum = w.sum()
        for _ in range(self.iters):
            p = sigmoid(Xs_ @ b)
            g = Xs_.T @ (w * (p - y)) / wsum
            g[1:] += self.l2 * b[1:] / wsum
            b -= self.lr * g
        self.beta = b
        return self

    def _prep(self, f: np.ndarray) -> np.ndarray:
        return np.concatenate([[1.0], (f[1:] - self.mu_) / self.sd_])

    def hazard(self, f: np.ndarray) -> float:
        return float(sigmoid(self._prep(f) @ self.beta))

    # -- forecasting ---------------------------------------------------------------
    def forecast(self, snap: RaceSnapshot, est_penalty: float,
                 est_slope: float, k: int = 5) -> PitForecast:
        hazards = []
        for j in range(1, k + 1):
            f = self.feats(snap.opp.tyre_age + j - 1, snap.lap + j - 1,
                            snap.laps_total, est_penalty + est_slope * (j - 1),
                            -snap.gap_own_behind_opp, snap.opp.stops_done)
            hazards.append(self.hazard(f))
        p_within = {}
        surv = 1.0
        for i, h in enumerate(hazards, 1):
            surv *= (1 - h)
            p_within[i] = 1 - surv
        # first-pit distribution + tail mass (no pit within k)
        dist, s = [], 1.0
        for h in hazards:
            dist.append(s * h)
            s *= (1 - h)
        dist.append(s)
        return PitForecast(hazards, p_within, dist)


def brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y_true) ** 2))


def log_loss(y_true: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def age_only_baseline(tyre_age: int) -> float:
    """Simple baseline the learned model must beat: pit prob ramps after age 12."""
    return float(sigmoid(np.array([(tyre_age - 14.0) / 2.0]))[0])
