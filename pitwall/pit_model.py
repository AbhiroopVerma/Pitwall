"""Module 3 — opponent pit-stop hazard model (discrete-time logistic, numpy only).

h_t = P(car pits at end of lap t | it has reached lap t, X_t) = sigmoid(b . X_t)

X_t holds ONLY what is known before lap t is driven: everything comes from laps
up to t-1. In particular ``stops`` excludes the stop being predicted.

The shipped weights are trained on real races by ``pitwall.train_hazard``.
Multi-lap: P(pit within k) = 1 - prod(1 - h_{t+j}), j = 1..k, ages projected.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .degradation import DegradationEstimator, MIN_CLEAN_LAPS, clean_laps
from .state import LapRecord, RaceSnapshot, Replay

WEIGHTS = Path(__file__).resolve().parent / "assets" / "hazard_weights.json"
FEATURES = ["bias", "age", "age2", "age_soft", "age_hard", "soft", "hard",
            "lap_frac", "lap_frac2", "laps_left_short", "first_stop_pending",
            "stops", "penalty", "slope", "neutral", "neutral_age"]
AGE_ONLY = [0, 1, 2]  # columns of the tyre-age-only baseline required by spec §12


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def hazard_features(age: float, compound: str, lap: int, laps_total: int,
                    stops: int, penalty: float, slope: float, neutral: bool) -> np.ndarray:
    a = age / 20.0
    soft, hard = float(compound == "SOFT"), float(compound == "HARD")
    frac = lap / laps_total
    left = laps_total - lap
    return np.array([1.0, a, a ** 2, a * soft, a * hard, soft, hard, frac, frac ** 2,
                     float(left <= 5), float(stops == 0), min(stops, 3) / 3.0,
                     penalty, slope * 10.0, float(neutral), float(neutral) * a], float)


def rows_from_records(recs: list[LapRecord], laps_total: int,
                      refit_every: int = 1) -> list[tuple]:
    """(lap, features, age, label) per eligible lap, using only earlier laps.

    The final lap is skipped since nobody stops at the flag.
    """
    rows, est, stops = [], None, 0
    for i in range(1, len(recs)):
        prev, cur = recs[i - 1], recs[i]
        stops += int(prev.pitted)
        if cur.lap >= laps_total:
            continue
        hist = recs[:i]
        age = 1 if prev.pitted else prev.tyre_age + 1
        penalty = slope = 0.0
        if i % refit_every == 0 or est is None:
            est = None
            if len(clean_laps(hist, laps_total)) >= MIN_CLEAN_LAPS:
                est = DegradationEstimator().fit(hist, laps_total)
        if est is not None:
            penalty = est.penalty_at(age, cur.compound)
            slope = est.slope_at(age, cur.compound)
        rows.append((cur.lap, hazard_features(age, cur.compound, cur.lap, laps_total,
                                              stops, penalty, slope, prev.safety_car),
                     age, 1.0 if cur.pitted else 0.0))
    return rows


def rows_for_replay(replay: Replay) -> list[tuple]:
    laps = set(replay.available_laps())
    return rows_from_records([r for r in replay._opp_list if r.lap in laps],
                             replay.laps_total)


@dataclass
class PitForecast:
    hazards: list[float]        # conditional h for laps t+1 .. t+k
    p_within: dict[int, float]  # k -> cumulative P(pit within k)
    pit_dist: list[float]       # first-pit distribution over t+1..t+k, then tail


class PitHazardModel:
    def __init__(self, l2: float = 1.0, iters: int = 30, columns: list[int] | None = None):
        self.l2, self.iters, self.columns = l2, iters, columns
        self.beta: np.ndarray | None = None
        self.mu_: np.ndarray | None = None
        self.sd_: np.ndarray | None = None

    # -- training -----------------------------------------------------------------
    def _select(self, X: np.ndarray) -> np.ndarray:
        return X if self.columns is None else X[..., self.columns]

    def fit_rows(self, X: np.ndarray, y: np.ndarray) -> "PitHazardModel":
        """Newton steps on the unweighted loss, so outputs stay real probabilities."""
        X = self._select(X)
        self.mu_ = X[:, 1:].mean(0)
        self.sd_ = X[:, 1:].std(0) + 1e-9
        Z = np.column_stack([X[:, 0], (X[:, 1:] - self.mu_) / self.sd_])
        b = np.zeros(Z.shape[1])
        penalty = np.eye(Z.shape[1]) * self.l2
        penalty[0, 0] = 0.0
        for _ in range(self.iters):
            p = sigmoid(Z @ b)
            w = np.clip(p * (1 - p), 1e-6, None)
            grad = Z.T @ (p - y) + penalty @ b
            step = np.linalg.solve((Z.T * w) @ Z + penalty + 1e-8 * np.eye(len(b)), grad)
            b -= step
            if np.max(np.abs(step)) < 1e-6:
                break
        self.beta = b
        return self

    def fit(self, replays: list[Replay]) -> "PitHazardModel":
        rows = [r for rp in replays for r in rows_for_replay(rp)]
        return self.fit_rows(np.array([r[1] for r in rows]), np.array([r[3] for r in rows]))

    # -- persistence --------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"beta": self.beta.tolist(), "mu": self.mu_.tolist(),
                "sd": self.sd_.tolist(), "columns": self.columns}

    @classmethod
    def from_dict(cls, d: dict) -> "PitHazardModel":
        m = cls(columns=d.get("columns"))
        m.beta = np.array(d["beta"])
        m.mu_ = np.array(d["mu"])
        m.sd_ = np.array(d["sd"])
        return m

    # -- prediction ---------------------------------------------------------------
    def hazards(self, X: np.ndarray) -> np.ndarray:
        X = self._select(X)
        Z = np.column_stack([X[:, 0], (X[:, 1:] - self.mu_) / self.sd_])
        return sigmoid(Z @ self.beta)

    def hazard(self, f: np.ndarray) -> float:
        return float(self.hazards(f[None, :])[0])

    def forecast(self, snap: RaceSnapshot, est: DegradationEstimator, k: int = 5) -> PitForecast:
        """Hazards for the next k laps. Nothing is forecast past the last pit-able lap."""
        hazards = []
        for j in range(1, k + 1):
            lap = snap.lap + j
            if lap >= snap.laps_total:
                hazards.append(0.0)
                continue
            age = snap.opp.next_age + j - 1
            compound = snap.opp.compound
            f = hazard_features(age, compound, lap, snap.laps_total, snap.opp.stops_done,
                                est.penalty_at(age, compound), est.slope_at(age, compound),
                                snap.safety_car and j == 1)
            hazards.append(self.hazard(f))
        p_within, dist, surv = {}, [], 1.0
        for i, h in enumerate(hazards, 1):
            dist.append(surv * h)
            surv *= (1 - h)
            p_within[i] = 1 - surv
        dist.append(surv)
        return PitForecast(hazards, p_within, dist)


def load_models(path: Path = WEIGHTS) -> tuple[PitHazardModel, PitHazardModel]:
    """(hazard model, tyre-age-only baseline), trained on real races."""
    d = json.loads(Path(path).read_text())
    return PitHazardModel.from_dict(d["model"]), PitHazardModel.from_dict(d["baseline"])


def brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y_true) ** 2))


def log_loss(y_true: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))
