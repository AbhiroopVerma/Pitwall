"""Facts learned from past races and used as starting points in a live race.

Everything here is measured on the 2023-24 training races, so the 2025 and 2026
races stay unseen when the terminal is scored on them. Rebuild with::

    python -m pitwall.priors
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .data import DEFAULT_PIT_LOSS, bundled_laps
from .degradation import COMPOUND_WEAR, FUEL_S_PER_LAP, OUTLIER_S, _fit_curve

PRIORS = Path(__file__).resolve().parent / "assets" / "race_priors.json"
NEUTRAL = Path(__file__).resolve().parent / "assets" / "openf1_neutral.json"
TRAIN_YEARS = (2023, 2024)
MIN_STOPS = 5
MIN_STINTS = 5


def neutral_periods_of(year: int, circuit: str) -> list[tuple[float, float, str]]:
    if not NEUTRAL.exists():
        return []
    rows = json.loads(NEUTRAL.read_text()).get(f"{year}|{circuit}", [])
    return [(float(a), float(b), str(k)) for a, b, k in rows]


def kind_at(periods: list[tuple[float, float, str]], t: float) -> str | None:
    for a, b, kind in periods:
        if a <= t < b:
            return kind
    return None


@dataclass
class RacePriors:
    pit_loss_by_circuit: dict = field(default_factory=dict)
    pit_loss_default: float = DEFAULT_PIT_LOSS
    # Stop cost as a share of the green-flag cost, and the chance the period is still
    # running on the next lap, per kind of neutral period.
    neutral_cost_ratio: dict = field(default_factory=dict)
    neutral_continue: dict = field(default_factory=dict)
    # 90th percentile stint length, by circuit and over all circuits.
    stint_p90: dict = field(default_factory=dict)
    stint_p90_all: dict = field(default_factory=dict)
    laps_by_circuit: dict = field(default_factory=dict)
    # Fitted by pitwall.live_eval on the training races.
    pace_shrink: float = 1.0
    pace_tau: float | None = None   # spread of true pace gaps, replaces a fixed shrink
    pace_sd: float = 0.0
    # Wear curve (b1, b2) by circuit and overall, and how hard it pulls on a live fit.
    wear_by_circuit: dict = field(default_factory=dict)
    wear_default: tuple | list | None = None
    prior_strength: float = 0.1
    # Displayed win chance: logit = a * logit(simulated) + b * gap / (spread * sqrt(laps left)).
    # Fitted on the training years by pitwall.live_eval. It only rescales: all options of
    # a decision share the gap term and a > 0, so it cannot change which one ranks first.
    win_a: float = 1.0
    win_b: float = 0.0
    win_spread: float = 1.0

    def win_chance(self, simulated: float, gap_ahead: float, laps_left: int) -> float:
        p = min(max(simulated, 0.005), 0.995)
        z = gap_ahead / (self.win_spread * math.sqrt(max(laps_left, 1)))
        logit = self.win_a * math.log(p / (1.0 - p)) + self.win_b * max(-6.0, min(z, 6.0))
        return 1.0 / (1.0 + math.exp(-logit))

    def wear_prior(self, circuit: str | None) -> tuple[float, float] | None:
        value = self.wear_by_circuit.get(circuit or "", self.wear_default)
        return tuple(value) if value is not None else None

    def pit_loss(self, circuit: str | None) -> float:
        return float(self.pit_loss_by_circuit.get(circuit or "", self.pit_loss_default))

    def stop_costs(self, pit_loss: float, kind: str | None, n: int = 4) -> list[float] | None:
        """Expected cost of boxing at the end of each of the next ``n`` laps.

        The period may have ended by then, so the saving decays with the measured
        chance that it carries on from one lap to the next.
        """
        if kind is None or kind not in self.neutral_cost_ratio:
            return None
        saving = 1.0 - float(self.neutral_cost_ratio[kind])
        keep = float(self.neutral_continue.get(kind, 0.5))
        return [pit_loss * (1.0 - saving * keep ** (h + 1)) for h in range(n)]

    def stint_limits(self, circuit: str | None) -> dict:
        return {c: self.stint_limit(circuit, c) for c in ("SOFT", "MEDIUM", "HARD")}

    def stint_limit(self, circuit: str | None, compound: str) -> int | None:
        by_circuit = self.stint_p90.get(circuit or "", {})
        value = by_circuit.get(compound, self.stint_p90_all.get(compound))
        return int(value) if value is not None else None


def load_priors(path: Path = PRIORS) -> RacePriors:
    if not Path(path).exists():
        return RacePriors()
    return RacePriors(**json.loads(Path(path).read_text()))


def _is_dry(race) -> bool:
    return not race["compound"].isin(["INTERMEDIATE", "WET"]).any()


def _stop_costs(race) -> list[tuple[int, float, bool]]:
    """(lap, stop cost against the cars not stopping, field neutral) for each stop."""
    times = race.pivot(index="lap", columns="driver", values="time")
    pit_in = race.pivot(index="lap", columns="driver", values="pit_in").astype("boolean").fillna(False)
    pit_out = race.pivot(index="lap", columns="driver", values="pit_out").astype("boolean").fillna(False)
    out = []
    for lap in times.index[1:-1]:
        if lap + 1 not in times.index:
            continue
        ref = 0.0
        for l in (lap, lap + 1):
            running = times.loc[l][~(pit_in.loc[l] | pit_out.loc[l])].dropna()
            ref += float(running.median()) if len(running) else float("nan")
        if not np.isfinite(ref):
            continue
        for d in times.columns:
            if pit_in.loc[lap, d]:
                cost = times.loc[lap, d] + times.loc[lap + 1, d] - ref
                if np.isfinite(cost) and -5.0 < cost < 60.0:
                    out.append((int(lap), float(cost), d))
    return out


def _line_times(race) -> dict[tuple, float]:
    """(driver number, lap) -> when the car finished that lap, from the next lap's start."""
    out = {}
    for d, car in race.sort_values("lap").groupby("driver"):
        starts = dict(zip(car["lap"], car["start"]))
        for lap, start in starts.items():
            nxt = starts.get(lap + 1)
            if nxt is not None and np.isfinite(nxt):
                out[(d, int(lap))] = float(nxt)
    return out


def _wear_rows(race) -> tuple[np.ndarray, np.ndarray]:
    """Curve columns and stint-demeaned pace for all clean racing laps of one race."""
    total = int(race["total_laps"].iloc[0])
    curves, ys = [], []
    for _, car in race.sort_values("lap").groupby("driver"):
        stint_no = car["pit_in"].astype(int).shift(1, fill_value=0).cumsum()
        ok = (car["lap"] > 1) & ~car["pit_in"].astype(bool) & ~car["pit_out"].astype(bool) \
            & ~car["neutral"].astype(bool) & car["time"].notna() & car["age"].notna() \
            & car["compound"].isin(list(COMPOUND_WEAR))
        for _, stint in car[ok].groupby(stint_no[ok]):
            if len(stint) < 8:
                continue
            y = stint["time"].to_numpy(float) - FUEL_S_PER_LAP * (total - stint["lap"].to_numpy(float))
            keep = np.abs(y - np.median(y)) <= OUTLIER_S
            if keep.sum() < 8:
                continue
            factor = stint["compound"].map(COMPOUND_WEAR).to_numpy(float)
            a = stint["age"].to_numpy(float) * factor / 10.0
            x = np.column_stack([a, a ** 2])[keep]
            curves.append(x - x.mean(0))
            ys.append(y[keep] - y[keep].mean())
    if not curves:
        return np.zeros((0, 2)), np.zeros(0)
    return np.vstack(curves), np.concatenate(ys)


def _nonneg_curve(x: np.ndarray, y: np.ndarray) -> list[float]:
    """b1, b2 >= 0 by least squares, with a small ridge since age and age^2 overlap."""
    fit, _, _ = _fit_curve(x, np.zeros((len(y), 0)), y, 1.0)
    return [round(float(v), 4) for v in fit]


def build_priors(years: tuple[int, ...] = TRAIN_YEARS) -> RacePriors:
    df = bundled_laps()
    df = df[df["year"].isin(years)]
    green: dict[str, list[float]] = {}
    neutral: dict[str, list[float]] = {}
    spans: dict[str, list[float]] = {}
    wear_x: dict[str, list[np.ndarray]] = {}
    wear_y: dict[str, list[np.ndarray]] = {}
    stints: dict[str, dict[str, list[int]]] = {}
    total_laps: dict[str, int] = {}
    for (year, circuit), race in df.groupby(["year", "circuit"], sort=False):
        total_laps[circuit] = int(race["total_laps"].iloc[0])
        if not _is_dry(race):
            continue
        periods = neutral_periods_of(int(year), circuit)
        typical = float(race.loc[~race["neutral"].astype(bool), "time"].median())
        for start, end, kind in periods:
            spans.setdefault(kind, []).append((end - start) / typical)
        x, y = _wear_rows(race)
        if len(y):
            wear_x.setdefault(circuit, []).append(x)
            wear_y.setdefault(circuit, []).append(y)
        race_green = []
        race_neutral = []
        entered = _line_times(race)
        for lap, cost, driver in _stop_costs(race):
            # Classified when the car actually reached the pits, and when it left.
            at_entry = entered.get((driver, lap))
            kind = kind_at(periods, at_entry) if at_entry is not None else None
            at_exit = kind_at(periods, at_entry + 25.0) if at_entry is not None else None
            if kind is None and at_exit is None:
                race_green.append(cost)
            elif kind in ("SC", "VSC"):
                race_neutral.append((kind, cost))
        green.setdefault(circuit, []).extend(race_green)
        if race_green:
            ref = float(np.median(race_green))
            for kind, cost in race_neutral:
                neutral.setdefault(kind, []).append(cost / ref)
        last_lap = int(race["lap"].max())
        for row in race[race["age"].notna()].itertuples():
            if row.pit_in or row.lap == last_lap:  # the stint ended here
                if row.age >= 3 and row.compound in ("SOFT", "MEDIUM", "HARD"):
                    stints.setdefault(circuit, {}).setdefault(row.compound, []).append(int(row.age))
    priors = RacePriors()
    for circuit, costs in green.items():
        if len(costs) >= MIN_STOPS:
            priors.pit_loss_by_circuit[circuit] = round(float(np.median(costs)), 2)
    every = [c for costs in green.values() for c in costs]
    priors.pit_loss_default = round(float(np.median(every)), 2)
    priors.neutral_cost_ratio = {k: round(float(np.median(v)), 3) for k, v in neutral.items()}
    # From a random moment in a period lasting d laps, the chance it still runs one lap
    # later is E[max(d - 1, 0)] / E[d], since longer periods are more likely to be hit.
    priors.neutral_continue = {k: round(float(np.sum(np.maximum(np.array(v) - 1.0, 0.0))
                                              / np.sum(v)), 3) for k, v in spans.items()}
    all_x = np.vstack([x for xs in wear_x.values() for x in xs])
    all_y = np.concatenate([y for ys in wear_y.values() for y in ys])
    priors.wear_default = _nonneg_curve(all_x, all_y)
    for circuit in wear_x:
        priors.wear_by_circuit[circuit] = _nonneg_curve(np.vstack(wear_x[circuit]),
                                                        np.concatenate(wear_y[circuit]))
    pooled: dict[str, list[int]] = {}
    for circuit, by_compound in stints.items():
        for compound, runs in by_compound.items():
            pooled.setdefault(compound, []).extend(runs)
            if len(runs) >= MIN_STINTS:
                priors.stint_p90.setdefault(circuit, {})[compound] = int(np.percentile(runs, 90))
    priors.stint_p90_all = {c: int(np.percentile(r, 90)) for c, r in pooled.items()}
    priors.laps_by_circuit = total_laps
    return priors


def save_priors(priors: RacePriors, path: Path = PRIORS) -> None:
    Path(path).write_text(json.dumps(priors.__dict__, indent=1, sort_keys=True))


def main() -> None:
    old = load_priors()
    new = build_priors()
    for name in ("pace_shrink", "pace_tau", "pace_sd", "prior_strength", "win_a", "win_b",
                 "win_spread"):
        setattr(new, name, getattr(old, name))
    save_priors(new)
    print(f"pit loss: {len(new.pit_loss_by_circuit)} circuits, default {new.pit_loss_default}s")
    print(f"neutral cost share: {new.neutral_cost_ratio}")
    print(f"neutral carries on: {new.neutral_continue}")
    print(f"stint p90: {new.stint_p90_all}")
    print(f"wear prior overall: {new.wear_default}")
    for circuit, curve in sorted(new.wear_by_circuit.items()):
        at_20 = curve[0] * 2.0 + curve[1] * 4.0
        print(f"  {circuit:20s} {curve}  medium tyre at age 20 loses {at_20:.2f}s")


if __name__ == "__main__":
    main()
