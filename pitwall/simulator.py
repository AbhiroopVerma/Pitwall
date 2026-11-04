"""Module 6 — 1v1 Monte Carlo counter-strategy simulation, run to the flag.

Each simulated race:
  1. Sample the opponent's next stop from the hazard forecast.
  2. Apply our candidate action. After that first choice, each car follows the
     cheapest remaining plan for its tyres (dynamic programming).
  3. Roll both cars lap by lap with the SAME pace model, noise and stop costs.
  4. A faster car that catches the other is held up unless its pace advantage on
     that lap reaches ``overtake_delta``. Cars can swap places freely in the pits.

Step 4 is what makes the opponent's timing matter: it decides whether we rejoin
ahead (undercut / overcut) or get stuck behind.

All candidate actions share the same random draws, so differences between them
are not Monte Carlo noise. The opponent does not react to our stop, which is a
stated limit of the model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .degradation import DegradationEstimator, fuel_effect
from .state import PIT_IN_SHARE

STAY_OUT = None  # the action "no stop in the action window"
CLIFF_S_PER_LAP = 0.25  # assumption: few stints run past the norm, so it cannot be measured


@dataclass
class CarInputs:
    model: DegradationEstimator
    next_age: int
    compound: str | None
    must_stop: bool
    noise_sd: float    # per-lap pace noise, seconds
    slope_se: float    # standard error of degradation slope, s per lap of age
    pace_adj: float = 0.0  # added to every lap: shrinks a pace gap measured from few laps
    # Stint length each compound normally reaches at this circuit. Drivers manage pace,
    # so lap times alone often show no wear at all until a tyre is finished. Past the
    # norm a tyre costs CLIFF_S_PER_LAP more for each extra lap.
    stint_limits: dict | None = None


@dataclass
class SimInputs:
    lap: int
    laps_total: int
    gap0: float            # effective gap, + = we trail
    pit_loss: float
    own: CarInputs
    opp: CarInputs
    pit_dist: list[float]  # over next k laps, last entry = no stop in k
    fresh_offset_sd: float = 0.4  # next compound is unknown: pace offset of a new stint
    overtake_delta: float = 1.0   # s/lap of pace advantage needed to pass on track
    follow_gap: float = 0.6       # how close a held-up car runs, seconds
    # Stop cost when boxing on each of the next laps, then ``pit_loss``. A safety car
    # makes the first ones cheap. None means every stop costs ``pit_loss``.
    stop_costs: list[float] | None = None
    # Spread of our pace relative to theirs for the rest of the race, s/lap. Lap noise
    # averages out over a long run, whereas a real pace difference does not.
    pace_sd: float = 0.0


@dataclass
class ActionResult:
    action: str
    our_pit_offset: int | None  # None = stay out beyond the window
    p_ahead: float
    exp_gap: float      # + = we finish ahead (s), mean over sims
    risk: float         # P5 worst gap (downside): lower is worse
    gaps: list[float]
    stops: tuple = ()   # our planned stops as race laps


def action_label(offset: int | None) -> str:
    if offset is None:
        return "stay out"
    return "pit now" if offset == 0 else f"pit +{offset}"


def valid_actions(lap: int, laps_total: int, own_next_age: int, own_must_stop: bool,
                  window: int = 3) -> list[int | None]:
    """Candidate actions after removing the impossible ones (spec §9).

    Offset h means box at the end of lap ``lap + 1 + h``.
    - Nobody boxes on the final lap.
    - A car that has just stopped does not stop again immediately.
    - Staying out is only legal if a required stop can still be made later.
    """
    actions: list[int | None] = []
    for h in range(window + 1):
        if lap + 1 + h >= laps_total:
            break
        if own_next_age + h < 3:
            continue
        actions.append(h)
    last_stop_lap = laps_total - 1
    if not own_must_stop or lap + 1 + window + 1 <= last_stop_lap:
        actions.append(STAY_OUT)
    return actions or [STAY_OUT]


class _Planner:
    """Cheapest remaining stops for one car, from its tyre curve."""

    def __init__(self, car: CarInputs, first_lap: int, n_laps: int, pit_loss: float,
                 stop_costs: list[float] | None = None):
        self.car, self.first_lap, self.n, self.pit_loss = car, first_lap, n_laps, pit_loss
        self.stop_costs = stop_costs or []
        self._memo = {}

    def stop_cost(self, h: int) -> float:
        return self.stop_costs[h] if h < len(self.stop_costs) else self.pit_loss

    def _penalty(self, age: int, compound: str | None) -> float:
        penalty = self.car.model.penalty_at(age, compound)
        limits = self.car.stint_limits
        if limits:
            limit = limits.get(compound or "MEDIUM")
            if limit is not None and age > limit:
                penalty += CLIFF_S_PER_LAP * (age - limit)
        return penalty

    def _value(self, h: int, age: int, on_start_compound: bool, must: bool):
        """Best (tyre cost, stops) from lap index h with the given tyre age."""
        if h >= self.n:
            return (1e6 if must else 0.0), ()
        key = (h, age, on_start_compound, must)
        if key in self._memo:
            return self._memo[key]
        comp = self.car.compound if on_start_compound else None
        lap_cost = self._penalty(age, comp)
        stay_cost, stay_plan = self._value(h + 1, age + 1, on_start_compound, must)
        best = (lap_cost + stay_cost, stay_plan)
        if h < self.n - 1:  # no stop on the final lap
            pit_cost, pit_plan = self._value(h + 1, 1, False, False)
            total = lap_cost + self.stop_cost(h) + pit_cost
            if total < best[0]:
                best = (total, (h,) + pit_plan)
        self._memo[key] = best
        return best

    def plan(self, first_stop: int | None, earliest: int) -> tuple:
        """All stops as lap indices. ``first_stop`` is forced when given;
        otherwise no stop happens before lap index ``earliest``."""
        c = self.car
        if first_stop is not None:
            _, rest = self._value(first_stop + 1, 1, False, False)
            return (first_stop,) + rest
        h = min(earliest, self.n)
        return self._value(h, c.next_age + h, True, c.must_stop)[1]

    def lap_times(self, stops: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(deterministic lap times, on-new-tyres flag, lap involves the pits)."""
        c = self.car
        times, fresh, pits = np.zeros(self.n), np.zeros(self.n), np.zeros(self.n, bool)
        ages = np.zeros(self.n)
        age, comp, has_stopped = c.next_age, c.compound, False
        stopset = set(stops)
        for h in range(self.n):
            lap = self.first_lap + h
            times[h] += c.model.baseline_ + c.pace_adj + self._penalty(age, comp)
            times[h] += fuel_effect(lap, c.model.laps_total_)
            fresh[h] = float(has_stopped)
            ages[h] = age
            age += 1
            if h in stopset:
                cost = self.stop_cost(h)
                times[h] += PIT_IN_SHARE * cost
                pits[h] = True
                if h + 1 < self.n:
                    times[h + 1] += (1.0 - PIT_IN_SHARE) * cost
                    pits[h + 1] = True
                age, comp, has_stopped = 1, None, True
        self._ages = ages
        return times, fresh, pits


def _car_matrix(planner: _Planner, stops: tuple, offsets: np.ndarray, noise: np.ndarray,
                slope_err: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    base, fresh, pits = planner.lap_times(stops)
    ages = planner._ages
    times = base + noise + np.outer(offsets, fresh) + np.outer(slope_err, ages)
    return times, np.broadcast_to(pits, times.shape)


def _roll(gap0: float, own: np.ndarray, opp: np.ndarray, own_pit: np.ndarray,
          opp_pit: np.ndarray, follow: float, delta: float) -> np.ndarray:
    g = np.full(own.shape[0], float(gap0))
    for h in range(own.shape[1]):
        diff = own[:, h] - opp[:, h]     # + = we lost time this lap
        new = g + diff
        on_track = ~(own_pit[:, h] | opp_pit[:, h])
        # we were behind and would catch them, without enough pace to pass
        stuck_us = on_track & (g > 0) & (new < follow) & (-diff < delta)
        # they were behind and would catch us, without enough pace to pass
        stuck_them = on_track & (g < 0) & (new > -follow) & (diff < delta)
        new = np.where(stuck_us, np.maximum(new, follow), new)
        new = np.where(stuck_them, np.minimum(new, -follow), new)
        g = new
    return g


def simulate_actions(inp: SimInputs, actions: list[int | None], n_sims: int,
                     rng: np.random.Generator, window: int = 3) -> list[ActionResult]:
    n_laps = inp.laps_total - inp.lap
    first = inp.lap + 1
    own_pl = _Planner(inp.own, first, n_laps, inp.pit_loss, inp.stop_costs)
    opp_pl = _Planner(inp.opp, first, n_laps, inp.pit_loss, inp.stop_costs)
    # Common random numbers: drawn once and reused for every action.
    own_noise = rng.normal(0, inp.own.noise_sd, (n_sims, n_laps))
    opp_noise = rng.normal(0, inp.opp.noise_sd, (n_sims, n_laps))
    own_off = rng.normal(0, inp.fresh_offset_sd, n_sims)
    opp_off = rng.normal(0, inp.fresh_offset_sd, n_sims)
    own_slope = rng.normal(0, inp.own.slope_se, n_sims)
    opp_slope = rng.normal(0, inp.opp.slope_se, n_sims)
    dist = np.asarray(inp.pit_dist, float)
    dist = dist / dist.sum()
    k = len(dist) - 1
    draws = rng.choice(len(dist), size=n_sims, p=dist)
    relative = rng.normal(0, inp.pace_sd, n_sims) if inp.pace_sd > 0 else None

    opp_times = np.zeros((n_sims, n_laps))
    opp_pit = np.zeros((n_sims, n_laps), bool)
    for idx in np.unique(draws):
        mask = draws == idx
        if idx < k and idx < n_laps - 1:
            stops = opp_pl.plan(int(idx), 0)
        else:
            stops = opp_pl.plan(None, k)
        t, p = _car_matrix(opp_pl, stops, opp_off[mask], opp_noise[mask], opp_slope[mask])
        opp_times[mask], opp_pit[mask] = t, p

    results = []
    for a in actions:
        stops = own_pl.plan(a, 0) if a is not None else own_pl.plan(None, window + 1)
        own_times, own_pit = _car_matrix(own_pl, stops, own_off, own_noise, own_slope)
        if relative is not None:
            own_times = own_times + relative[:, None]
        g = _roll(inp.gap0, own_times, opp_times, own_pit, opp_pit,
                  inp.follow_gap, inp.overtake_delta)
        gaps = -g
        results.append(ActionResult(
            action_label(a), a, float(np.mean(gaps > 0)), float(np.mean(gaps)),
            float(np.percentile(gaps, 5)), gaps.tolist(),
            tuple(first + s for s in stops)))
    return results
