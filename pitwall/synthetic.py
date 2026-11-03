"""A fabricated 20-car race in the same format as the real data.

Everything here is invented and every screen says so. It exists to rehearse what real
races offer only by luck: a safety car on a chosen lap, a rival who undercuts, a rival
who retires. Physics are separate from the models, which only ever see lap times.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .live_feed import ScheduledFeed, feed_from_frame
from .state import PIT_IN_SHARE

SCENARIOS = {
    "undercut": "the rival stops early to get ahead of us",
    "safety-car": "a safety car comes out mid-race, close to the pit window",
    "rival-out": "the rival stops on track and the target must change",
}
EPOCH = 1_800_000_000.0
PIT_LOSS = 21.5
COMPOUND_PACE = {"SOFT": -0.7, "MEDIUM": 0.0, "HARD": 0.5}
COMPOUND_WEAR = {"SOFT": 0.13, "MEDIUM": 0.085, "HARD": 0.055}
SC_LAP_FACTOR = 1.45
FIELD = ["OWN", "RIV"] + [f"C{i:02d}" for i in range(3, 21)]


def synthetic_race(scenario: str = "undercut", seed: int = 11, laps: int = 52) -> tuple[pd.DataFrame, list]:
    """(lap rows, neutral periods). Deterministic for a given seed."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}: choose from {sorted(SCENARIOS)}")
    rng = np.random.default_rng(seed)
    base_lap = 88.0
    sc_laps = set(range(24, 28)) if scenario == "safety-car" else set()
    plans: dict[str, tuple] = {}
    pace: dict[str, float] = {}
    for i, car in enumerate(FIELD):
        first = int(rng.integers(19, 28))
        plans[car] = (("MEDIUM", "HARD"), (first,))
        pace[car] = base_lap + 0.09 * i + float(rng.normal(0, 0.06))
    # The two cars this race is about: close on pace, ours placed just behind.
    pace["RIV"], pace["OWN"] = base_lap + 0.25, base_lap + 0.28
    plans["OWN"] = (("MEDIUM", "HARD"), (26,))
    plans["RIV"] = (("MEDIUM", "HARD"), (18,) if scenario == "undercut" else (25,))
    grid = {"RIV": 3, "OWN": 4}
    others = [c for c in FIELD if c not in grid]
    for car, slot in zip(others, [1, 2] + list(range(5, 30))):
        grid[car] = slot
    retire = {"RIV": 21} if scenario == "rival-out" else {}

    state = {c: EPOCH + 0.45 * grid[c] for c in FIELD}
    age = {c: 1 for c in FIELD}
    stint = {c: 0 for c in FIELD}
    rows = []
    periods = []
    for lap in range(1, laps + 1):
        neutral = lap in sc_laps
        order = sorted((c for c in FIELD if not (c in retire and lap > retire[c])),
                       key=lambda c: state[c])
        lead_time = state[order[0]]
        if neutral and not periods:
            periods.append([lead_time + 20.0, lead_time, "SC"])
        out_cars = {c for c in order
                    if lap > 1 and rows and any(r["acronym"] == c and r["lap"] == lap - 1
                                                and r["pit_in"] for r in rows[-len(FIELD):])}
        for pos, car in enumerate(order, 1):
            start = state[car]
            compounds, stops = plans[car]
            comp = compounds[min(stint[car], len(compounds) - 1)]
            wear = COMPOUND_WEAR[comp] * age[car] + 0.0035 * age[car] ** 2
            fuel = 0.038 * (laps - lap)
            t = pace[car] + COMPOUND_PACE[comp] + wear + fuel + float(rng.normal(0, 0.18))
            is_in = lap in stops
            cost = PIT_LOSS * (0.6 if neutral else 1.0)
            if lap == 1:
                t += 4.0
            if neutral:
                # Under a safety car the field slows and bunches up behind the leader.
                t = pace[order[0]] * SC_LAP_FACTOR
                if pos > 1:
                    t -= 0.35 * (start - lead_time - (pos - 1) * 1.0)
            if is_in:
                t += PIT_IN_SHARE * cost
            if car in out_cars:
                t += (1 - PIT_IN_SHARE) * cost
            rows.append({"year": 2099, "race": f"Fabricated: {scenario}", "circuit": "Synthetic",
                         "driver": FIELD.index(car) + 1, "acronym": car, "lap": lap,
                         "compound": comp, "age": float(age[car]), "time": round(t, 3),
                         "start": round(start, 3), "pit_in": is_in, "pit_out": car in out_cars,
                         "neutral": neutral, "pos": float(pos), "total_laps": laps})
            state[car] = start + t
            age[car] += 1
            if is_in:
                age[car] = 1
                stint[car] += 1
        if neutral:
            periods[-1][1] = float(max(state[c] for c in order)) - 40.0
    return pd.DataFrame(rows), [(a, b, k) for a, b, k in periods]


def synthetic_feed(scenario: str = "undercut", seed: int = 11) -> ScheduledFeed:
    race, periods = synthetic_race(scenario, seed)
    return feed_from_frame(race, periods, f"Fabricated race: {scenario}", "FABRICATED",
                           None, None)
