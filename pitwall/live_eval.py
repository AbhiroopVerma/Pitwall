"""Scores the live terminal against what really happened.

Each race is played through the real engine, one line crossing at a time, exactly as
on screen. Then we look at the future it never saw:

    pace      next-lap pace forecast vs the lap that followed
    tyre      lap time forecast 10 laps ahead in the same stint
    win       P(finish ahead) for the option the team really took, vs the real result
    gap       forecast gap at the flag vs the real one, same option
    rejoin    predicted position after a stop vs where the car really was
    rival     rival stop warnings vs real stops
    stability how often the call changes
    latency   calculation time per decision

Settings are fitted on 2023-24 only. 2025 and 2026 stay unseen.

    python -m pitwall.live_eval               # score, writes assets/live_eval.json
    python -m pitwall.live_eval --calibrate   # refit shrink, spread and prior strength
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np

from .data import bundled_race, list_bundled_races
from .live_engine import LiveEngine
from .live_feed import LapEvent, bundled_feed
from .pit_model import load_models
from .priors import PRIORS, RacePriors, load_priors, save_priors

RESULTS = Path(__file__).resolve().parent / "assets" / "live_eval.json"
TRAIN = (2023, 2024)
TEST = (2025, 2026)
HORIZONS = ((1, 10), (11, 25), (26, 999))


def dry_races(years: tuple[int, ...]) -> list[tuple[int, str]]:
    out = []
    for year, circuit in list_bundled_races():
        if year not in years:
            continue
        race = bundled_race(year, circuit)
        if race["compound"].isin(["INTERMEDIATE", "WET"]).mean() < 0.02:
            out.append((year, circuit))
    return out


def battle_pairs(events: list[LapEvent], n_pairs: int, seed: int) -> list[tuple[str, str]]:
    """Neighbours on the road at lap 5, as a team would really pick its rival.

    Picking neighbours at the finish would use the result to choose the test. Which car
    is 'us' alternates, so neither side is favoured.
    """
    at_lap = sorted((e.t, e.driver) for e in events if e.lap == 5)
    order = [d for _, d in at_lap]
    pairs = []
    for i in range(0, min(2 * n_pairs, len(order) - 1), 2):
        a, b = order[i], order[i + 1]
        pairs.append((a, b) if (seed + i // 2) % 2 else (b, a))
    return pairs


def _truth(events: list[LapEvent]) -> dict:
    by: dict[str, dict[int, LapEvent]] = {}
    for e in events:
        by.setdefault(e.driver, {})[e.lap] = e
    return by


def _final_gap(by: dict, own: str, opp: str) -> float | None:
    a, b = by.get(own, {}), by.get(opp, {})
    common = set(a) & set(b)
    if not common:
        return None
    lap = max(common)
    return b[lap].t - a[lap].t  # + = own ahead


def _position_at(by: dict, driver: str, lap: int) -> int | None:
    me = by[driver].get(lap)
    if me is None:
        return None
    return 1 + sum(1 for d, laps in by.items() if d != driver and lap in laps and laps[lap].t < me.t)


def play_pair(year: int, circuit: str, own: str, opp: str, pit_model, priors: RacePriors,
              n_sims: int, every: int = 1) -> dict:
    """Run one pairing through the engine and collect what it said and what happened."""
    feed = bundled_feed(year, circuit)
    laps_only = [e for e in feed.events if isinstance(e, LapEvent)]
    truth = _truth(laps_only)
    total = feed.info.laps_total
    engine = LiveEngine(feed.info, own, opp, n_sims=n_sims, pit_model=pit_model, priors=priors)
    final = _final_gap(truth, own, opp)
    finished_both = (max(truth[own]) >= total - 1 and max(truth[opp]) >= total - 1)
    in_laps = sorted(l for l, e in truth[own].items() if e.pit_in)
    out = {"win": [], "rejoin": [], "pace": [], "calc_ms": [], "calls": [], "stint10": [],
           "warned": [], "plan_error": []}
    seen = -1
    # Skipping laps saves time in calibration, but the engine must still see every event.
    for e in feed.events:
        engine.ingest([e])
        relevant = getattr(e, "driver", None) in (own, opp) or not hasattr(e, "driver")
        if not relevant:
            continue
        if not (every == 1 or (getattr(e, "lap", 0) % every == 0)):
            continue
        frame = engine.tick(e.t)
        if frame.decision_id == seen or not frame.options:
            continue
        seen = frame.decision_id
        lap = frame.lap
        out["calc_ms"].append(frame.calc_ms)
        out["calls"].append((lap, frame.call.kind, frame.call.box_lap))
        if engine.opp != opp or frame.projected:
            continue
        taken = next((l for l in in_laps if l > lap), None)
        option = None
        for o in frame.options:
            if taken is not None and o.box_lap == taken and o.offset is not None:
                option = o
        if option is None:
            option = next((o for o in frame.options if o.offset is None), None)
            if option is None or (taken is not None and taken <= lap + 4):
                option = None
        if option is not None and final is not None and finished_both:
            out["win"].append((total - lap, option.p_ahead, option.exp_gap, final,
                               -frame.gap, bool(frame.flag)))
        if taken == lap + 1 and not frame.flag:
            match = next((o for o in frame.options if o.box_lap == taken and o.offset == 0), None)
            real = _position_at(truth, own, taken + 1)
            if match is not None and match.rejoin is not None and real is not None:
                out["rejoin"].append((match.rejoin.position, real))
        if taken is not None and taken - lap == 5:
            first_plan = frame.call.box_lap
            if first_plan is not None:
                out["plan_error"].append(first_plan - taken)
        own_next = truth[own].get(lap + 1)
        model = (engine._out or {}).get("own_model")
        if model is not None and own_next is not None and _clean(own_next):
            state = engine.field.cars[own][lap]
            if not state.pit_in:
                pred = model.pace(state.age + 1, lap + 1, state.compound)
                out["pace"].append(abs(own_next.lap_time - pred))
                later = truth[own].get(lap + 10)
                if later is not None and _clean(later) and not any(
                        truth[own][n].pit_in for n in range(lap, lap + 10) if n in truth[own]):
                    pred10 = model.pace(state.age + 10, lap + 10, state.compound)
                    out["stint10"].append(later.lap_time - pred10)
    out["rival_stops"], out["rival_warned"] = engine._rival_stops, engine._rival_warned
    return out


def _clean(e: LapEvent) -> bool:
    return not (e.pit_in or e.pit_out or e.neutral) and np.isfinite(e.lap_time) and e.lap > 1


def collect(races, priors, n_pairs, n_sims, pit_model, every=1, log=False) -> dict:
    merged: dict = {"win": [], "rejoin": [], "pace": [], "calc_ms": [], "calls_per_race": [],
                    "stint10": [], "plan_error": [], "rival_stops": 0, "rival_warned": 0}
    for i, (year, circuit) in enumerate(races):
        feed = bundled_feed(year, circuit)
        laps = [e for e in feed.events if isinstance(e, LapEvent)]
        for own, opp in battle_pairs(laps, n_pairs, i):
            try:
                got = play_pair(year, circuit, own, opp, pit_model, priors, n_sims, every)
            except Exception as e:
                print(f"  skip {year} {circuit} {own}/{opp}: {type(e).__name__}: {e}", flush=True)
                continue
            for key in ("win", "rejoin", "pace", "calc_ms", "stint10", "plan_error"):
                merged[key].extend(got[key])
            merged["rival_stops"] += got["rival_stops"]
            merged["rival_warned"] += got["rival_warned"]
            merged["calls_per_race"].append(_changes(got["calls"]))
        if log:
            print(f"  {year} {circuit} done", flush=True)
    return merged


def _changes(calls: list[tuple]) -> int:
    """Times the call really changed. A planned stop drifting past the window is not one."""
    changes, prev = 0, None
    for lap, kind, box in calls:
        now = (kind, box) if kind in ("BOX", "STANDBY") else (kind, None)
        same_target = prev is not None and prev[0] in ("BOX", "STANDBY") and \
            kind in ("BOX", "STANDBY") and prev[1] == box
        if prev is not None and now != prev and not same_target and kind != "PIT" and prev[0] != "PIT":
            changes += 1
        prev = now
    return changes


def _gap_only(gap_now: np.ndarray, left: np.ndarray, spread: float) -> np.ndarray:
    """Win chance from the gap alone: gap over spread * sqrt(laps left). No tyres, no stops."""
    z = gap_now / (spread * np.sqrt(np.maximum(left, 1.0)))
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def fit_gap_only(rows: list) -> float:
    """The spread that scores best on the rows given (the training years)."""
    win = np.array(rows, float)
    y = (win[:, 3] > 0).astype(float)
    grid = np.arange(0.5, 12.01, 0.25)
    briers = [np.mean((_gap_only(win[:, 4], win[:, 0], s) - y) ** 2) for s in grid]
    return float(grid[int(np.argmin(briers))])


def fit_win_blend(rows: list, spread: float) -> tuple[float, float]:
    """Logistic fit of the real result on the simulated logit and the gap signal."""
    win = np.array(rows, float)
    p = np.clip(win[:, 1], 0.005, 0.995)
    x = np.column_stack([np.log(p / (1 - p)),
                         np.clip(win[:, 4] / (spread * np.sqrt(np.maximum(win[:, 0], 1.0))), -6, 6)])
    y = (win[:, 3] > 0).astype(float)
    beta = np.array([1.0, 0.0])
    for _ in range(25):
        q = 1.0 / (1.0 + np.exp(-x @ beta))
        w = np.clip(q * (1 - q), 1e-6, None)
        beta -= np.linalg.solve((x.T * w) @ x + 1e-6 * np.eye(2), x.T @ (q - y))
    return round(float(beta[0]), 3), round(float(beta[1]), 3)


def summarise(m: dict, gap_spread: float | None = None, priors: RacePriors | None = None) -> dict:
    out: dict = {}
    win = np.array(m["win"], float) if m["win"] else np.zeros((0, 6))
    if len(win):
        left, p, gap, final, gap_now = win[:, 0], win[:, 1], win[:, 2], win[:, 3], win[:, 4]
        y = (final > 0).astype(float)
        if priors is not None:  # what the screen shows, rather than the raw simulation
            out["win_brier_simulated"] = round(float(np.mean((p - y) ** 2)), 4)
            p = np.array([priors.win_chance(v, g, int(n)) for v, g, n in zip(p, gap_now, left)])
        hold = (gap_now > 0).astype(float)
        out["decisions"] = int(len(win))
        out["win_brier"] = round(float(np.mean((p - y) ** 2)), 4)
        out["win_brier_current_order"] = round(float(np.mean((hold - y) ** 2)), 4)
        out["win_brier_coinflip"] = 0.25
        if gap_spread is not None:
            only = _gap_only(gap_now, left, gap_spread)
            out["win_brier_gap_only"] = round(float(np.mean((only - y) ** 2)), 4)
        out["gap_mae"] = round(float(np.mean(np.abs(gap - final))), 2)
        out["gap_mae_hold"] = round(float(np.mean(np.abs(gap_now - final))), 2)
        out["by_horizon"] = {}
        for a, b in HORIZONS:
            mask = (left >= a) & (left <= b)
            if mask.sum():
                out["by_horizon"][f"{a}-{b}"] = {
                    "n": int(mask.sum()),
                    "win_brier": round(float(np.mean((p[mask] - y[mask]) ** 2)), 4),
                    "current_order_brier": round(float(np.mean((hold[mask] - y[mask]) ** 2)), 4),
                    **({"gap_only_brier": round(float(np.mean(
                        (_gap_only(gap_now[mask], left[mask], gap_spread) - y[mask]) ** 2)), 4)}
                       if gap_spread is not None else {}),
                    "gap_mae": round(float(np.mean(np.abs(gap[mask] - final[mask]))), 2),
                    "hold_mae": round(float(np.mean(np.abs(gap_now[mask] - final[mask]))), 2)}
        bins = []
        for lo, hi in ((0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.0001)):
            mask = (p >= lo) & (p < hi)
            if mask.sum():
                bins.append({"said": f"{lo:.0%}-{min(hi, 1):.0%}", "n": int(mask.sum()),
                             "mean_said": round(float(p[mask].mean()), 3),
                             "happened": round(float(y[mask].mean()), 3)})
        out["reliability"] = bins
    if m["rejoin"]:
        r = np.array(m["rejoin"], float)
        err = np.abs(r[:, 0] - r[:, 1])
        out["rejoin"] = {"n": int(len(r)), "mean_abs_places": round(float(err.mean()), 2),
                         "exact": round(float(np.mean(err == 0)), 3),
                         "within_1": round(float(np.mean(err <= 1)), 3),
                         "within_2": round(float(np.mean(err <= 2)), 3)}
    if m["pace"]:
        out["next_lap_pace_mae"] = round(float(np.mean(m["pace"])), 3)
    if m["stint10"]:
        s = np.array(m["stint10"], float)
        out["pace_10_laps_ahead"] = {"n": int(len(s)), "mae": round(float(np.mean(np.abs(s))), 3),
                                     "bias": round(float(np.mean(s)), 3)}
    if m["plan_error"]:
        e = np.abs(np.array(m["plan_error"], float))
        out["plan_vs_team_5_laps_before"] = {"n": int(len(e)), "median_abs_laps": float(np.median(e)),
                                             "within_3": round(float(np.mean(e <= 3)), 3)}
    if m["calc_ms"]:
        c = np.array(m["calc_ms"], float)
        out["calc_ms"] = {"median": round(float(np.median(c)), 1),
                          "p99": round(float(np.percentile(c, 99)), 1),
                          "max": round(float(c.max()), 1)}
    if m["calls_per_race"]:
        out["call_changes_per_race"] = round(float(np.mean(m["calls_per_race"])), 2)
    if m["rival_stops"]:
        out["rival_stops_flagged"] = {"stops": m["rival_stops"], "flagged": m["rival_warned"],
                                      "share": round(m["rival_warned"] / m["rival_stops"], 3)}
    return out


def calibrate(n_pairs: int, n_sims: int) -> RacePriors:
    """Coordinate search on the training years, on the real outcomes."""
    races = dry_races(TRAIN)[::2]
    model = load_models()[0]
    base = load_priors()

    def score(**changes) -> tuple[float, dict]:
        trial = dataclasses.replace(base, **changes)
        got = summarise(collect(races, trial, n_pairs, n_sims, model, every=4))
        # The blend sits on top of the simulation, so tune the simulation on its own.
        return got["win_brier"], got

    best = {"prior_strength": base.prior_strength, "pace_tau": base.pace_tau, "pace_sd": base.pace_sd}
    grids = {"prior_strength": (1.0, 3.0, 10.0, 30.0),
             "pace_tau": (0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.2),
             "pace_sd": (0.0, 0.05, 0.1, 0.2)}
    for name, grid in grids.items():
        results = []
        for value in grid:
            s, detail = score(**{**best, name: value})
            results.append((s, value))
            print(f"{name}={value}: brier {s}, gap mae {detail.get('gap_mae')}, "
                  f"tyre+10 {detail.get('pace_10_laps_ahead')}", flush=True)
        best[name] = min(results)[1]
        print(f"-> {name} = {best[name]}", flush=True)
    return dataclasses.replace(base, **best)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--sims", type=int, default=300)
    a = ap.parse_args(argv)
    warnings.simplefilter("ignore")
    if a.calibrate:
        tuned = calibrate(min(a.pairs, 3), a.sims)
        save_priors(tuned, PRIORS)
        print(f"saved: tau {tuned.pace_tau}, sd {tuned.pace_sd}, "
              f"prior strength {tuned.prior_strength}")
        return
    model = load_models()[0]
    priors = load_priors()
    train = collect(dry_races(TRAIN), priors, a.pairs, a.sims, model)
    spread = fit_gap_only(train["win"])
    priors.win_a, priors.win_b = fit_win_blend(train["win"], spread)
    priors.win_spread = spread
    save_priors(priors, PRIORS)
    print(f"win blend: a={priors.win_a} b={priors.win_b} spread={spread}", flush=True)
    results = {"settings": {"pairs_per_race": a.pairs, "sims": a.sims,
                            "pace_tau": priors.pace_tau, "pace_sd": priors.pace_sd,
                            "prior_strength": priors.prior_strength,
                            "gap_only_spread": spread, "win_a": priors.win_a,
                            "win_b": priors.win_b},
               "train_2023_24": summarise(train, spread, priors)}
    print(f"training years: {results['train_2023_24'].get('win_brier')}, gap-only spread {spread}", flush=True)
    for year in TEST:
        races = dry_races((year,))
        started = time.time()
        results[str(year)] = {"races": len(races),
                              **summarise(collect(races, priors, a.pairs, a.sims, model), spread, priors)}
        print(f"{year}: {time.time() - started:.0f}s", flush=True)
        print(json.dumps(results[str(year)], indent=1), flush=True)
    RESULTS.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
