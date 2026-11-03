"""Closed-loop strategy evaluation in controlled synthetic races (spec §12).

    python -m pitwall.evaluate --races 150

Each policy drives our car lap by lap through the same randomised races with the
same noise, so comparisons are paired. Policies see only past laps. Results are
simulated counterfactuals, not observed history, and the races share the hold-up
assumption with the simulator, so they test decision quality within that world.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
from pathlib import Path

import numpy as np

from .data import COMPOUND_PACE, generate_race, true_lap_time, true_wear
from .demo import decide_on_replay
from .pit_model import load_models
from .state import PIT_IN_SHARE, LapRecord, Replay, must_stop

OUT = Path(__file__).resolve().parent / "assets" / "strategy_eval.json"
FOLLOW, DELTA = 0.6, 1.0


def scenario(seed: int) -> dict:
    """A random race that models and policies are not told about."""
    rng = np.random.default_rng(10_000 + seed)
    total = int(rng.choice([44, 52, 57, 66]))
    pick = rng.random()
    if pick < 0.5:
        stops = (int(total * rng.uniform(.30, .60)),)
    else:
        a = int(total * rng.uniform(.22, .38))
        stops = (a, a + int(total * rng.uniform(.28, .40)))
    return dict(seed=seed, total=total, opp_stops=stops,
                pit_loss=float(rng.uniform(19, 25)),
                base_own=93.0, base_opp=93.0 + float(rng.normal(0, .25)),
                start_gap=float(rng.uniform(-4, 4)),
                own_seq=tuple(rng.permutation(["MEDIUM", "HARD"])) + ("HARD",),
                opp_seq=tuple(rng.permutation(["MEDIUM", "HARD"])) + ("HARD",))


def run_race(sc: dict, policy, pit=None) -> dict:
    total = sc["total"]
    template = generate_race(sc["seed"], laps_total=total, pit_loss=sc["pit_loss"],
                             opp_pit_laps=sc["opp_stops"], own_pit_laps=(),
                             base_own=sc["base_own"], base_opp=sc["base_opp"],
                             compounds=sc["opp_seq"])
    opp_all = [copy.copy(r) for r in template._opp_list]
    rng = np.random.default_rng(50_000 + sc["seed"])
    noise = rng.normal(0, .25, total + 1)
    wear_noise = rng.normal(0, .05, total + 1)
    own, opp, gap = [], [], sc["start_gap"]
    age, idx, out_lap, stops = 0, 0, False, []
    box_now = False
    for lap in range(1, total + 1):
        age += 1
        comp = sc["own_seq"][min(idx, 2)]
        box = box_now
        lt = true_lap_time(sc["base_own"], comp, age, lap, total) + noise[lap]
        lt += (PIT_IN_SHARE * sc["pit_loss"] if box else 0.0)
        lt += ((1 - PIT_IN_SHARE) * sc["pit_loss"] if out_lap else 0.0)
        p = opp_all[lap - 1]
        diff = lt - p.lap_time
        new = gap + diff
        on_track = not (box or out_lap or p.pitted or p.pit_out)
        if on_track and gap > 0 and new < FOLLOW and -diff < DELTA:  # we are held up
            lt += max(new, FOLLOW) - new
            new = max(new, FOLLOW)
        elif on_track and gap < 0 and new > -FOLLOW and diff < DELTA:  # they are held up
            p.lap_time += new - min(new, -FOLLOW)
            new = min(new, -FOLLOW)
        gap = new
        p.gap_to_rival = -gap
        own.append(LapRecord(lap, "OWN", comp, age, float(lt), gap_to_rival=gap, pitted=box,
                             pit_out=out_lap,
                             private_wear=true_wear(comp, age) + float(wear_noise[lap])))
        opp.append(p)
        out_lap = box
        if box:
            stops.append(lap)
            age, idx = 0, idx + 1
        if lap == total:
            break
        replay = Replay(total, list(own), list(opp), 22.0)
        # A car on its in-lap is already stopping, whatever the rule says.
        box_now = not box and bool(policy(replay, lap, sc, own, opp, pit))
        # the sporting rule is enforced for every policy
        if must_stop(own) and lap + 1 >= total - 1:
            box_now = True
    return {"gap": -gap, "ahead": gap < 0, "stops": stops}


def pitwall_policy(replay, lap, sc, own, opp, pit):
    res = decide_on_replay(replay, lap, 0.0, pit, verbose=False, n_sims=200)
    return "recommendation" in res and res["recommendation"].best_offset == 0


def ignore_opponent_policy(replay, lap, sc, own, opp, pit):
    res = decide_on_replay(replay, lap, 0.0, pit, verbose=False, n_sims=200,
                           pit_dist_override=[1 / 6] * 6)
    return "recommendation" in res and res["recommendation"].best_offset == 0


def reactive_policy(replay, lap, sc, own, opp, pit):
    return (opp[-1].pitted and own[-1].tyre_age >= 3) or own[-1].tyre_age >= int(.5 * sc["total"])


def age_rule_policy(replay, lap, sc, own, opp, pit):
    return own[-1].tyre_age >= int(.42 * sc["total"])


def fixed_policy(replay, lap, sc, own, opp, pit):
    return lap + 1 == int(.45 * sc["total"])


POLICIES = {"PITWALL": pitwall_policy, "PITWALL ignoring opponent": ignore_opponent_policy,
            "reactive (cover their stop)": reactive_policy,
            "tyre-age rule": age_rule_policy, "fixed one-stop": fixed_policy}


def _ci(diff: np.ndarray, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    means = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--races", type=int, default=150)
    a = ap.parse_args(argv)
    pit, _ = load_models()
    scenarios = [scenario(s) for s in range(a.races)]
    results = {name: [run_race(sc, fn, pit) for sc in scenarios] for name, fn in POLICIES.items()}
    base = np.array([r["gap"] for r in results["PITWALL"]])
    base_win = np.array([r["ahead"] for r in results["PITWALL"]], float)
    table = {}
    for name, rows in results.items():
        g = np.array([r["gap"] for r in rows])
        w = np.array([r["ahead"] for r in rows], float)
        table[name] = {"p_ahead": float(w.mean()), "mean_gap": float(g.mean()),
                       "mean_stops": float(np.mean([len(r["stops"]) for r in rows])),
                       "pitwall_gain_s": float((base - g).mean()),
                       "pitwall_gain_s_ci95": _ci(base - g),
                       "pitwall_gain_wins": float((base_win - w).mean()),
                       "pitwall_gain_wins_ci95": _ci(base_win - w)}
    out = {"races": a.races, "table": table}
    OUT.write_text(json.dumps(out, indent=1))
    for name, row in table.items():
        print(f"{name:30s} P(ahead)={row['p_ahead']:.0%} gap={row['mean_gap']:+6.1f}s "
              f"stops={row['mean_stops']:.2f} | PITWALL gain {row['pitwall_gain_s']:+.1f}s "
              f"CI95 {np.round(row['pitwall_gain_s_ci95'], 1)}")
    return out


if __name__ == "__main__":
    main()
