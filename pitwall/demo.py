"""End-to-end PITWALL pipeline (Modules 1→7).

Per decision lap, using ONLY laps <= lap (no future leakage):
  Replay → DegradationEstimator → PitHazardModel → ThreatDetector
      → OwnCarModel → Monte Carlo simulator → optimiser → recommendation.

Usage:
    python -m pitwall.demo                          # synthetic race, lap 13
    python -m pitwall.demo --race 2023 Sakhir --own ALO --opp VER --lap 12
    python -m pitwall.demo --backtest               # held-out real-race scores
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .data import generate_race, load_bundled_replay
from .degradation import DegradationEstimator, clean_laps
from .owncar import OwnCarModel
from .pit_model import PitHazardModel, WEIGHTS, load_models
from .optimizer import optimise
from .simulator import CarInputs, SimInputs, action_label, simulate_actions, valid_actions
from .state import Replay, WET_COMPOUNDS
from .threshold import ThreatDetector, Trigger, expected_loss

WINDOW = 3     # candidate actions: pit now, +1, +2, +3, or stay out
N_SIMS = 1000
# A slope error compounds over every remaining lap, and short noisy stints can report
# a very large one. This cap keeps the tails from being set by extrapolation alone.
MAX_SLOPE_SE = 0.03


def stint_laps(history: list, laps_total: int) -> int:
    """Clean laps in the latest stint, the laps its pace baseline rests on."""
    kept = clean_laps(history, laps_total)
    if not kept:
        return 0
    latest = kept[-1][1]
    return sum(1 for _, stint in kept if stint == latest)


def _monitor(lap: int, theta: float, reason: str, p3: float = 0.0, fc=None, est=None,
             own=None) -> dict:
    return {"lap": lap, "p3": p3, "trigger": Trigger(False, "MONITOR", reason, p3, 3, theta),
            "forecast": fc, "est": est, "own_model": own}


def decide_at(lap: int, theta: float = 0.5, seed_demo: int = 7,
              pit: PitHazardModel | None = None, verbose: bool = True) -> dict:
    """Pipeline on the synthetic demo race."""
    demo = generate_race(seed=seed_demo, opp_pit_laps=(14, 28), own_pit_laps=(15, 29))
    return decide_on_replay(demo, lap, theta, pit, verbose)


def decide_on_replay(replay: Replay, lap: int, theta: float = 0.5,
                     pit: PitHazardModel | None = None, verbose: bool = True,
                     overtake_delta: float = 1.0, n_sims: int = N_SIMS,
                     pit_dist_override: list[float] | None = None,
                     always_simulate: bool = False, pit_loss: float | None = None,
                     stop_costs: list[float] | None = None, neutral: bool | None = None,
                     pace_shrink: float = 1.0, pace_sd: float = 0.0,
                     wear_prior: tuple[float, float] | None = None,
                     prior_strength: float = 0.1, stint_limits: dict | None = None,
                     pace_tau: float | None = None) -> dict:
    """Full pipeline on any Replay. Never raises for a lap the replay contains.

    The live terminal sets the optional arguments. ``always_simulate`` keeps our own
    stop plan current when the rival is not a threat, ``pit_loss`` and ``neutral``
    replace what these two cars' laps alone can show, and ``pace_shrink`` pulls the
    measured pace gap towards zero before it is held to the flag. ``wear_prior`` is the
    wear curve this circuit normally shows. It steers the plan early in a stint, when
    lap times cannot show wear yet. The hazard model keeps the plain fit it was trained on.
    """
    if pit is None:
        pit, _ = load_models()
    snap = replay.state_at(lap)  # leakage-free: history <= lap only
    if pit_loss is not None:
        snap.pit_loss = pit_loss
    if neutral is not None:
        snap.safety_car = neutral
    if snap.laps_left <= 0:
        return _monitor(lap, theta, "race finished")
    if {snap.own.compound, snap.opp.compound} & WET_COMPOUNDS:
        return _monitor(lap, theta, "wet conditions: the models only cover dry running")
    try:
        deg = DegradationEstimator().fit(snap.history_opp, replay.laps_total)
    except ValueError:
        return _monitor(lap, theta, "opponent model not ready: too few clean laps")
    est = deg.estimate(snap.history_opp, replay.laps_total, lap)
    fc = pit.forecast(snap, deg, k=5)
    p3 = fc.p_within[3]
    try:
        own = OwnCarModel(ridge=prior_strength, prior=wear_prior).fit(snap.history_own,
                                                                      replay.laps_total)
        if wear_prior is not None:
            deg = DegradationEstimator(ridge=prior_strength, prior=wear_prior).fit(
                snap.history_opp, replay.laps_total)
            est = deg.estimate(snap.history_opp, replay.laps_total, lap)
    except ValueError:
        return _monitor(lap, theta, "own-car model not ready: too few clean laps", p3, fc, est)
    loss = expected_loss(p3, own.penalty_at(snap.own.next_age, snap.own.compound),
                         snap.effective_gap, snap.pit_loss)
    trig = ThreatDetector(theta=theta, horizon=3).check(p3, loss, snap.opp.just_pitted)
    if verbose:
        print(f"--- lap {lap}/{replay.laps_total} | gap(own behind opp)={snap.effective_gap:+.1f}s "
              f"| opp {snap.opp.compound} age {snap.opp.tyre_age} | "
              f"own {snap.own.compound} age {snap.own.tyre_age} | pit loss {snap.pit_loss:.1f}s")
        print(f"OPP deg: penalty={est.penalty:+.2f}s slope={est.slope:+.3f}s/lap "
              f"next-lap forecast={est.forecast[0]:.2f}s ±{est.std:.2f}")
        print("OPP hazards:", " ".join(f"L{lap + j}={h:.0%}" for j, h in enumerate(fc.hazards, 1)))
        print(f"P(pit within 3)={p3:.0%} → {trig.status} ({trig.reason})")
    out: dict = {"lap": lap, "p3": p3, "trigger": trig, "forecast": fc, "est": est,
                 "own_model": own}
    if not trig.active and not always_simulate:
        if verbose:
            print("No counter-strategy simulation (monitoring only).")
        return out
    own_slope = min(own.estimate(snap.history_own, replay.laps_total, lap).slope_se, MAX_SLOPE_SE)
    pace_gap = own.baseline_ - deg.baseline_
    if pace_tau is not None:
        # Empirical Bayes. A pace gap from a few laps is mostly noise, one from a long
        # stint mostly is not, so it is pulled towards the gap over the race so far.
        # tau is how far the current gap can truly differ from that.
        noise = (own.resid_std_ ** 2 / max(stint_laps(snap.history_own, replay.laps_total), 1)
                 + deg.resid_std_ ** 2 / max(stint_laps(snap.history_opp, replay.laps_total), 1))
        pace_shrink = pace_tau ** 2 / (pace_tau ** 2 + noise)
        pace_gap -= own.race_baseline_ - deg.race_baseline_
    adj = 0.5 * (1.0 - pace_shrink) * pace_gap
    inp = SimInputs(
        lap=lap, laps_total=replay.laps_total, gap0=snap.effective_gap,
        pit_loss=snap.pit_loss,
        own=CarInputs(own, snap.own.next_age, None if snap.own.just_pitted else snap.own.compound,
                      snap.own.must_stop, own.resid_std_, own_slope, -adj, stint_limits),
        opp=CarInputs(deg, snap.opp.next_age, None if snap.opp.just_pitted else snap.opp.compound,
                      snap.opp.must_stop, deg.resid_std_, min(est.slope_se, MAX_SLOPE_SE), adj,
                      stint_limits),
        pit_dist=pit_dist_override if pit_dist_override is not None else fc.pit_dist,
        overtake_delta=overtake_delta, stop_costs=stop_costs, pace_sd=pace_sd)
    actions = valid_actions(lap, replay.laps_total, snap.own.next_age, snap.own.must_stop, WINDOW)
    results = simulate_actions(inp, actions, n_sims, np.random.default_rng(lap), WINDOW)
    rec = optimise(results, snap.pit_loss, snap.effective_gap, p3, snap.opp.just_pitted)
    if verbose:
        print(f"\nCounter-strategy (Monte Carlo to the flag, n={n_sims}/action):")
        for r in results:
            print(f"  {r.action:9s}: P(ahead)={r.p_ahead:.0%} exp_gap={r.exp_gap:+.1f}s "
                  f"worst5={r.risk:+.1f}s  planned stops at laps {list(r.stops)}")
        print(f"\nRECOMMENDATION: {rec.explanation}")
    out.update({"results": results, "recommendation": rec})
    return out


def print_backtest() -> dict:
    """Show the held-out real-race scores produced by pitwall.train_hazard."""
    m = json.loads(WEIGHTS.read_text())["metrics"]
    print(f"Trained on {m['train']['races']} races ({m['train']['years']}), "
          f"tested on {m['test']['races']} unseen races ({m['test']['years']}).")
    for k in ("model", "baseline", "base_rate"):
        print(f"  {k:10s} Brier={m['test'][k]['brier']:.4f}  LogLoss={m['test'][k]['log_loss']:.4f}")
    return m


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description="PITWALL demo")
    ap.add_argument("--lap", type=int, default=13)
    ap.add_argument("--theta", type=float, default=0.3)
    ap.add_argument("--seed-demo", type=int, default=7)
    ap.add_argument("--race", nargs=2, metavar=("YEAR", "CIRCUIT"))
    ap.add_argument("--own", default="ALO")
    ap.add_argument("--opp", default="VER")
    ap.add_argument("--backtest", action="store_true")
    a = ap.parse_args(argv)
    if a.backtest:
        print_backtest()
    elif a.race:
        replay = load_bundled_replay(int(a.race[0]), a.race[1], a.own, a.opp)
        decide_on_replay(replay, a.lap, a.theta)
    else:
        decide_at(a.lap, a.theta, a.seed_demo)


if __name__ == "__main__":
    main()
