"""End-to-end PITWALL demo (Modules 1→7).

Pipeline per decision lap, using ONLY laps <= lap (no future leakage):
  Replay → DegradationEstimator → PitHazardModel → ThreatDetector
      → OwnCarModel → Monte Carlo simulator → optimiser → recommendation.

Usage:
    python -m pitwall.demo [--lap 12] [--theta 0.5] [--seed-train 1] [--seed-demo 7]
    python -m pitwall.demo --backtest
"""
from __future__ import annotations

import argparse

import numpy as np

from .data import generate_race
from .state import Replay
from .degradation import DegradationEstimator
from .pit_model import PitHazardModel, brier_score, log_loss, age_only_baseline
from .owncar import OwnCarModel
from .threshold import ThreatDetector
from .simulator import simulate_action
from .optimizer import optimise

OFFSETS = [0, 1, 2, 3]
HORIZON = 8
N_SIMS = 1000


def build_models(seed_train: int = 1):
    """Train on several synthetic races with varied pit windows (more positives)."""
    windows = [(12, 26), (13, 27), (14, 28), (15, 29),
               (13, 28), (12, 27), (14, 29), (16, 30)]
    trains = [generate_race(seed=seed_train * 100 + i,
                             opp_pit_laps=w, own_pit_laps=(w[0] + 1, w[1] + 1))
              for i, w in enumerate(windows)]
    pit = PitHazardModel().fit(trains)
    return trains, pit


def decide_at(lap: int, theta: float = 0.5, seed_demo: int = 7,
              pit: PitHazardModel | None = None, verbose: bool = True) -> dict:
    """Pipeline on the synthetic demo race (wrapper, behaviour unchanged)."""
    demo = generate_race(seed=seed_demo, opp_pit_laps=(14, 28), own_pit_laps=(15, 29))
    return decide_on_replay(demo, lap, theta, pit, verbose)


def decide_on_replay(replay: Replay, lap: int, theta: float = 0.5,
                     pit: PitHazardModel | None = None,
                     verbose: bool = True) -> dict:
    """Full pipeline on any Replay (synthetic or FastF1-loaded)."""
    if pit is None:
        _, pit = build_models()
    snap = replay.state_at(lap)  # leakage-free: history <= lap only
    try:
        deg = DegradationEstimator().fit(snap.history_opp, replay.laps_total)
    except ValueError:
        from .threshold import Trigger as _Trigger
        if verbose:
            print(f"--- lap {lap}: insufficient clean laps → MONITOR (model not ready).")
        return {"lap": lap, "p3": 0.0,
                "trigger": _Trigger(False, "MONITOR", "insufficient history", 0.0, 3, theta),
                "forecast": None, "est": None}
    est = deg.estimate(snap.history_opp, replay.laps_total, lap)
    fc = pit.forecast(snap, est.penalty, est.slope, k=5)
    p3 = fc.p_within[3]
    # crude expected-loss proxy: p3 * fresh-tyre advantage over 3 laps
    exp_loss = p3 * max(0.0, est.slope) * 3 * 10
    trig = ThreatDetector(theta=theta, horizon=3).check(p3, exp_loss)
    if verbose:
        print(f"--- lap {lap} | gap(own behind opp)={snap.gap_own_behind_opp:+.1f}s "
              f"| opp {snap.opp.compound} age {snap.opp.tyre_age} | "
              f"own {snap.own.compound} age {snap.own.tyre_age}")
        print(f"OPP deg: penalty={est.penalty:+.2f}s slope={est.slope:+.3f}s/lap "
              f"next-lap forecast={est.forecast[0]:.2f}s ±{est.std:.2f}")
        print("OPP hazards:", " ".join(f"L{lap + j}={h:.0%}" for j, h in enumerate(fc.hazards, 1)))
        print(f"P(pit within 3)={p3:.0%} → {trig.status} ({trig.reason})")
    out: dict = {"lap": lap, "p3": p3, "trigger": trig, "forecast": fc, "est": est}
    if not trig.active:
        if verbose:
            print("No counter-strategy simulation (monitoring only).")
        return out
    own = OwnCarModel().fit(snap.history_own, replay.laps_total)
    own_stay = own.predict_stay(snap.own.tyre_age, lap, replay.laps_total, snap.track_temp, HORIZON)
    own_fresh = own.predict_fresh(lap, replay.laps_total, snap.track_temp, HORIZON)
    opp_stay = list(est.forecast) + [est.forecast[-1] + est.slope * i for i in range(1, HORIZON - 4)]
    clean = [r.lap_time for r in snap.history_opp[-6:] if not r.pitted and not r.safety_car]
    opp_fresh = (min(clean) - 0.3) if clean else est.forecast[0] - 1.0
    rng = np.random.default_rng(0)
    results = [simulate_action(own_stay, own_fresh, opp_stay[:HORIZON], opp_fresh,
                               fc.pit_dist, snap.gap_own_behind_opp, replay.pit_loss,
                               a, opp_stay[:HORIZON], HORIZON,
                               (own.std_ + est.std) / 2, N_SIMS, rng) for a in OFFSETS]
    rec = optimise(results, opp_p3=p3)
    if verbose:
        print("\nCounter-strategy (Monte Carlo, n=1000/action):")
        for r in results:
            print(f"  pit +{r.our_pit_offset}: P(ahead)={r.p_ahead:.0%} "
                  f"exp_gap={r.exp_gap:+.1f}s worst5={r.risk:+.1f}s")
        print(f"\nRECOMMENDATION: {rec.explanation}")
        actual = next((r.lap for r in replay._opp_list if r.pitted and r.lap > lap), None)
        print(f"(History check: opponent actually pitted next at lap {actual}.)")
    out.update({"results": results, "recommendation": rec})
    return out


def backtest(seed_demo: int = 7) -> dict:
    """Score pit model over every lap of a held-out synthetic race vs age-only baseline."""
    _, pit = build_models()
    demo = generate_race(seed=seed_demo, opp_pit_laps=(14, 28), own_pit_laps=(15, 29))
    ys, ps, bs = [], [], []
    for lap in demo.available_laps():
        snap = demo.state_at(lap)
        hist = list(snap.history_opp)
        if len([r for r in hist if not r.pitted and not r.safety_car]) < 6:
            continue
        deg = DegradationEstimator().fit(hist, demo.laps_total)
        est = deg.estimate(hist, demo.laps_total, lap)
        f = PitHazardModel.feats(snap.opp.tyre_age, lap, demo.laps_total,
                                 est.penalty, -snap.gap_own_behind_opp, snap.opp.stops_done)
        ps.append(pit.hazard(f))
        bs.append(age_only_baseline(snap.opp.tyre_age))
        rec = next(r for r in hist if r.lap == lap)
        ys.append(1.0 if rec.pitted else 0.0)
    y = np.asarray(ys)
    out = {"n": len(y), "brier_model": brier_score(y, np.asarray(ps)),
           "brier_base": brier_score(y, np.asarray(bs)),
           "ll_model": log_loss(y, np.asarray(ps)), "ll_base": log_loss(y, np.asarray(bs))}
    print(f"Backtest on held-out race ({out['n']} decision laps):")
    print(f"  Brier  model={out['brier_model']:.4f}  baseline={out['brier_base']:.4f}")
    print(f"  LogLoss model={out['ll_model']:.4f}  baseline={out['ll_base']:.4f}")
    return out


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description="PITWALL proof-of-concept demo")
    ap.add_argument("--lap", type=int, default=13)
    ap.add_argument("--theta", type=float, default=0.3)
    ap.add_argument("--seed-train", type=int, default=1)
    ap.add_argument("--seed-demo", type=int, default=7)
    ap.add_argument("--backtest", action="store_true")
    a = ap.parse_args(argv)
    _, pit = build_models(a.seed_train)
    if a.backtest:
        backtest(a.seed_demo)
    else:
        decide_at(a.lap, a.theta, a.seed_demo, pit)


if __name__ == "__main__":
    main()
