"""Train the hazard model on real races and score it on later, unseen ones.

    python -m pitwall.train_hazard

The split is by season: nothing from a test race touches training (spec §12).
Wet races are left out, because those stops are driven by weather and the models
are dry-only. Results are stored next to the weights so the app can show them.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .data import bundled_laps, driver_frame, records_from_frames
from .degradation import DegradationEstimator, MIN_CLEAN_LAPS, clean_laps
from .pit_model import (AGE_ONLY, FEATURES, WEIGHTS, PitHazardModel, brier_score, log_loss,
                        rows_from_records)
from .state import Replay, WET_COMPOUNDS


def car_rows(race, acronym: str, total: int):
    frame = driver_frame(race, acronym)
    _, recs, _, _ = records_from_frames(frame, frame, laps_total=total)
    return recs, rows_from_records(recs, total)


def is_wet(race) -> bool:
    return float(race["compound"].isin(WET_COMPOUNDS).mean()) > 0.05


def collect(df, years: list[int]):
    """Feature rows plus the records needed for multi-lap scoring."""
    X, y, cars, races = [], [], [], 0
    for (year, circuit), race in df[df["year"].isin(years)].groupby(["year", "circuit"]):
        if is_wet(race):
            continue
        races += 1
        total = int(race["total_laps"].iloc[0])
        for acr in race["acronym"].unique():
            recs, rows = car_rows(race, acr, total)
            X += [r[1] for r in rows]
            y += [r[3] for r in rows]
            cars.append((recs, total))
    return np.array(X), np.array(y), cars, races


def calibration_table(p: np.ndarray, y: np.ndarray, edges) -> list[dict]:
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum():
            out.append({"from": lo, "to": min(hi, 1.0), "n": int(m.sum()),
                        "predicted": float(p[m].mean()), "actual": float(y[m].mean())})
    return out


def score_windows(model: PitHazardModel, cars, theta: float = 0.3) -> dict:
    """Score P(stop within 3 laps): calibration, recall and false alarms."""
    ps, ys, caught, stops = [], [], 0, 0
    alarms_laps = fired = 0
    for recs, total in cars:
        replay = Replay(total, recs, recs)
        pits = [r.lap for r in recs if r.pitted]
        for lap in replay.available_laps():
            if lap >= total - 1:
                continue
            snap = replay.state_at(lap)
            if len(clean_laps(snap.history_opp, total)) < MIN_CLEAN_LAPS:
                continue
            est = DegradationEstimator().fit(snap.history_opp, total)
            p3 = model.forecast(snap, est, k=3).p_within[3]
            hit = any(lap < s <= lap + 3 for s in pits)
            ps.append(p3)
            ys.append(float(hit))
            if not hit:
                alarms_laps += 1
                fired += int(p3 >= theta)
        for s in pits:
            if s - 1 not in replay._opp:
                continue
            windows = []
            for lap in (s - 3, s - 2, s - 1):
                if lap not in replay._opp:
                    continue
                snap = replay.state_at(lap)
                if len(clean_laps(snap.history_opp, total)) < MIN_CLEAN_LAPS:
                    continue
                est = DegradationEstimator().fit(snap.history_opp, total)
                windows.append(model.forecast(snap, est, k=3).p_within[3])
            if windows:
                stops += 1
                caught += int(max(windows) >= theta)
    p, y = np.array(ps), np.array(ys)
    return {"theta": theta, "stops_scored": stops,
            "stop_recall": caught / max(stops, 1),
            "false_alarm_rate": fired / max(alarms_laps, 1),
            "brier": brier_score(y, p), "base_rate": float(y.mean()),
            "brier_base_rate": brier_score(y, np.full_like(p, y.mean())),
            "calibration": calibration_table(p, y, [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.01])}


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-years", nargs="+", type=int, default=[2023, 2024])
    ap.add_argument("--test-years", nargs="+", type=int, default=[2025])
    a = ap.parse_args(argv)
    df = bundled_laps()
    Xtr, ytr, _, ntr = collect(df, a.train_years)
    Xte, yte, cars, nte = collect(df, a.test_years)
    print(f"train: {ntr} races, {len(ytr)} laps, {int(ytr.sum())} stops | "
          f"test: {nte} races, {len(yte)} laps, {int(yte.sum())} stops", flush=True)
    model = PitHazardModel().fit_rows(Xtr, ytr)
    base = PitHazardModel(columns=AGE_ONLY).fit_rows(Xtr, ytr)
    pm, pb = model.hazards(Xte), base.hazards(Xte)
    rate = np.full_like(yte, ytr.mean())
    metrics = {
        "features": FEATURES,
        "train": {"years": a.train_years, "races": ntr, "laps": len(ytr), "stops": int(ytr.sum())},
        "test": {"years": a.test_years, "races": nte, "laps": len(yte), "stops": int(yte.sum()),
                 "model": {"brier": brier_score(yte, pm), "log_loss": log_loss(yte, pm)},
                 "baseline": {"brier": brier_score(yte, pb), "log_loss": log_loss(yte, pb)},
                 "base_rate": {"brier": brier_score(yte, rate), "log_loss": log_loss(yte, rate)},
                 "calibration": calibration_table(pm, yte, [0, .02, .05, .1, .2, .3, .5, 1.01]),
                 "window_model": score_windows(model, cars)}}
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    WEIGHTS.write_text(json.dumps({"model": model.to_dict(), "baseline": base.to_dict(),
                                   "metrics": metrics}, indent=1))
    print(json.dumps(metrics["test"], indent=1))
    return metrics


if __name__ == "__main__":
    main()
