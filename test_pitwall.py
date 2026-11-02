"""Smoke tests: no-leakage replay, model training, trigger, full decide_at."""
import numpy as np

from pitwall.data import (generate_race, load_replay_csv,
                         normalize_fastf1_frame, records_from_frames,
                         save_replay_csv)
from pitwall.degradation import DegradationEstimator
from pitwall.pit_model import PitHazardModel
from pitwall.state import Replay
from pitwall.threshold import ThreatDetector
from pitwall.demo import decide_at, decide_on_replay, backtest


def test_replay_no_leakage():
    rp = generate_race(seed=7)
    s12 = rp.state_at(12)
    assert all(r.lap <= 12 for r in s12.history_opp + s12.history_own)
    assert s12.opp.private_wear is None  # asymmetric info
    assert s12.own.private_wear is not None


def test_deg_and_hazard_train():
    train = generate_race(seed=1, opp_pit_laps=(13, 27))
    pit = PitHazardModel(iters=500).fit([train])
    demo = generate_race(seed=7)
    snap = demo.state_at(12)
    est = DegradationEstimator().fit(snap.history_opp, demo.laps_total).estimate(
        snap.history_opp, demo.laps_total, 12)
    assert est.slope > 0  # tyres degrade
    fc = pit.forecast(snap, est.penalty, est.slope)
    assert abs(sum(fc.pit_dist) - 1.0) < 1e-6
    assert fc.p_within[5] >= fc.p_within[1]  # cumulative is monotone


def test_trigger_and_decision():
    out = decide_at(lap=12, theta=0.0, verbose=False)  # force trigger
    assert out["trigger"].active
    assert out["recommendation"].best_offset in (0, 1, 2, 3)
    out2 = decide_at(lap=5, theta=0.999, verbose=False)  # early lap, strict theta
    assert out2["trigger"].status in ("MONITOR", "TRIGGERED")


def test_backtest_beats_baseline():
    res = backtest()
    assert res["brier_model"] <= res["brier_base"]


def _fake_frames():
    """12-lap fabricated FastF1-shaped frames (no network).

    Opp stops end of L8, own stops end of L10; ~21 s in-laps, slow out-laps.
    """
    import pandas as pd
    from datetime import timedelta

    def frame(pit_lap, base, start_offset):
        rows, t, age = [], 0.0, 0
        for lap in range(1, 13):
            age += 1
            lt = base + 0.08 * age
            pit_in, pit_out = lap == pit_lap, lap == pit_lap + 1
            if pit_in:
                lt += 21.0
            if pit_out:
                lt += 4.0
            rows.append({
                "LapNumber": lap, "Compound": "MEDIUM",
                "TyreLife": float(age), "LapTime": timedelta(seconds=lt),
                "Position": 2.0, "TrackStatus": "1",
                "PitInTime": (timedelta(seconds=t + lt - 21.0)
                              if pit_in else pd.NaT),
                "PitOutTime": (timedelta(seconds=t + 3.0)
                               if pit_out else pd.NaT),
                "LapStartTime": timedelta(seconds=t + start_offset)})
            t += lt
            if pit_in:
                age = 0
        return pd.DataFrame(rows)

    return frame(10, 95.0, 1.5), frame(8, 94.8, 0.0)


def _fake_replay():
    own_n = normalize_fastf1_frame(_fake_frames()[0])
    opp_n = normalize_fastf1_frame(_fake_frames()[1])
    laps_total, own_r, opp_r, ploss = records_from_frames(own_n, opp_n)
    return Replay(laps_total, own_r, opp_r, ploss)


def test_fastf1_mapping_fabricated():
    own_n = normalize_fastf1_frame(_fake_frames()[0])
    assert len(own_n) == 12 and own_n["pit_in"].sum() == 1
    laps_total, own_r, opp_r, ploss = records_from_frames(
        own_n, normalize_fastf1_frame(_fake_frames()[1]))
    assert laps_total == 12
    assert [r.lap for r in own_r if r.pitted] == [10]
    assert [r.lap for r in opp_r if r.pitted] == [8]
    assert opp_r[8].tyre_age == 1  # age resets after the stop (lap 9)
    assert opp_r[8].lap_time < 100.0  # out-lap sanitized, not raw slow lap
    assert abs(own_r[0].gap_to_rival - 1.5) < 1e-9  # initial start offset
    assert all(o.gap_to_rival == -w.gap_to_rival
               for o, w in zip(opp_r, own_r))
    assert 19.0 < ploss < 23.0  # in-lap minus clean pace ~= 21 s


def test_csv_cache_roundtrip():
    import tempfile
    from pathlib import Path

    rp = _fake_replay()
    with tempfile.TemporaryDirectory() as d:
        csv_path = Path(d) / "race.csv"
        save_replay_csv(rp, rp.pit_loss, {"t": 1}, csv_path)
        rp2 = load_replay_csv(csv_path)
    assert rp2.laps_total == rp.laps_total
    assert abs(rp2.pit_loss - rp.pit_loss) < 1e-6
    assert ([r.lap for r in rp2._opp_list if r.pitted]
            == [r.lap for r in rp._opp_list if r.pitted])


def test_decide_on_replay_fabricated():
    train = generate_race(seed=1, opp_pit_laps=(13, 27))
    pit = PitHazardModel(iters=500).fit([train])
    out = decide_on_replay(_fake_replay(), lap=10, theta=0.0,
                           pit=pit, verbose=False)  # force trigger
    assert out["trigger"].active
    assert abs(sum(out["forecast"].pit_dist) - 1.0) < 1e-6
    assert out["recommendation"].best_offset in (0, 1, 2, 3)
