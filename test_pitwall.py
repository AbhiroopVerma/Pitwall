"""Regression tests. Most pin a specific bug from the review, so it cannot return."""
import numpy as np
import pytest

from pitwall.data import (generate_race, load_replay_csv, normalize_fastf1_frame,
                          records_from_frames, save_replay_csv)
from pitwall.degradation import DegradationEstimator
from pitwall.demo import decide_at, decide_on_replay
from pitwall.optimizer import optimise, utility
from pitwall.owncar import OwnCarModel
from pitwall.pit_model import (FEATURES, PitHazardModel, hazard_features, load_models,
                               rows_for_replay, rows_from_records)
from pitwall.simulator import (STAY_OUT, ActionResult, CarInputs, SimInputs, simulate_actions,
                               valid_actions)
from pitwall.state import PIT_IN_SHARE, Replay, must_stop, observed_pit_loss
from pitwall.threshold import ThreatDetector, expected_loss


@pytest.fixture(scope="module")
def pit():
    return load_models()[0]


def demo_race(seed=7):
    return generate_race(seed, opp_pit_laps=(14, 28), own_pit_laps=(15, 29))


# --- replay and leakage -------------------------------------------------------------


def test_replay_no_leakage():
    s = generate_race(seed=7).state_at(12)
    assert all(r.lap <= 12 for r in s.history_opp + s.history_own)
    assert s.opp.private_wear is None  # asymmetric info
    assert s.own.private_wear is not None


def test_changing_future_laps_cannot_change_a_decision(pit):
    a, b = demo_race(), demo_race()
    for r in b._opp_list + b._own_list:
        if r.lap > 13:
            r.lap_time += 50.0
            r.pitted = not r.pitted
    ra = decide_on_replay(a, 13, 0.0, pit, verbose=False, n_sims=100)
    rb = decide_on_replay(b, 13, 0.0, pit, verbose=False, n_sims=100)
    assert ra["p3"] == rb["p3"]
    assert [x.exp_gap for x in ra["results"]] == [x.exp_gap for x in rb["results"]]


def test_hazard_features_exclude_the_stop_being_predicted():
    """The leak: stops_done used to count the label's own stop."""
    rp = demo_race()
    stops_col = FEATURES.index("stops")
    rows = {r[0]: r for r in rows_for_replay(rp)}
    assert rows[14][3] == 1.0
    assert rows[14][1][stops_col] == 0.0  # lap 14 is the stop, nothing done before it
    assert rows[15][1][stops_col] == pytest.approx(1 / 3)


def test_hazard_rows_ignore_the_lap_being_predicted():
    rp = demo_race()
    other = demo_race()
    other._opp_list[13].lap_time += 40.0  # lap 14 in-lap is much slower
    other._opp_list[13].pitted = False
    a = {r[0]: r[1] for r in rows_from_records(rp._opp_list, 40)}
    b = {r[0]: r[1] for r in rows_from_records(other._opp_list, 40)}
    assert np.array_equal(a[14], b[14])


def test_forecast_starts_on_the_next_lap(pit):
    snap = demo_race().state_at(13)
    est = DegradationEstimator().fit(snap.history_opp, 40)
    fc = pit.forecast(snap, est, k=2)
    want = hazard_features(14, "MEDIUM", 14, 40, 0, est.penalty_at(14, "MEDIUM"),
                           est.slope_at(14, "MEDIUM"), False)
    assert fc.hazards[0] == pytest.approx(pit.hazard(want))


def test_no_false_alarm_right_after_a_stop(pit):
    """Used to show about 100% again on the lap the opponent boxed."""
    out = decide_at(14, theta=0.3, pit=pit, verbose=False, )
    assert out["p3"] < 0.2
    assert "just stopped" in out["trigger"].reason


def test_forecast_never_predicts_stop_at_or_past_the_flag(pit):
    snap = demo_race().state_at(38)
    est = DegradationEstimator().fit(snap.history_opp, 40)
    fc = pit.forecast(snap, est, k=5)
    assert fc.hazards[1:] == [0.0, 0.0, 0.0, 0.0]  # L40 onward


def test_forecast_distribution_sums_to_one(pit):
    snap = demo_race().state_at(12)
    est = DegradationEstimator().fit(snap.history_opp, 40)
    fc = pit.forecast(snap, est)
    assert sum(fc.pit_dist) == pytest.approx(1.0)
    assert fc.p_within[5] >= fc.p_within[1]


# --- pace model ---------------------------------------------------------------------


def test_degradation_is_recovered_in_first_stint():
    """Age and fuel are collinear here; the fuel assumption makes it identifiable."""
    errs = []
    for seed in range(40):
        rp = demo_race(400 + seed)
        s = rp.state_at(13)
        est = DegradationEstimator().fit(s.history_opp, 40).estimate(s.history_opp, 40, 13)
        errs.append(est.penalty - (.08 * 13 + .004 * 169))
    assert abs(np.mean(errs)) < 0.35


def test_tyres_never_get_faster_with_age():
    for seed in range(30):
        rp = demo_race(seed)
        for lap in (8, 20, 33):
            s = rp.state_at(lap)
            est = DegradationEstimator().fit(s.history_opp, 40).estimate(s.history_opp, 40, lap)
            assert est.slope >= 0 and est.penalty >= 0


def test_compound_change_is_not_read_as_wear():
    rp = generate_race(3, opp_pit_laps=(14,), own_pit_laps=(15,),
                       compounds=("HARD", "SOFT"))
    s = rp.state_at(30)
    est = DegradationEstimator().fit(s.history_opp, 40)
    assert est.penalty_at(10, "MEDIUM") < 3.0


def test_out_lap_and_in_lap_are_excluded_from_pace_fit():
    s = demo_race().state_at(20)
    base = DegradationEstimator().fit(s.history_opp, 40)
    assert base.resid_std_ < 1.0  # a 21 s lap in the fit would blow this up


def test_missing_values_do_not_poison_the_decision(pit):
    rp = demo_race()
    rp._opp_list[4].track_temp = float("nan")
    rp._opp_list[6].lap_time = float("nan")
    out = decide_on_replay(rp, 13, 0.0, pit, verbose=False, n_sims=50)
    assert np.isfinite(out["p3"]) and "recommendation" in out


def test_private_signal_improves_own_car_estimate():
    pub, priv = [], []
    for seed in range(30):
        s = demo_race(500 + seed).state_at(13)
        truth = .08 * 13 + .004 * 169
        priv.append(abs(OwnCarModel().fit(s.history_own, 40).penalty_at(13, "MEDIUM") - truth))
        pub.append(abs(DegradationEstimator().fit(s.history_own, 40).penalty_at(13, "MEDIUM") - truth))
    assert np.mean(priv) < 0.5 * np.mean(pub)


# --- simulator ----------------------------------------------------------------------


def sim_inputs(lap=13, gap=0.0, dist=None, delta=1.0, seed=7, must=True):
    rp = generate_race(seed, opp_pit_laps=(14, 28), own_pit_laps=(14, 28),
                       base_own=93.0, base_opp=93.0)
    s = rp.state_at(lap)
    o = OwnCarModel().fit(s.history_own, 40)
    d = DegradationEstimator().fit(s.history_opp, 40)
    return SimInputs(lap, 40, gap, 21.0,
                     CarInputs(o, s.own.next_age, s.own.compound, must, o.resid_std_, 0.0),
                     CarInputs(d, s.opp.next_age, s.opp.compound, must, d.resid_std_, 0.0),
                     dist or [1, 0, 0, 0, 0, 0], overtake_delta=delta)


def test_each_stop_costs_exactly_the_pit_loss():
    from pitwall.simulator import _Planner
    inp = sim_inputs()
    pl = _Planner(inp.own, 14, 27, 21.0)
    none = pl.lap_times(())[0]
    ages_none = pl._ages.copy()
    one = pl.lap_times((5,))[0]
    ages_one = pl._ages.copy()
    tyres = (sum(pl._penalty(a, "MEDIUM" if i <= 5 else None) for i, a in enumerate(ages_one))
             - sum(pl._penalty(a, "MEDIUM") for a in ages_none))
    assert one.sum() - none.sum() == pytest.approx(21.0 + tyres)


def test_identical_cars_are_a_coin_flip():
    """Symmetry: this used to give our car +4.8 s and a 99% chance of being ahead."""
    gaps, wins = [], []
    for seed in range(500, 530):
        inp = sim_inputs(seed=seed, dist=[1, 0, 0, 0, 0, 0], delta=99.0)
        # the two cars get the same model, so only the random draws differ
        inp.opp = CarInputs(inp.own.model, inp.own.next_age, inp.own.compound, True,
                            inp.own.noise_sd, 0.0)
        r = simulate_actions(inp, [0], 400, np.random.default_rng(seed))[0]
        gaps.append(r.exp_gap)
        wins.append(r.p_ahead)
    assert abs(np.mean(gaps)) < 0.5
    assert 0.35 < np.mean(wins) < 0.65


def test_opponent_timing_changes_the_relative_value_of_our_options():
    """The core bug: gap differences between actions were identical for any forecast."""
    spreads = []
    for dist in ([1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]):
        inp = sim_inputs(gap=-0.8, dist=dist, delta=2.0)
        r = simulate_actions(inp, [0, 3, STAY_OUT], 300, np.random.default_rng(1))
        spreads.append(round(r[0].exp_gap - r[1].exp_gap, 1))
    assert len(set(spreads)) > 1


def test_common_random_numbers_make_equal_actions_identical():
    inp = sim_inputs()
    a, b = simulate_actions(inp, [1, 1], 200, np.random.default_rng(0))
    assert a.gaps == b.gaps


def test_held_up_car_cannot_pass_without_pace_advantage():
    def run(delta):
        inp = sim_inputs(lap=35, gap=2.0, dist=[0, 0, 0, 0, 0, 1], delta=delta, must=False)
        inp.own.noise_sd = inp.opp.noise_sd = 0.0
        inp.opp.model = inp.own.model
        inp.opp.next_age = inp.own.next_age + 10  # they are slower on older tyres
        return simulate_actions(inp, [STAY_OUT], 20, np.random.default_rng(0))[0]
    assert run(delta=50.0).p_ahead == 0.0   # a Monaco: stuck behind
    assert run(delta=0.05).p_ahead == 1.0   # a Monza: straight past


# --- candidate actions ---------------------------------------------------------------


def test_stay_out_is_always_offered_when_legal():
    assert STAY_OUT in valid_actions(20, 40, 8, False)


def test_final_laps_do_not_offer_pointless_stops():
    assert valid_actions(39, 40, 10, False) == [STAY_OUT]
    assert 0 in valid_actions(37, 40, 10, False) and 2 not in valid_actions(37, 40, 10, False)


def test_car_that_just_stopped_is_not_told_to_stop_again():
    assert 0 not in valid_actions(20, 40, 1, False)


def test_required_stop_removes_stay_out_when_too_late():
    assert STAY_OUT not in valid_actions(36, 40, 30, True)


def test_late_race_does_not_recommend_a_pointless_stop(pit):
    """Lap 39 of 40 used to recommend paying 21 s for one lap of tyres."""
    for lap in (37, 38, 39):
        out = decide_on_replay(demo_race(), lap, 0.0, pit, verbose=False, n_sims=200)
        assert out["recommendation"].best_offset is STAY_OUT


def test_must_stop_rule():
    rp = generate_race(1, opp_pit_laps=(), own_pit_laps=())
    assert must_stop(rp.state_at(10).history_own)
    rp2 = demo_race()
    assert not must_stop(rp2.state_at(20).history_opp)


# --- optimiser and trigger -----------------------------------------------------------


def result(offset, p, gap, risk):
    return ActionResult("x", offset, p, gap, risk, [])


def test_utility_terms_share_a_scale():
    assert utility(result(0, 0.5, -500.0, -900.0), 21.0) >= 0.5 - 0.25 - 0.10


def test_ties_go_to_the_better_gap_not_the_first_listed():
    """A car certain to win was told to take a pointless stop."""
    res = [result(0, 1.0, 5.0, 1.0), result(None, 1.0, 25.0, 20.0)]
    assert optimise(res, 21.0, gap0=-20.0).best_offset is None


def test_explanation_matches_the_situation():
    res = [result(0, 0.6, 1.0, -1.0), result(None, 0.4, -1.0, -3.0)]
    lead = optimise(res, 21.0, gap0=-1.0, opp_p3=0.5).explanation
    trail = optimise(res, 21.0, gap0=1.0, opp_p3=0.5).explanation
    assert "protects our lead" in lead and "undercut" in trail
    stay = optimise([result(None, 0.7, 1.0, -1.0), result(0, 0.4, -1.0, -3.0)],
                    21.0, gap0=-1.0, opp_p3=0.5).explanation
    assert stay.startswith("Recommend: stay out")
    assert "protects" not in stay


def test_explanation_does_not_promise_an_undercut_in_a_lost_fight():
    text = optimise([result(0, 0.0, -100.0, -150.0)], 21.0, gap0=26.0, opp_p3=0.6).explanation
    assert "out of reach" in text and "undercut" not in text


def test_threat_fades_with_the_gap():
    near = expected_loss(0.6, 1.5, 1.0, 21.0)
    far = expected_loss(0.6, 1.5, 60.0, 21.0)
    assert near > 2.0 and far == 0.0


def test_theta_only_acts_as_threshold():
    d = ThreatDetector(theta=0.5, lam=99.0)
    assert not d.check(0.49).active
    assert d.check(0.5).active
    assert d.check(0.0, opp_just_stopped=True).active


# --- robustness ----------------------------------------------------------------------


def test_too_few_own_laps_monitors_instead_of_crashing(pit):
    rp = generate_race(seed=7)
    for r in rp._own_list[1:4]:
        r.safety_car = True  # own car has fewer clean laps than the opponent
    out = decide_on_replay(rp, 8, 0.0, pit, verbose=False)
    assert out["trigger"].status == "MONITOR" and "own-car" in out["trigger"].reason


def test_wet_running_is_declared_out_of_scope(pit):
    rp = demo_race()
    for r in rp._opp_list:
        r.compound = "INTERMEDIATE"
    out = decide_on_replay(rp, 13, 0.0, pit, verbose=False)
    assert "wet" in out["trigger"].reason


def test_lap_after_race_end_is_reported_not_simulated(pit):
    out = decide_on_replay(demo_race(), 40, 0.0, pit, verbose=False)
    assert out["trigger"].reason == "race finished"


# --- real-data mapping ---------------------------------------------------------------


def _fake_frames():
    """12-lap FastF1-shaped frames (no network).

    Stops follow real timing: a small in-lap loss and most of it on the out-lap.
    Opp stops end of L8, own stops end of L10.
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
                lt += 3.5
            if pit_out:
                lt += 21.0
            rows.append({
                "LapNumber": lap, "Compound": "MEDIUM",
                "TyreLife": float(age), "LapTime": timedelta(seconds=lt),
                "Position": 2.0, "TrackStatus": "1",
                "PitInTime": timedelta(seconds=t + lt - 1) if pit_in else pd.NaT,
                "PitOutTime": timedelta(seconds=t + 20) if pit_out else pd.NaT,
                "LapStartTime": timedelta(seconds=t + start_offset)})
            t += lt
            if pit_in:
                age = 0
        return pd.DataFrame(rows)

    return frame(10, 95.0, 1.5), frame(8, 94.8, 0.0)


def _fake_replay():
    a, b = _fake_frames()
    total, own_r, opp_r, prior = records_from_frames(normalize_fastf1_frame(a),
                                                     normalize_fastf1_frame(b))
    return Replay(total, own_r, opp_r, prior)


def test_real_stop_cost_includes_the_out_lap():
    """Used to measure the in-lap only, so every real race hit the 15 s floor."""
    rp = _fake_replay()
    assert 22.0 < rp.state_at(12).pit_loss < 27.0


def test_pit_loss_cannot_use_a_stop_that_has_not_happened():
    rp = _fake_replay()
    assert rp.state_at(7).pit_loss == rp.pit_loss  # prior: no stop seen
    assert rp.state_at(8).pit_loss == rp.pit_loss  # out-lap not driven yet
    assert rp.state_at(9).pit_loss != rp.pit_loss


def test_real_laps_stay_raw_and_flagged():
    rp = _fake_replay()
    out = rp._opp_list[8]  # lap 9
    assert out.pit_out and out.lap_time > 110.0
    assert out.tyre_age == 1
    assert [r.lap for r in rp._own_list if r.pitted] == [10]
    assert [r.lap for r in rp._opp_list if r.pitted] == [8]


def test_real_gap_is_measured_at_the_end_of_the_lap():
    rp = _fake_replay()
    own, opp = rp._own_list[0], rp._opp_list[0]
    assert own.gap_to_rival == pytest.approx(1.5 + own.lap_time - opp.lap_time)
    assert all(o.gap_to_rival == -w.gap_to_rival for o, w in zip(rp._opp_list, rp._own_list))


def test_effective_gap_counts_the_rest_of_a_stop_in_progress():
    rp = _fake_replay()
    s = rp.state_at(8)
    assert s.opp.just_pitted
    pending = (1 - PIT_IN_SHARE) * s.pit_loss
    assert s.effective_gap == pytest.approx(s.gap_own_behind_opp - pending)


def test_yellow_flag_is_not_a_safety_car():
    a, b = _fake_frames()
    a.loc[2, "TrackStatus"] = "2"
    a.loc[3, "TrackStatus"] = "4"
    a.loc[4, "TrackStatus"] = "12"
    _, own_r, _, _ = records_from_frames(normalize_fastf1_frame(a), normalize_fastf1_frame(b))
    assert [r.safety_car for r in own_r[2:5]] == [False, True, False]


def test_official_distance_is_used_when_known():
    a, b = _fake_frames()
    total, *_ = records_from_frames(normalize_fastf1_frame(a), normalize_fastf1_frame(b),
                                    laps_total=57)
    assert total == 57


def test_csv_cache_roundtrip(tmp_path):
    rp = _fake_replay()
    path = tmp_path / "race.csv"
    save_replay_csv(rp, rp.pit_loss, {"t": 1}, path)
    rp2 = load_replay_csv(path)
    assert rp2.laps_total == rp.laps_total
    assert [r.lap for r in rp2._opp_list if r.pitted] == [r.lap for r in rp._opp_list if r.pitted]
    assert [r.pit_out for r in rp2._opp_list] == [r.pit_out for r in rp._opp_list]


def test_decide_on_real_shaped_replay(pit):
    out = decide_on_replay(_fake_replay(), 10, 0.0, pit, verbose=False, n_sims=50)
    assert out["trigger"].active
    assert out["recommendation"].best_offset in (0, 1, 2, 3, STAY_OUT)


# --- shipped model and dashboard ------------------------------------------------------


def test_hazard_fit_learns_from_replays_without_network():
    trains = [generate_race(100 + i, opp_pit_laps=(12 + i, 26 + i), own_pit_laps=(13, 27))
              for i in range(4)]
    m = PitHazardModel().fit(trains)
    assert m.beta is not None and np.all(np.isfinite(m.beta))


def test_shipped_model_beats_age_baseline_on_unseen_races():
    import json
    from pitwall.pit_model import WEIGHTS
    t = json.loads(WEIGHTS.read_text())["metrics"]["test"]
    assert t["model"]["log_loss"] < t["baseline"]["log_loss"]
    assert t["model"]["log_loss"] < t["base_rate"]["log_loss"]


APP = "pitwall/app.py"


def _app():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(APP, default_timeout=120)


def test_dashboard_runs_on_every_source_and_stays_honest():
    from unittest import mock
    at = _app().run()
    assert not at.exception
    at.sidebar.radio[0].set_value("Synthetic demo").run()
    assert not at.exception
    with mock.patch("pitwall.data.load_fastf1_replay", side_effect=RuntimeError("no network")):
        at = _app().run()
        at.sidebar.radio[0].set_value("FastF1 (needs F1 login)").run()
    assert not at.exception
    assert any("SYNTHETIC" in c.value for c in at.caption)
    assert not any("REAL" in c.value for c in at.caption)


def test_dashboard_survives_an_opponent_retirement():
    from unittest import mock
    base = demo_race()
    dnf = Replay(40, base._own_list, [r for r in base._opp_list if r.lap <= 20], 21.0)
    with mock.patch("pitwall.data.load_fastf1_replay", return_value=dnf):
        at = _app().run()
        at.sidebar.radio[0].set_value("FastF1 (needs F1 login)").run()
        at.sidebar.select_slider[0].set_value(20).run()
    assert not at.exception
    assert list(at.sidebar.select_slider[0].options)[-1] == "20"
