"""Tests for the live terminal. Offline, except for a local fake of the OpenF1 server."""
from __future__ import annotations

import gzip
import http.server
import json
import os
import pty
import select
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from pitwall.data import bundled_race
from pitwall.live import Clock, main
from pitwall.live_engine import CallTracker, Frame, LiveEngine, Option, _confidence
from pitwall.live_eval import _changes, battle_pairs, dry_races, play_pair
from pitwall.live_feed import (FlagEvent, LapEvent, OpenF1Client, OpenF1LiveFeed, OpenF1Normalizer,
                               PitEvent, RaceInfo, RehearsalClient, ScheduledFeed, bundled_feed)
from pitwall.live_field import FieldState
from pitwall.live_view import clip, lap_time, render, rival_stop_outcome, visible_len
from pitwall.pit_model import load_models
from pitwall.priors import RacePriors, load_priors
from pitwall.simulator import CarInputs, SimInputs, simulate_actions
from pitwall.degradation import DegradationEstimator
from pitwall.state import LapRecord
from pitwall.synthetic import SCENARIOS, synthetic_feed

MODEL = load_models()[0]
FIXTURE = Path(__file__).parent / "fixtures" / "openf1_mexico_2026.json.gz"


def lap(driver, n, t, time_=90.0, compound="MEDIUM", age=None, pit_in=False, pit_out=False,
        neutral=False):
    return LapEvent(t, driver, n, time_, compound, age if age is not None else n, pit_in, pit_out, neutral)


def info(laps=50, drivers=("AAA", "BBB", "CCC")):
    return RaceInfo("test", "REAL REPLAY", None, None, laps, list(drivers), 0.0)


def engine_for(feed, own, opp, **kw):
    return LiveEngine(feed.info, own, opp, n_sims=kw.pop("n_sims", 150), pit_model=MODEL, **kw)


def run(feed, engine, until=None):
    frame = None
    for e in feed.events:
        if until is not None and e.t > until:
            break
        engine.ingest([e])
        frame = engine.tick(e.t)
    return frame


# -- feeds --------------------------------------------------------------------------


def test_scheduled_feed_only_releases_what_has_happened():
    feed = ScheduledFeed(info(), [lap("AAA", 1, 100.0), lap("AAA", 2, 190.0), lap("AAA", 3, 280.0)])
    assert [e.lap for e in feed.poll(50.0)] == []
    assert [e.lap for e in feed.poll(190.0)] == [1, 2]
    assert feed.next_time() == 280.0
    assert [e.lap for e in feed.poll(1e9)] == [3]
    assert feed.exhausted


def test_bundled_feed_is_in_time_order_with_one_event_per_finished_lap():
    feed = bundled_feed(2025, "Monza")
    times = [e.t for e in feed.events]
    assert times == sorted(times)
    laps = [(e.driver, e.lap) for e in feed.events if isinstance(e, LapEvent)]
    assert len(laps) == len(set(laps))
    assert feed.info.laps_total == 53


def test_engine_output_does_not_depend_on_the_future():
    """The core promise of a replay: cut the race off and nothing before it changes."""
    full = bundled_feed(2025, "Silverstone")
    cut_t = next(e.t for e in full.events if getattr(e, "driver", "") == "LEC" and e.lap == 20)
    frame_a = run(full, engine_for(full, "LEC", "HAM"), until=cut_t)
    cut = bundled_feed(2025, "Silverstone")
    cut.events = [e for e in cut.events if e.t <= cut_t]
    frame_b = run(cut, engine_for(cut, "LEC", "HAM"))
    assert frame_a.lap == frame_b.lap
    assert frame_a.call.headline == frame_b.call.headline
    assert frame_a.gap == frame_b.gap
    assert [o.p_ahead for o in frame_a.options] == [o.p_ahead for o in frame_b.options]


# -- field --------------------------------------------------------------------------


def _field(*cars):
    f = FieldState()
    for code, offset, pace in cars:
        for n in range(1, 11):
            f.add(lap(code, n, offset + n * pace, pace))
    return f


def test_order_and_gaps_use_the_last_lap_both_finished():
    f = _field(("AAA", 0.0, 90.0), ("BBB", 5.0, 90.0))
    f.add(lap("AAA", 11, 990.0))  # A is a lap further on than B for a moment
    assert f.order() == ["AAA", "BBB"]
    assert f.gap_to("BBB", "AAA") == pytest.approx(5.0)
    assert f.gap_to("AAA", "BBB") == pytest.approx(-5.0)


def test_rejoin_counts_the_cars_we_fall_behind():
    # We lead. C is 10s back, D is 30s back, and a stop costs 22s.
    f = _field(("AAA", 0.0, 90.0), ("CCC", 10.0, 90.0), ("DDD", 30.0, 90.0))
    now = 10 * 90.0 + 31.0
    r = f.rejoin("AAA", 11, 22.0, now)
    assert r.position == 2
    assert r.ahead == "CCC" and r.ahead_gap == pytest.approx(12.0)
    assert r.behind == "DDD" and r.behind_gap == pytest.approx(8.0)


def test_rejoin_sees_a_lapped_car_on_the_road():
    f = FieldState()
    for n in range(1, 11):
        f.add(lap("AAA", n, n * 90.0, 90.0))
    # This car runs one full lap and 1 second down: nowhere in the race, but right there
    # on the road when we come out.
    for n in range(1, 10):
        f.add(lap("LAP", n, (n + 1) * 90.0 + 21.0 - 90.0 + 90.0, 90.0))
    r = f.rejoin("AAA", 11, 22.0, 10 * 90.0 + 1.0)
    assert r.position == 1
    assert r.road_ahead == "LAP" and r.road_gap == pytest.approx(1.0)
    assert r.traffic


def test_pit_loss_prefers_this_race_once_enough_stops_are_seen():
    f = FieldState()
    for code in ("AAA", "BBB", "CCC"):
        for n in range(1, 9):
            f.add(lap(code, n, n * 90.0, 90.0))
        assert f.pit_loss(25.0) == 25.0
        f.add(lap(code, 9, 9 * 90.0 + 3.0, 93.0, pit_in=True))
        f.add(lap(code, 10, 10 * 90.0 + 18.0 + 3.0, 108.0, pit_out=True))
    assert f.pit_loss(25.0) == pytest.approx(21.0)


def test_retired_car_is_noticed():
    f = _field(("AAA", 0.0, 90.0))
    assert f.is_running("AAA", 950.0)
    assert not f.is_running("AAA", 900.0 + 4 * 90.0)


# -- OpenF1 parsing -----------------------------------------------------------------


def _raw_lap(n, start_s, dur=90.0, out=False, driver=1):
    return {"driver_number": driver, "lap_number": n, "is_pit_out_lap": out,
            "date_start": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(start_s)),
            "lap_duration": dur}


T0 = 1_800_000_000


def test_normalizer_holds_back_a_lap_still_in_progress():
    n = OpenF1Normalizer()
    n.add(laps=[_raw_lap(1, T0, None)])
    assert n.events(T0 + 300) == []


def test_normalizer_is_idempotent_under_repeated_polls():
    n = OpenF1Normalizer()
    rows = [_raw_lap(1, T0), _raw_lap(2, T0 + 90)]
    n.add(laps=rows)
    first = n.events(T0 + 91)
    n.add(laps=rows)
    assert [e.lap for e in first] == [1]
    assert n.events(T0 + 92) == []


def test_normalizer_waits_briefly_for_the_next_lap_row():
    n = OpenF1Normalizer(grace_s=8.0)
    n.add(laps=[_raw_lap(1, T0)])
    assert n.events(T0 + 92) == []   # line crossed, grace not over
    assert [e.lap for e in n.events(T0 + 99)] == [1]


def test_pit_row_on_the_out_lap_still_marks_the_in_lap():
    """OpenF1 gave the out-lap as pit.lap_number for 9 of 67 stops in the 2026 Mexico race."""
    n = OpenF1Normalizer()
    n.add(laps=[_raw_lap(5, T0), _raw_lap(6, T0 + 90, out=True), _raw_lap(7, T0 + 200)],
          pits=[{"driver_number": 1, "lap_number": 6}])
    events = {e.lap: e for e in n.events(T0 + 400)}
    assert events[5].pit_in and not events[6].pit_in
    assert events[6].pit_out


def test_pit_news_after_the_lap_was_reported_is_sent_as_a_correction():
    n = OpenF1Normalizer(grace_s=1.0)
    n.add(laps=[_raw_lap(5, T0)])
    assert not n.events(T0 + 95)[0].pit_in
    n.add(pits=[{"driver_number": 1, "lap_number": 5}])
    late = n.events(T0 + 96)
    assert isinstance(late[0], PitEvent) and late[0].lap == 5


def test_tyre_comes_from_the_stint_and_counts_used_sets():
    n = OpenF1Normalizer()
    n.add(laps=[_raw_lap(4, T0), _raw_lap(5, T0 + 90)],
          stints=[{"driver_number": 1, "stint_number": 1, "lap_start": 1, "lap_end": None,
                   "compound": "HARD", "tyre_age_at_start": 3}])
    e = n.events(T0 + 91)[0]
    assert e.compound == "HARD" and e.tyre_age == 7


def test_flags_open_and_close_and_mark_laps():
    n = OpenF1Normalizer()
    stamp = lambda s: time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(s))
    n.add(control=[
        {"date": stamp(T0 + 30), "category": "SafetyCar", "message": "VSC DEPLOYED"},
        {"date": stamp(T0 + 60), "category": "SafetyCar", "message": "SAFETY CAR DEPLOYED"},
        {"date": stamp(T0 + 400), "category": "SafetyCar", "message": "SAFETY CAR IN THIS LAP"},
    ], laps=[_raw_lap(1, T0), _raw_lap(2, T0 + 90)])
    events = n.events(T0 + 100)
    assert [(type(e).__name__, getattr(e, "kind", None)) for e in events] == [
        ("FlagEvent", "VSC"), ("FlagEvent", "SC"), ("LapEvent", None)]
    assert events[2].neutral
    assert [e.kind for e in n.events(T0 + 500) if isinstance(e, FlagEvent)] == [None]


@pytest.fixture(scope="module")
def mexico():
    with gzip.open(FIXTURE, "rt") as f:
        return json.load(f)


def test_live_parser_agrees_with_the_published_race(mexico):
    """Real 2026 rows, revealed one poll at a time, must rebuild the same race."""
    now = {"t": 0.0}
    client = RehearsalClient(mexico, lambda: now["t"])
    norm = OpenF1Normalizer()
    norm.add(drivers=mexico["drivers"])
    from pitwall.live_feed import _ts

    starts = [_ts(r["date_start"]) for r in mexico["laps"] if r.get("date_start")]
    laps_seen, pit_marks = {}, set()
    t = min(starts)
    while t < max(starts) + 200:
        now["t"] = t
        norm.add(laps=client.get("laps"), stints=client.get("stints"), pits=client.get("pit"),
                 control=client.get("race_control"))
        for e in norm.events(t):
            if isinstance(e, LapEvent):
                assert (e.driver, e.lap) not in laps_seen, "a lap was reported twice"
                assert e.t <= t + 1e-6, "an event was released before it happened"
                laps_seen[(e.driver, e.lap)] = e
                if e.pit_in:
                    pit_marks.add((e.driver, e.lap))
            elif isinstance(e, PitEvent):
                pit_marks.add((e.driver, e.lap))
        t += 8.0
    race = bundled_race(2026, "Mexico City")
    assert len(race) > 1000
    compared = wrong_compound = 0
    for r in race.itertuples():
        e = laps_seen.get((r.acronym, r.lap))
        if e is None:
            continue
        compared += 1
        wrong_compound += e.compound != r.compound
    assert compared >= 0.97 * len(race)
    # A live stint has no end lap yet, but the in-lap must still get its old compound.
    assert wrong_compound <= 0.01 * compared
    real_stops = {(r.acronym, r.lap) for r in race.itertuples() if r.pit_in}
    assert len(real_stops - pit_marks) <= 0.05 * len(real_stops)


# -- server failures ----------------------------------------------------------------


class FakeOpenF1(http.server.BaseHTTPRequestHandler):
    laps: list = []
    fail_next = 0
    tokens = 0
    auth_seen: list = []

    def log_message(self, *a):
        pass

    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        FakeOpenF1.tokens += 1
        self._send(200, {"access_token": "tok", "expires_in": "3600", "token_type": "bearer"})

    def do_GET(self):
        FakeOpenF1.auth_seen.append(self.headers.get("Authorization"))
        if FakeOpenF1.fail_next > 0:
            FakeOpenF1.fail_next -= 1
            return self._send(503, {"error": "busy"})
        if "/v1/laps" in self.path:
            return self._send(200, FakeOpenF1.laps)
        return self._send(404, {"detail": "No results found."})


@pytest.fixture
def server():
    FakeOpenF1.laps, FakeOpenF1.fail_next, FakeOpenF1.tokens = [], 0, 0
    FakeOpenF1.auth_seen = []
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenF1)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def _wait_for(check, seconds=5.0):
    end = time.time() + seconds
    while time.time() < end:
        if check():
            return True
        time.sleep(0.02)
    return False


def test_live_feed_survives_an_outage_and_reports_it(server):
    FakeOpenF1.laps = [_raw_lap(1, T0), _raw_lap(2, T0 + 90)]
    FakeOpenF1.fail_next = 1000
    feed = OpenF1LiveFeed(info(), OpenF1Client(server, timeout=2.0), 123, poll_s=0.02,
                          clock=lambda: T0 + 100.0)
    try:
        assert _wait_for(lambda: not feed.health.ok)
        assert "503" in feed.health.message
        assert feed.poll(0.0) == []
        FakeOpenF1.fail_next = 0
        assert _wait_for(lambda: feed.health.ok, 40.0)
        got = []
        assert _wait_for(lambda: got.extend(feed.poll(0.0)) or got)
        assert got[0].lap == 1
    finally:
        feed.close()


def test_client_logs_in_once_and_sends_the_token(server):
    client = OpenF1Client(server, "user", "pass")
    assert client.get("laps", session_key=1) == []
    assert client.get("laps", session_key=1) == []
    assert FakeOpenF1.tokens == 1
    assert FakeOpenF1.auth_seen == ["Bearer tok", "Bearer tok"]


def test_client_treats_404_as_no_rows_yet(server):
    assert OpenF1Client(server).get("pit", session_key=1) == []


# -- call stability -----------------------------------------------------------------


def opt(box, offset, u, p=0.5, gap=0.0):
    return Option("x", offset, box, p, gap, -5.0, u, (), None)


def test_call_holds_against_noise_and_tracks_the_lap_not_the_offset():
    t = CallTracker(0.03)
    first, _ = t.choose([opt(21, 1, 0.60), opt(20, 0, 0.58)], lap=19)
    assert first.box_lap == 21
    # One lap later "box +1" has become "box now": the same call, despite the new label.
    kept, held = t.choose([opt(22, 1, 0.61), opt(21, 0, 0.60)], lap=20)
    assert kept.box_lap == 21 and held
    assert t.since == 19


def test_call_changes_when_something_clearly_better_appears():
    t = CallTracker(0.03)
    t.choose([opt(21, 1, 0.60), opt(20, 0, 0.58)], lap=19)
    new, held = t.choose([opt(22, 1, 0.70), opt(21, 0, 0.60)], lap=20)
    assert new.box_lap == 22 and not held
    assert t.since == 20


def test_a_stop_must_earn_the_call_against_staying_out():
    t = CallTracker(0.03)
    near_tie = [opt(20, 0, 0.61, gap=0.5), opt(30, None, 0.60, gap=0.0)]
    assert t.choose(near_tie, lap=19)[0].offset is None
    t = CallTracker(0.03)
    clear_gain = [opt(20, 0, 0.70, gap=4.0), opt(30, None, 0.60, gap=0.0)]
    assert t.choose(clear_gain, lap=19)[0].box_lap == 20
    t = CallTracker(0.03)
    same_win_but_faster = [opt(20, 0, 1.0, gap=5.0), opt(30, None, 1.0, gap=0.0)]
    assert t.choose(same_win_but_faster, lap=19)[0].box_lap == 20


def test_force_ignores_the_previous_call():
    t = CallTracker(0.03)
    t.choose([opt(21, 1, 0.60), opt(20, 0, 0.59)], lap=19)
    new, _ = t.choose([opt(20, 0, 0.61), opt(21, 1, 0.60)], lap=19, force=True)
    assert new.box_lap == 20


def test_later_planned_stop_can_drift_without_being_a_new_call():
    t = CallTracker(0.03)
    t.choose([opt(30, None, 0.60), opt(20, 0, 0.50)], lap=19)
    kept, _ = t.choose([opt(21, 0, 0.61), opt(31, None, 0.60)], lap=20)
    assert kept.offset is None and kept.box_lap == 31
    assert t.since == 19


def test_confidence_reflects_the_margin():
    a, b = opt(20, 0, 0.70), opt(21, 1, 0.50)
    assert _confidence(a, [a, b]) == "HIGH"
    a, b = opt(20, 0, 0.51), opt(21, 1, 0.50)
    assert _confidence(a, [a, b]) == "LOW"


def test_changes_counts_real_changes_only():
    assert _changes([(1, "STAY", 30), (2, "STAY", 31), (3, "STANDBY", 6), (4, "BOX", 6)]) == 1
    assert _changes([(1, "STAY", 30), (2, "BOX", 3), (3, "STAY", 30)]) == 2


# -- stop cost, compared to a neutral period ----------------------------------------


def _sim_inputs(**kw):
    recs = [LapRecord(n, "X", "MEDIUM", n, 90.0 + 0.05 * n) for n in range(2, 16)]
    model = DegradationEstimator().fit(recs, 40)
    car = CarInputs(model, 16, "MEDIUM", True, 0.2, 0.01)
    return SimInputs(lap=15, laps_total=40, gap0=2.0, pit_loss=22.0, own=car, opp=car,
                     pit_dist=[0.1, 0.9], **kw)


def test_cheap_stop_under_a_safety_car_is_reflected_in_the_simulation():
    normal = simulate_actions(_sim_inputs(), [0], 300, np.random.default_rng(1))[0]
    cheap = simulate_actions(_sim_inputs(stop_costs=[12.0]), [0], 300, np.random.default_rng(1))[0]
    assert cheap.exp_gap > normal.exp_gap + 3.0


def test_stop_cost_saving_fades_with_the_chance_the_period_ends():
    p = RacePriors(neutral_cost_ratio={"SC": 0.5}, neutral_continue={"SC": 0.8})
    costs = p.stop_costs(20.0, "SC", 3)
    assert costs[0] == pytest.approx(20.0 * (1 - 0.5 * 0.8))
    assert costs[0] < costs[1] < costs[2] < 20.0
    assert p.stop_costs(20.0, None) is None


def test_win_chance_rescales_without_changing_the_order_of_options():
    p = RacePriors(win_a=0.7, win_b=0.5, win_spread=1.0)
    chances = [p.win_chance(v, 3.0, 20) for v in (0.1, 0.3, 0.5, 0.9)]
    assert chances == sorted(chances)
    assert p.win_chance(0.5, 10.0, 20) > 0.5 > p.win_chance(0.5, -10.0, 20)
    assert 0.0 < p.win_chance(0.0, -1e6, 1) < p.win_chance(1.0, 1e6, 1) < 1.0
    assert RacePriors().win_chance(0.3, 5.0, 10) == pytest.approx(0.3)


def test_priors_cover_every_kind_of_decision():
    p = load_priors()
    assert 15.0 < p.pit_loss("Spa-Francorchamps") < p.pit_loss("Singapore") < 35.0
    assert p.pit_loss("Never Heard Of It") == p.pit_loss_default
    assert 0.0 < p.neutral_cost_ratio["SC"] < 1.0
    assert p.stint_limit("Monza", "HARD") >= p.stint_limit("Monza", "SOFT")


# -- whole engine -------------------------------------------------------------------


def test_engine_waits_then_estimates_a_rival_far_behind():
    laps = [lap("AAA", n, n * 90.0, 90.0) for n in range(1, 13)]
    laps += [lap("BBB", n, n * 90.0 + 80.0, 90.2) for n in range(1, 12)]
    feed = ScheduledFeed(info(), laps)
    engine = engine_for(feed, "AAA", "BBB", opp_wait_s=25.0, priors=RacePriors())
    engine.ingest(laps)
    t12 = 12 * 90.0
    assert engine.tick(t12 + 5.0).lap == 11      # rival has not finished lap 12 yet
    frame = engine.tick(t12 + 30.0)
    assert frame.lap == 12 and frame.projected
    engine.ingest([lap("BBB", 12, t12 + 80.0, 90.2)])
    frame = engine.tick(t12 + 81.0)
    assert frame.lap == 12 and not frame.projected


def test_rival_starting_the_next_lap_does_not_rerun_the_decision():
    laps = [lap(c, n, n * 90.0 + o, 90.0) for n in range(1, 12)
            for c, o in (("AAA", 5.0), ("BBB", 0.0))]
    feed = ScheduledFeed(info(), laps)
    engine = engine_for(feed, "AAA", "BBB", priors=RacePriors())
    run(feed, engine)
    ran = engine.decision_id
    engine.ingest([lap("BBB", 12, 12 * 90.0, 90.0)])
    engine.tick(12 * 90.0)
    assert engine.decision_id == ran
    engine.ingest([lap("AAA", 12, 12 * 90.0 + 5.0, 90.0)])
    engine.tick(12 * 90.0 + 5.0)
    assert engine.decision_id == ran + 1


def test_pace_gap_shrinks_more_when_it_rests_on_few_laps():
    from pitwall.demo import stint_laps
    recs = [LapRecord(n, "X", "MEDIUM", n, 90.0) for n in range(2, 12)]
    assert stint_laps(recs, 50) == 10
    recs[5].pitted = True
    assert stint_laps(recs, 50) == 4


def test_rival_retiring_moves_the_target_and_says_so():
    feed = synthetic_feed("rival-out")
    engine = engine_for(feed, "OWN", "RIV")
    last = feed.events[-1].t
    for e in feed.events:
        engine.ingest([e])
        engine.tick(e.t)
    engine.tick(last)
    assert engine.opp != "RIV"
    assert any("RIV has stopped running" in a.text for a in engine.alerts)


def test_call_changes_at_once_when_a_safety_car_appears_mid_lap():
    feed = synthetic_feed("safety-car")
    engine = engine_for(feed, "OWN", "RIV")
    flag = next(e for e in feed.events if isinstance(e, FlagEvent) and e.kind == "SC")
    before = next(e.t for e in reversed(feed.events) if e.t < flag.t)
    frame = run(feed, engine, until=before)
    ran = engine.decision_id
    assert frame.flag is None
    engine.ingest([flag])
    frame = engine.tick(flag.t)
    assert engine.decision_id == ran + 1
    assert frame.flag == "SC"
    assert frame.stop_cost_now < frame.pit_loss


def test_red_flag_gives_no_stop_call():
    feed = synthetic_feed("undercut")
    engine = engine_for(feed, "OWN", "RIV")
    cut = next(e.t for e in feed.events if getattr(e, "lap", 0) == 15)
    run(feed, engine, until=cut)
    engine.ingest([FlagEvent(cut + 1, "RED", "RED FLAG")])
    frame = engine.tick(cut + 1)
    assert "RED FLAG" in frame.call.headline
    assert frame.options == []


def test_a_calculation_failure_keeps_the_last_call_and_says_so(monkeypatch):
    feed = synthetic_feed("undercut")
    engine = engine_for(feed, "OWN", "RIV")
    events = iter(feed.events)
    frame = None
    for e in events:
        engine.ingest([e])
        frame = engine.tick(e.t)
        if frame.options:
            break
    held = frame.call.headline

    def boom(*a, **k):
        raise RuntimeError("bad fit")
    monkeypatch.setattr("pitwall.live_engine.decide_on_replay", boom)
    for e in events:
        engine.ingest([e])
        frame = engine.tick(e.t)
        if getattr(e, "driver", "") == "OWN" and frame.stale_reason:
            break
    assert "bad fit" in frame.stale_reason
    assert frame.call.headline == held


def test_wet_race_gives_no_call_and_says_why():
    laps = [LapEvent(n * 90.0, d, n, 90.0, "INTERMEDIATE", n, False, False, False)
            for n in range(1, 12) for d in ("AAA", "BBB")]
    feed = ScheduledFeed(info(), laps)
    frame = run(feed, engine_for(feed, "AAA", "BBB", priors=RacePriors()))
    assert frame.options == []
    assert "wet" in frame.call.why


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_every_fabricated_scenario_plays_to_the_flag(scenario):
    feed = synthetic_feed(scenario)
    engine = engine_for(feed, "OWN", "RIV", n_sims=80)
    frame = run(feed, engine)
    assert frame.finished
    assert frame.call.headline == "CHEQUERED FLAG"
    assert frame.stale_reason == ""


@pytest.mark.parametrize("race,own,opp", [
    ((2025, "Monza"), "LEC", "NOR"),
    ((2025, "Monte Carlo"), "LEC", "NOR"),      # hardly any overtaking, two stops forced
    ((2026, "Mexico City"), "LEC", "RUS"),      # safety cars and a VSC
    ((2023, "Montreal"), "LEC", "VER"),
])
def test_real_races_play_through_with_no_failures(race, own, opp):
    feed = bundled_feed(*race)
    engine = engine_for(feed, own, opp, n_sims=80)
    stale = []
    for e in feed.events:
        engine.ingest([e])
        frame = engine.tick(e.t)
        if frame.stale_reason:
            stale.append(frame.stale_reason)
        render(frame, 100, 50, color=False)
    assert stale == []
    assert engine.decision_id > 20


def test_box_laps_are_never_in_the_past():
    feed = bundled_feed(2025, "Silverstone")
    engine = engine_for(feed, "LEC", "HAM", n_sims=80)
    for e in feed.events:
        engine.ingest([e])
        frame = engine.tick(e.t)
        for o in frame.options:
            if o.box_lap is not None:
                assert o.box_lap > frame.lap


def test_scoring_picks_rivals_without_using_the_result():
    feed = bundled_feed(2025, "Monza")
    laps = [e for e in feed.events if isinstance(e, LapEvent)]
    early = [e for e in laps if e.lap <= 5]
    assert battle_pairs(laps, 3, 0) == battle_pairs(early, 3, 0)


def test_scoring_runs_on_one_race():
    got = play_pair(2025, "Spielberg", "LEC", "HAM", MODEL, load_priors(), 80)
    assert got["win"] and got["pace"]
    assert all(0.0 <= row[1] <= 1.0 for row in got["win"])
    assert dry_races((2025,))


# -- what the screen says -----------------------------------------------------------


def _frame_with(gap, stop):
    from pitwall.live_engine import CarPanel
    f = Frame(info(), 0.0)
    f.gap, f.stop_cost_now = gap, stop
    f.own, f.rival = CarPanel("AAA"), CarPanel("BBB")
    return f


def test_rival_stop_outcome_has_the_right_sign():
    # We trail by 12.3s and their stop costs 22.3s, so they come out 10s behind us.
    assert "10.0s behind us" in rival_stop_outcome(_frame_with(12.3, 22.3))
    # We trail by 30s: even after stopping they are 7.7s up the road.
    assert "7.7s ahead of us" in rival_stop_outcome(_frame_with(30.0, 22.3))
    # We lead by 5s: they come out further behind.
    assert "27.3s behind us" in rival_stop_outcome(_frame_with(-5.0, 22.3))


def test_rival_stop_outcome_is_hidden_while_a_car_is_in_the_pits():
    f = _frame_with(5.0, 22.0)
    f.rival.in_pit = True
    assert rival_stop_outcome(f) == ""


@pytest.mark.parametrize("width,height", [(60, 24), (80, 30), (100, 50), (180, 70)])
def test_no_line_is_ever_wider_than_the_screen(width, height):
    feed = bundled_feed(2026, "Mexico City")
    engine = engine_for(feed, "LEC", "RUS", n_sims=60)
    for e in feed.events:
        engine.ingest([e])
        frame = engine.tick(e.t)
        for line in render(frame, width, height, color=True, keys="keys here"):
            assert visible_len(line) <= max(60, min(width, 110))


def test_the_call_stays_on_screen_at_small_sizes():
    feed = synthetic_feed("undercut")
    frame = run(feed, engine_for(feed, "OWN", "RIV"), until=feed.events[400].t)
    lines = render(frame, 60, 24, color=False)
    text = "\n".join(lines[:9])
    assert frame.call.headline[:20] in text
    assert frame.call.radio[:20] in text


def test_fabricated_data_is_always_labelled():
    feed = synthetic_feed("undercut")
    frame = run(feed, engine_for(feed, "OWN", "RIV"), until=feed.events[300].t)
    text = "\n".join(render(frame, 100, 50, color=False))
    assert "FABRICATED" in text
    assert "not real data" in text


def test_option_table_compares_against_the_call_not_an_absolute_finish_gap():
    feed = bundled_feed(2025, "Monza")
    frame = run(feed, engine_for(feed, "LEC", "NOR"),
                until=next(e.t for e in feed.events if getattr(e, "lap", 0) == 15))
    text = "\n".join(render(frame, 100, 50, color=False))
    assert " call " in text
    assert "AT FLAG" not in text


def test_a_call_is_never_left_describing_a_finished_lap_for_long():
    feed = bundled_feed(2025, "Monza")
    engine = engine_for(feed, "NOR", "LEC", n_sims=60)  # NOR leads LEC by many seconds
    for e in feed.events:
        engine.ingest([e])
        frame = engine.tick(e.t)
        if getattr(e, "driver", "") == "NOR" and e.lap == 30:
            break
    frame = engine.tick(e.t + 3.5)
    assert frame.lap == 30 and frame.projected


def test_wear_too_small_to_see_is_not_reported_as_a_measurement_of_zero():
    from pitwall.live_engine import CarPanel
    f = _frame_with(3.0, 22.0)
    f.own = CarPanel("AAA", penalty=0.0, slope=0.0, compound="MEDIUM", age=20)
    text = "\n".join(render(f, 100, 50, color=False))
    assert "none visible yet" in text and "+0.00s" not in text


def test_replays_say_the_cars_do_not_follow_the_call():
    feed = bundled_feed(2025, "Monza")
    frame = run(feed, engine_for(feed, "LEC", "NOR", n_sims=60), until=feed.events[300].t)
    assert "not what is called here" in "\n".join(render(frame, 100, 50, color=False))


def test_helpers():
    assert lap_time(83.5) == "1:23.500"
    assert lap_time(float("nan")) == "--:--.---"
    assert visible_len(clip("\033[1mhello world\033[0m", 6)) == 6
    assert visible_len("\033[1;91mabc\033[0m") == 3


# -- command line and real terminal -------------------------------------------------


def test_clock_runs_pauses_and_jumps():
    c = Clock(1000.0, 100.0)
    time.sleep(0.02)
    assert c() > 1000.0
    c.toggle()
    frozen = c()
    time.sleep(0.01)
    assert c() == frozen
    c.toggle()
    c.jump(5000.0)
    assert c() >= 5000.0


def test_bad_arguments_exit_cleanly_with_help(capsys):
    assert main(["--race", "2025", "Atlantis"]) == 2
    assert "not bundled" in capsys.readouterr().err
    assert main(["--race", "2025", "Monza", "--own", "ZZZ", "--plain"]) == 2
    err = capsys.readouterr().err
    assert "unknown car ZZZ" in err and "LEC" in err
    assert main(["--race", "2025", "Monza", "--own", "LEC", "--opp", "LEC", "--plain"]) == 2


def test_plain_mode_prints_decisions_and_writes_the_log(tmp_path, capsys):
    log = tmp_path / "race.jsonl"
    code = main(["--fabricated", "undercut", "--plain", "--no-color", "--speed", "2000",
                 "--start-lap", "16", "--until-lap", "18", "--log", str(log)])
    assert code == 0
    out = capsys.readouterr().out
    assert "RADIO" in out and "OPTION" in out and "\033[" not in out
    rows = [json.loads(l) for l in log.read_text().splitlines()]
    assert rows and {"call", "options", "forecast", "gap"} <= rows[0].keys()


def _read_all(fd, seconds):
    out = b""
    end = time.time() + seconds
    while time.time() < end:
        if select.select([fd], [], [], 0.05)[0]:
            try:
                got = os.read(fd, 65536)
            except OSError:
                break
            if not got:
                break
            out += got
    return out


@pytest.mark.skipif(sys.platform == "win32", reason="needs a pseudo-terminal")
def test_full_screen_mode_in_a_real_terminal():
    """Runs in a pseudo-terminal: draws, obeys keys, restores the terminal on quit."""
    master, slave = pty.openpty()
    proc = subprocess.Popen(
        [sys.executable, "-m", "pitwall.live", "--fabricated", "undercut", "--speed", "300",
         "--start-lap", "15"],
        stdin=slave, stdout=slave, stderr=slave, close_fds=True,
        env={**os.environ, "COLUMNS": "100", "LINES": "50"})
    os.close(slave)
    try:
        first = _read_all(master, 3.0)
        assert b"\x1b[?1049h" in first, first[:300]      # entered the alternate screen
        assert b"PITWALL" in first and b"RADIO" in first
        os.write(master, b" ")                            # pause
        paused = _read_all(master, 0.8)
        assert b"PAUSED" in paused
        os.write(master, b" ")
        os.write(master, b"tC03\n")                       # change target
        changed = _read_all(master, 1.0)
        assert b"C03" in changed
        os.write(master, b"q")
        rest = _read_all(master, 2.0)
        assert proc.wait(timeout=5) == 0
        assert b"\x1b[?1049l" in rest and b"\x1b[?25h" in rest    # terminal given back
    finally:
        proc.kill()
        os.close(master)


def test_module_runs_as_documented():
    out = subprocess.run([sys.executable, "-m", "pitwall.live", "--list"], capture_output=True,
                         text=True, timeout=60)
    assert out.returncode == 0
    assert "2026:" in out.stdout and "Mexico City" in out.stdout and "safety-car" in out.stdout
