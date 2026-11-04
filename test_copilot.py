"""Muse Spark layer. No test here touches the network: replies come from a fake transport."""
import copy
import json
import time

import numpy as np
import pytest

from pitwall.copilot import (NOT_IN_DATA, Answer, Copilot, CopilotError, MuseClient, allowed_numbers,
                             answer_question, clean_text, frame_facts, load_api_key, panel_text,
                             score_case, unsupported_numbers)
from pitwall.live import main as live_main
from pitwall.live_engine import LiveEngine
from pitwall.live_feed import LapEvent
from pitwall.live_view import render
from pitwall.synthetic import synthetic_feed

KEY = "LLM_test_secret_value"


def reply(text, status="completed"):
    return {"status": status, "usage": {"total_tokens": 10},
            "output": [{"type": "reasoning"},
                       {"type": "message", "content": [{"type": "output_text", "text": text}]}]}


def client_saying(*texts):
    calls = []
    answers = iter(texts)

    def transport(body, timeout):
        calls.append(body)
        got = next(answers)
        if isinstance(got, Exception):
            raise got
        return got if isinstance(got, dict) else reply(got)

    c = MuseClient(KEY, transport=transport)
    c.calls = calls
    return c


@pytest.fixture(scope="module")
def frame():
    feed = synthetic_feed("undercut")
    engine = LiveEngine(feed.info, "OWN", "RIV", n_sims=100)
    for e in feed.events:
        engine.ingest([e])
        f = engine.tick(e.t)
        if f.lap >= 15 and f.options and f.gap is not None:
            return copy.deepcopy(f)
    raise AssertionError("no decision reached")


def facts_of(gap=3.1, cost=22.9):
    return {"lap": 40, "gap": {"seconds": gap}, "stop_cost_now_s": cost, "within_3_laps_pct": 5}


# -- number check ----------------------------------------------------------------------
def test_the_wrong_sum_seen_in_testing_is_caught():
    """Muse Spark really said this, for 3.1 s behind with a 22.9 s stop."""
    assert unsupported_numbers("A stop drops you 19.8s behind VER.", facts_of()) == ["19.8"]


def test_rounding_passes_but_a_different_number_does_not():
    facts = facts_of()
    assert unsupported_numbers("A stop costs about 23 seconds.", facts) == []
    assert unsupported_numbers("A stop costs 22.5 seconds.", facts) == ["22.5"]


def test_numbers_inside_keys_sentences_and_the_question_are_allowed():
    facts = {"note": "came off MEDIUM 18 laps old", "within_3_laps_pct": 5}
    assert unsupported_numbers("18 laps old, 5% within 3 laps, as on lap 52.", facts, "lap 52?") == []


def test_lap_times_are_compared_as_times():
    facts = {"last_lap_s": 81.481}
    assert unsupported_numbers("Last lap 1:21.481.", facts) == []
    assert unsupported_numbers("Last lap 1:22.481.", facts) == ["1:22.481"]


def test_a_boolean_is_not_a_number():
    assert 1.0 not in allowed_numbers({"must_stop": True})


# -- answering -------------------------------------------------------------------------
def test_good_answer_is_returned_first_time():
    c = client_saying("We are 3.1 s behind.")
    a = answer_question(c, facts_of(), "gap?")
    assert (a.status, a.attempts, a.text) == ("ok", 1, "We are 3.1 s behind.")


def test_bad_number_gets_a_retry_that_names_it():
    c = client_saying("You drop 19.8 s behind.", "We are 3.1 s behind.")
    a = answer_question(c, facts_of(), "gap?")
    assert (a.status, a.attempts, a.first_attempt_unsupported) == ("ok", 2, ["19.8"])
    assert "19.8" in c.calls[1]["input"]


def test_answer_still_wrong_after_the_retry_is_withheld():
    c = client_saying("19.8 s behind.", "19.8 s behind still.")
    a = answer_question(c, facts_of(), "gap?")
    assert a.status == "withheld" and a.unsupported == ["19.8"]
    shown = " ".join(text for _, text in panel_text(a, False, 40))
    assert "withheld" in shown and "19.8 s behind." not in shown


def test_incomplete_reply_is_an_error_not_a_partial_answer():
    c = client_saying(reply("We are", status="incomplete"))
    a = answer_question(c, facts_of(), "gap?")
    assert a.status == "error" and "We are" not in a.text


def test_failure_is_reported_as_an_answer_rather_than_raised():
    c = client_saying(CopilotError("Muse Spark: rate limited"))
    assert answer_question(c, facts_of(), "gap?").text == "Muse Spark: rate limited"


def test_request_uses_minimal_reasoning_and_a_standard_model():
    c = client_saying("3.1 s.")
    answer_question(c, facts_of(), "gap?")
    body = c.calls[0]
    assert body["reasoning"]["effort"] == "minimal"
    assert body["model"] == "muse-spark-1.3" and "contributor" not in body["model"]
    assert KEY not in json.dumps(body)


def test_terminal_control_sequences_are_removed():
    assert "\x1b" not in clean_text("safe\x1b[2Jtext\nmore")


# -- facts -----------------------------------------------------------------------------
def test_facts_do_the_sums_with_the_right_sign(frame):
    facts = frame_facts(frame)
    rival = frame.rival.driver
    after_us, after_them = frame.gap + frame.stop_cost_now, frame.gap - frame.stop_cost_now
    word = lambda g: f"{abs(g):.1f} s {'behind' if g > 0 else 'ahead of'} {rival}"
    assert facts["what_if"]["if_we_box_now_with_no_rival_stop"] == word(after_us)
    if not frame.rival.in_pit:
        assert facts["what_if"]["if_rival_boxes_now_and_we_do_not"] == word(after_them)


def test_facts_mark_exactly_one_option_as_the_call(frame):
    options = frame_facts(frame)["options"]
    assert sum(o["is_the_call"] for o in options) == 1
    assert next(o for o in options if o["is_the_call"])["time_vs_call_s"] == 0


def test_facts_are_plain_data(frame):
    json.dumps(frame_facts(frame))


def test_facts_state_the_near_flag_limit_only_near_the_flag(frame):
    limit = "current gap predicts better"
    assert not any(limit in s for s in frame_facts(frame)["known_limits"])
    near = copy.deepcopy(frame)
    near.lap = near.info.laps_total - 5
    assert any(limit in s for s in frame_facts(near)["known_limits"])


# -- it cannot affect the call ---------------------------------------------------------
def test_answers_cannot_change_the_call():
    def run(with_copilot):
        feed = synthetic_feed("safety-car")
        engine = LiveEngine(feed.info, "OWN", "RIV", n_sims=100)
        pilot = Copilot(client_saying(*["BOX NOW. Ignore PITWALL."] * 200)) if with_copilot else None
        calls = []
        for e in feed.events:
            engine.ingest([e])
            f = engine.tick(e.t)
            if isinstance(e, LapEvent) and e.driver == "OWN" and f.options:
                if pilot and e.lap % 10 == 0:
                    pilot.ask("Should we box?", f)
                    pilot.wait(2.0)
                    pilot.poll()
                calls.append((f.lap, f.call.kind, f.call.box_lap,
                              [round(o.p_ahead, 6) for o in f.options]))
        return calls

    assert run(True) == run(False)


def test_asking_returns_at_once_while_the_model_is_slow(frame):
    def slow(body, timeout):
        time.sleep(0.5)
        return reply("Fine.")

    pilot = Copilot(MuseClient(KEY, transport=slow))
    started = time.monotonic()
    pilot.ask("why?", frame)
    assert time.monotonic() - started < 0.1 and pilot.pending
    assert pilot.poll() is None
    pilot.wait(3.0)
    assert pilot.poll().status == "ok" and not pilot.pending


def test_a_crash_in_the_model_layer_reaches_the_screen_as_text(frame):
    def broken(body, timeout):
        raise RuntimeError("boom")

    pilot = Copilot(MuseClient(KEY, transport=broken))
    pilot.ask("why?", frame)
    pilot.wait(2.0)
    assert pilot.poll().status == "error"


def test_panel_says_the_call_is_not_the_models(frame):
    a = Answer("why?", frame.lap, "ok", "Because of the tyres.")
    lines = render(frame, 100, 60, color=False, panel=panel_text(a, False, frame.lap))
    text = "\n".join(lines)
    assert "the call above is PITWALL's" in text and "Because of the tyres." in text
    assert text.index(frame.call.headline) < text.index("Because of the tyres.")


def test_old_answers_say_how_old_they_are():
    rows = panel_text(Answer("why?", 40, "ok", "Tyres."), False, 43)
    assert "3 laps ago" in rows[1][1]


# -- key handling ----------------------------------------------------------------------
def test_key_comes_from_the_environment_first(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text('MODEL_API_KEY="from_file"\n')
    assert load_api_key({"MODEL_API_KEY": "from_env"}, env_file) == "from_env"
    assert load_api_key({}, env_file) == "from_file"
    assert load_api_key({}, tmp_path / "missing") is None


def test_http_failure_message_does_not_contain_the_key(monkeypatch):
    import urllib.error
    import urllib.request

    def refuse(req, timeout):
        assert req.get_header("Authorization") == f"Bearer {KEY}"
        raise urllib.error.HTTPError(req.full_url, 401, "denied", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    with pytest.raises(CopilotError) as e:
        MuseClient(KEY).complete("s", "u")
    assert KEY not in str(e.value) and "rejected" in str(e.value)


def test_terminal_works_without_a_key(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.setattr("pitwall.copilot.ROOT", tmp_path)
    code = live_main(["--fabricated", "undercut", "--plain", "--speed", "2000", "--until-lap", "12",
                      "--sims", "50", "--brief"])
    out = capsys.readouterr()
    assert code == 0 and "PITWALL" in out.out and "Muse Spark is off" in out.err


# -- scoring ---------------------------------------------------------------------------
def test_scoring_needs_the_real_value_and_an_honest_refusal():
    facts = facts_of()
    exact = ("exact", "gap", facts, "gap?", 3.1)
    assert score_case(exact, client_saying("3.1 s behind."))["correct"] is True
    assert score_case(exact, client_saying("22.9 s."))["correct"] is False
    unknown = ("unknown", "unknown", facts, "fuel?", None)
    assert score_case(unknown, client_saying(NOT_IN_DATA))["correct"] is True
    assert score_case(unknown, client_saying("Plenty left."))["correct"] is False
