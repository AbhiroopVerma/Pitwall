"""Muse Spark on the pit wall: ask about the race and get an answer from PITWALL's numbers.

The engine alone makes the call. Muse Spark explains it, and nothing here can change it:
facts flow one way, from a finished Frame to the model. Three rules keep a wrong answer
off the radio:

- **Facts only.** The model sees the current facts as JSON, with the sums already done,
  and must say NOT IN DATA for anything else.
- **Numbers are checked.** An answer quoting a number that is not in those facts gets one
  retry and is then withheld. In testing the model turned "3.1 s behind, a stop costs
  22.9 s" into "19.8 s behind" when the right answer is 26.0 s.
- **Never in the way.** Calls run on a background thread, take seconds, and may fail. The
  terminal carries on without them, and without a key at all.

    export MODEL_API_KEY="..."            # or put it in a git-ignored .env file
    python -m pitwall.live --fabricated safety-car --brief    # press a to ask a question
    python -m pitwall.copilot --eval                          # score it on 2026 races
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

BASE_URL = "https://api.meta.ai/v1/responses"
# Standard mode. The cheaper "contributor" models may use content for training.
MODEL = "muse-spark-1.3"
# At "low" and above, reasoning used up 600 output tokens and returned no answer.
EFFORT = "minimal"
MAX_OUTPUT_TOKENS = 1500
TIMEOUT_S = 30.0
ATTEMPTS = 2
NOT_IN_DATA = "NOT IN DATA"
BRIEF_QUESTION = "In one or two sentences, why is this the call right now?"
RESULTS = Path(__file__).resolve().parent / "assets" / "copilot_eval.json"
ROOT = Path(__file__).resolve().parents[1]

SYSTEM = f"""You support a Formula 1 race engineer during a race. PITWALL, a statistical
engine, has already made the strategy call. You explain it. You never make or change a call.

Rules:
- Use only the JSON facts supplied. They describe this moment of the race.
- Quote numbers exactly as given. Never do arithmetic or estimate. The what_if section has
  the sums already done, so use it.
- A positive time_vs_call_s means that option finishes that much better than the call.
- If the facts do not answer the question, reply with exactly: {NOT_IN_DATA}
- If known_limits covers the question, say so.
- Answer in at most two short sentences that can be read aloud. No preamble, no markdown."""

_NUMBER = re.compile(r"\d+:\d{2}(?:\.\d+)?|\d+(?:\.\d+)?")
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]+")


class CopilotError(Exception):
    """A failure safe to show on screen. Never contains the key."""


def load_api_key(env: dict | None = None, dotenv: Path | None = None) -> str | None:
    """MODEL_API_KEY from the environment, or from a .env file in the project."""
    env = os.environ if env is None else env
    if env.get("MODEL_API_KEY"):
        return env["MODEL_API_KEY"].strip()
    path = dotenv if dotenv is not None else ROOT / ".env"
    try:
        for line in path.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "MODEL_API_KEY" and value.strip():
                return value.strip().strip("\"'")
    except OSError:
        pass
    return None


# -- facts ---------------------------------------------------------------------------
def _round(v, digits: int = 1):
    return None if v is None else round(float(v), digits)


def _pct(v) -> int:
    return round(float(v) * 100)


def _car(c) -> dict | None:
    if c is None:
        return None
    return {"car": c.driver, "position": c.position, "tyre": c.compound, "tyre_age_laps": c.age,
            "stops_made": c.stops, "must_still_stop": c.must_stop, "in_pit": c.in_pit,
            "running": c.running, "last_lap_s": _round(c.last_lap, 3), "pace_s": _round(c.pace, 3),
            "lost_to_wear_s": _round(c.penalty, 2), "wear_per_lap_s": _round(c.slope, 3)}


def _option(o, reference) -> dict:
    name = (f"box end of lap {o.box_lap}" if o.offset is not None
            else f"stay out, stop around lap {o.box_lap}" if o.box_lap else "stay out to the flag")
    row = {"option": name, "is_the_call": o is reference, "win_pct": _pct(o.win),
           "planned_stops": len(o.stops), "time_vs_call_s": _round(o.exp_gap - reference.exp_gap)}
    if o.rejoin is not None:
        row["rejoin_position"] = o.rejoin.position
        row["traffic_on_rejoin"] = (f"{o.rejoin.road_ahead} {o.rejoin.road_gap:.1f} s ahead"
                                    if o.rejoin.traffic else "clear")
    return row


def _what_if(frame) -> dict:
    """The sums the model must not do itself. Gap + means we trail."""
    if frame.gap is None or not frame.rival or not frame.own:
        return {}

    def words(gap: float) -> str:
        return (f"{gap:.1f} s behind {frame.rival.driver}" if gap > 0
                else f"{-gap:.1f} s ahead of {frame.rival.driver}")

    out = {"if_we_box_now_with_no_rival_stop": words(frame.gap + frame.stop_cost_now)}
    if not frame.rival.in_pit:
        out["if_rival_boxes_now_and_we_do_not"] = words(frame.gap - frame.stop_cost_now)
    return out


def frame_facts(frame) -> dict:
    """Everything the model may use, as plain data, so it is safe to pass between threads."""
    info = frame.info
    reference = next((o for o in frame.options if o.box_lap == frame.call.box_lap),
                     frame.options[0] if frame.options else None)
    fc = frame.forecast
    limits = ["Absolute finish gaps are unreliable, so only time between options is given.",
              "Most rival stops come with no warning: 35-43% were flagged on unseen races.",
              "Rival stop readings of 40-50% came true 25% of the time on unseen races.",
              "The race simulation has two cars and does not simulate future safety cars."]
    laps_left = info.laps_total - frame.lap
    if laps_left <= 10:
        limits.append("With 10 laps or fewer left, the current gap predicts better than win chance.")
    if frame.projected:
        limits.append("The rival's latest lap is estimated, not yet measured.")
    if info.source != "LIVE":
        limits.append(f"This is {info.source}, not a live race.")
    return {
        "race": info.label, "source": info.source, "lap": frame.lap,
        "laps_total": info.laps_total, "laps_left": laps_left, "flag": frame.flag or "GREEN",
        "us": _car(frame.own), "rival": _car(frame.rival),
        "gap": None if frame.gap is None or not frame.rival else {
            "words": f"{abs(frame.gap):.1f} s {'behind' if frame.gap > 0 else 'ahead of'} {frame.rival.driver}",
            "seconds": _round(abs(frame.gap))},
        "our_pace_vs_rival": None if frame.pace_delta is None else (
            f"{abs(frame.pace_delta):.2f} s/lap {'faster' if frame.pace_delta < 0 else 'slower'}"),
        "stop_cost_now_s": _round(frame.stop_cost_now), "stop_cost_green_s": _round(frame.pit_loss),
        "what_if": _what_if(frame),
        "call": {"kind": frame.call.kind, "headline": frame.call.headline,
                 "radio": frame.call.radio, "reason": frame.call.why,
                 "box_lap": frame.call.box_lap, "confidence": frame.call.confidence,
                 "since_lap": frame.call.since_lap},
        "options": [_option(o, reference) for o in frame.options],
        "rival_stop_forecast": {"this_lap_pct": _pct(fc.p_this_lap), "within_3_laps_pct": _pct(fc.p3),
                                "within_5_laps_pct": _pct(fc.p5), "likely_lap": fc.likely_lap},
        "tyre_note": frame.tyre_note,
        "recent_events": [a.text for a in frame.alerts[-5:]],
        "known_limits": limits,
    }


# -- checking ------------------------------------------------------------------------
def _to_float(token: str) -> float:
    if ":" in token:
        minutes, seconds = token.split(":")
        return int(minutes) * 60 + float(seconds)
    return float(token)


def numbers_in(text: str) -> list[str]:
    return _NUMBER.findall(text)


def allowed_numbers(value) -> set[float]:
    """Every number anywhere in the facts, in keys, values and sentences."""
    found: set[float] = set()
    if isinstance(value, dict):
        for k, v in value.items():
            found |= allowed_numbers(str(k)) | allowed_numbers(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            found |= allowed_numbers(v)
    elif isinstance(value, bool) or value is None:
        pass
    elif isinstance(value, (int, float)):
        found.add(abs(float(value)))
        found |= allowed_numbers(str(value))
    else:
        found |= {_to_float(t) for t in numbers_in(str(value))}
    return found


def _matches(token: str, allowed: set[float]) -> bool:
    """Rounding is fine: 23 s may quote 22.9 s, but 22.5 s may not."""
    v = _to_float(token)
    decimals = len(token.rpartition(".")[2]) if "." in token else 0
    tolerance = 0.5 * 10 ** -decimals + 1e-9
    return any(abs(v - a) <= tolerance for a in allowed)


def unsupported_numbers(answer: str, facts: dict, question: str = "") -> list[str]:
    """Numbers in the answer found neither in the facts nor in the question."""
    allowed = allowed_numbers(facts) | allowed_numbers(question)
    return [t for t in numbers_in(answer) if not _matches(t, allowed)]


def clean_text(text: str) -> str:
    """Model output goes to a terminal, so it must not carry control sequences."""
    return " ".join(_CONTROL.sub(" ", text).split())


# -- model ---------------------------------------------------------------------------
Transport = Callable[[dict, float], dict]


class MuseClient:
    """Responses API over the standard library, so the terminal gains no dependency."""

    def __init__(self, key: str, model: str = MODEL, timeout: float = TIMEOUT_S,
                 transport: Transport | None = None):
        self._key, self.model, self.timeout = key, model, timeout
        self._transport = transport or self._post

    def _post(self, body: dict, timeout: float) -> dict:
        req = urllib.request.Request(BASE_URL, json.dumps(body).encode(), {
            "Authorization": f"Bearer {self._key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            words = {401: "API key rejected", 402: "API credit problem", 404: "model not found",
                     429: "rate limited"}
            raise CopilotError(f"Muse Spark: {words.get(e.code, f'HTTP {e.code}')}") from None
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            raise CopilotError(f"Muse Spark unreachable: {type(e).__name__}") from None

    def complete(self, instructions: str, user: str) -> tuple[str, dict]:
        """(answer text, usage). Raises CopilotError if there is no complete answer."""
        reply = self._transport({"model": self.model, "instructions": instructions, "input": user,
                                 "reasoning": {"effort": EFFORT},
                                 "max_output_tokens": MAX_OUTPUT_TOKENS}, self.timeout)
        text = "".join(part.get("text", "") for item in reply.get("output", [])
                       if item.get("type") == "message" for part in item.get("content", [])
                       if part.get("type") == "output_text")
        if reply.get("status") != "completed" or not text.strip():
            raise CopilotError(f"Muse Spark returned no complete answer ({reply.get('status')})")
        return clean_text(text), reply.get("usage") or {}


@dataclass
class Answer:
    question: str
    lap: int
    status: str             # ok | withheld | error
    text: str = ""
    unsupported: list[str] = field(default_factory=list)
    attempts: int = 0
    first_attempt_unsupported: list[str] = field(default_factory=list)
    seconds: float = 0.0
    tokens: int = 0


def answer_question(client: MuseClient, facts: dict, question: str) -> Answer:
    """Ask, check the numbers, retry once, and withhold anything still unsupported."""
    started = time.monotonic()
    answer = Answer(question, int(facts.get("lap") or 0), "error")
    payload = f"FACTS:\n{json.dumps(facts, indent=0)}\n\nQUESTION FROM THE ENGINEER:\n{question}"
    prompt = payload
    try:
        for attempt in range(1, ATTEMPTS + 1):
            answer.attempts = attempt
            text, usage = client.complete(SYSTEM, prompt)
            answer.tokens += int(usage.get("total_tokens") or 0)
            bad = unsupported_numbers(text, facts, question)
            if attempt == 1:
                answer.first_attempt_unsupported = bad
            answer.text, answer.unsupported = text, bad
            if not bad:
                answer.status = "ok"
                return answer
            prompt = (f"{payload}\n\nYour last answer used numbers not in the facts: "
                      f"{', '.join(bad)}. Answer again using only numbers that appear in the facts.")
        answer.status = "withheld"
    except CopilotError as e:
        answer.status, answer.text = "error", str(e)
    finally:
        answer.seconds = time.monotonic() - started
    return answer


class Copilot:
    """One background worker. A newer question replaces one that has not started."""

    def __init__(self, client: MuseClient):
        self.client = client
        self._jobs: queue.Queue = queue.Queue(maxsize=1)
        self._done: queue.Queue = queue.Queue()
        self._busy = threading.Event()
        self._thread = threading.Thread(target=self._work, daemon=True, name="pitwall-muse")
        self._thread.start()

    def ask(self, question: str, frame) -> None:
        """Takes the facts now, on the caller's thread, because the Frame keeps changing."""
        job = (frame_facts(frame), question)
        try:
            self._jobs.get_nowait()
        except queue.Empty:
            pass
        self._busy.set()
        self._jobs.put_nowait(job)

    @property
    def pending(self) -> bool:
        return self._busy.is_set()

    def poll(self) -> Answer | None:
        try:
            return self._done.get_nowait()
        except queue.Empty:
            return None

    def wait(self, timeout: float) -> None:
        end = time.monotonic() + timeout
        while self.pending and time.monotonic() < end:
            time.sleep(0.02)

    def _work(self) -> None:
        while True:
            facts, question = self._jobs.get()
            try:
                self._done.put(answer_question(self.client, facts, question))
            except Exception as e:  # a bug here must never reach the race screen
                self._done.put(Answer(question, int(facts.get("lap") or 0), "error",
                                      f"Muse Spark failed: {type(e).__name__}"))
            finally:
                if self._jobs.empty():
                    self._busy.clear()


def panel_text(answer: Answer | None, pending: bool, current_lap: int) -> list[tuple[str, str]]:
    """(style, text) rows for the screen, labelled so an answer is never taken for the call."""
    title = "MUSE SPARK  explains only, the call above is PITWALL's"
    if answer is None:
        return [("dim", title), ("dim", "Asking...")] if pending else []
    rows = [("dim", title + ("  (asking again...)" if pending else ""))]
    age = current_lap - answer.lap
    when = f"lap {answer.lap}" + (f", {age} lap{'s' if age != 1 else ''} ago" if age > 0 else "")
    rows.append(("dim", f"Q ({when}): {clean_text(answer.question)}"))
    if answer.status == "ok":
        rows.append(("bold", answer.text))
    elif answer.status == "withheld":
        rows.append(("t_red", "Answer withheld: it quoted numbers PITWALL did not supply "
                              f"({', '.join(answer.unsupported)}). Do not use it."))
    else:
        rows.append(("t_yellow", answer.text))
    return rows


# -- scoring -------------------------------------------------------------------------
ASK_EXACT = {
    "gap": ("What is the gap to the rival right now, in seconds?", lambda f: f["gap"]["seconds"]),
    "stop_cost": ("What does a stop cost us right now, in seconds?", lambda f: f["stop_cost_now_s"]),
    "win": ("What is the win chance of the current call, as a percentage?",
            lambda f: next(o["win_pct"] for o in f["options"] if o["is_the_call"])),
    "rival_stop": ("What is the chance the rival stops within 3 laps, as a percentage?",
                   lambda f: f["rival_stop_forecast"]["within_3_laps_pct"]),
}
ASK_WHY = "Why is this the call, and why not the next best option?"
ASK_UNKNOWN = ("How many kilograms of fuel do we have left?",
               "What did the driver last say on the radio about tyre temperatures?")


def _sample_frames(n_races: int, laps_each: int, sims: int) -> list[dict]:
    """Facts from races the engine never trained on, at set points in each race."""
    from .live_engine import LiveEngine
    from .live_eval import battle_pairs, dry_races
    from .live_feed import LapEvent, bundled_feed

    races = dry_races((2026,))
    step = max(len(races) // n_races, 1)
    out = []
    for year, circuit in races[::step][:n_races]:
        feed = bundled_feed(year, circuit)
        laps = [e for e in feed.events if isinstance(e, LapEvent)]
        own, opp = battle_pairs(laps, 1, 0)[0]
        engine = LiveEngine(feed.info, own, opp, n_sims=sims)
        total = feed.info.laps_total
        targets = [round(total * (i + 1) / (laps_each + 1)) for i in range(laps_each)]
        for e in feed.events:
            engine.ingest([e])
            frame = engine.tick(e.t)
            if targets and frame.lap >= targets[0] and frame.options and frame.gap is not None:
                targets.pop(0)
                out.append(frame_facts(frame))
    return out


def _cases(facts_list: list[dict]) -> list[tuple]:
    cases = []
    for i, facts in enumerate(facts_list):
        for name, (question, truth) in ASK_EXACT.items():
            cases.append(("exact", name, facts, question, truth(facts)))
        cases.append(("explain", "why", facts, ASK_WHY, None))
        cases.append(("unknown", "unknown", facts, ASK_UNKNOWN[i % len(ASK_UNKNOWN)], None))
    return cases


def score_case(case: tuple, client: MuseClient) -> dict:
    group, name, facts, question, truth = case
    a = answer_question(client, facts, question)
    row = {"group": group, "name": name, "status": a.status, "attempts": a.attempts,
           "seconds": round(a.seconds, 2), "tokens": a.tokens, "text": a.text,
           "first_attempt_unsupported": a.first_attempt_unsupported, "question": question,
           "truth": truth, "lap": facts.get("lap"), "race": facts.get("race")}
    if group == "exact":
        row["correct"] = a.status == "ok" and any(_matches(t, {abs(float(truth))})
                                                  for t in numbers_in(a.text))
    elif group == "unknown":
        row["correct"] = a.status == "ok" and NOT_IN_DATA in a.text.upper()
    return row


def evaluate(client: MuseClient, n_races: int = 3, laps_each: int = 3, sims: int = 300,
             workers: int = 4) -> dict:
    facts_list = _sample_frames(n_races, laps_each, sims)
    with ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(lambda c: score_case(c, client), _cases(facts_list)))
    ok = [r for r in rows if r["status"] == "ok"]
    seconds = sorted(r["seconds"] for r in rows)

    def share(group: str) -> dict:
        got = [r for r in rows if r["group"] == group]
        return {"n": len(got), "correct": sum(bool(r.get("correct")) for r in got)}

    return {"model": client.model, "effort": EFFORT, "decisions": len(facts_list),
            "calls_asked": len(rows), "answered": len(ok),
            "withheld": sum(r["status"] == "withheld" for r in rows),
            "errors": sum(r["status"] == "error" for r in rows),
            "first_attempt_with_unsupported_numbers": sum(bool(r["first_attempt_unsupported"]) for r in rows),
            "exact_values": share("exact"), "unanswerable_refused": share("unknown"),
            "explanations_answered": sum(r["group"] == "explain" and r["status"] == "ok" for r in rows),
            "explanations_asked": sum(r["group"] == "explain" for r in rows),
            "seconds": {"median": seconds[len(seconds) // 2], "max": seconds[-1]},
            "total_tokens": sum(r["tokens"] for r in rows), "rows": rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--eval", action="store_true", help="score answers on 2026 races")
    ap.add_argument("--races", type=int, default=3)
    ap.add_argument("--laps", type=int, default=3)
    a = ap.parse_args(argv)
    key = load_api_key()
    if not key:
        print("error: set MODEL_API_KEY, or put it in a .env file", flush=True)
        return 2
    if not a.eval:
        ap.print_help()
        return 0
    result = evaluate(MuseClient(key), a.races, a.laps)
    RESULTS.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
