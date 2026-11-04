"""PITWALL live terminal: real-time strategy calls as text.

Sources:
    python -m pitwall.live                                      a real 2025 race, played back
    python -m pitwall.live --race 2025 Monza --own LEC --opp NOR
    python -m pitwall.live --fabricated safety-car              an invented race
    python -m pitwall.live --openf1 11404 --rehearse --own LEC --opp RUS
                                                                real API data, past session
    python -m pitwall.live --openf1 latest --own LEC --opp NOR  a race happening now

Live sessions need a paid OpenF1 account: set OPENF1_USERNAME and OPENF1_PASSWORD.
Muse Spark answers need MODEL_API_KEY. Without it, everything else works the same.

Keys while running: a ask Muse Spark, space pause, + faster, - slower, n next lap,
t change target, q quit.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import select
import signal
import sys
import time

from .copilot import BRIEF_QUESTION, MODEL, Answer, Copilot, MuseClient, load_api_key, panel_text
from .data import list_bundled_races
from .demo import N_SIMS
from .live_engine import Frame, LiveEngine
from .live_feed import (OpenF1Client, OpenF1LiveFeed, OpenF1Normalizer, RehearsalClient,
                        bundled_feed, open_session)
from .live_view import render, terminal_size, wrap
from .priors import load_priors
from .synthetic import SCENARIOS, synthetic_feed

KEYS = "a ask | space pause | + faster | - slower | n next lap | t target | q quit"
LIVE_KEYS = "a ask | t target | q quit"
DEFAULT_RACE = (2026, "Mexico City")
DEFAULT_CARS = ("LEC", "VER")
PLAIN_WIDTH = 100


class Clock:
    """Race time. Runs at ``speed`` and can pause or jump, for playback."""

    def __init__(self, start: float, speed: float = 1.0):
        self.speed = speed
        self.paused = False
        self._race = start
        self._wall = time.monotonic()

    def __call__(self) -> float:
        if self.paused:
            return self._race
        return self._race + (time.monotonic() - self._wall) * self.speed

    def _anchor(self) -> None:
        self._race, self._wall = self(), time.monotonic()

    def set_speed(self, speed: float) -> None:
        self._anchor()
        self.speed = max(0.25, min(speed, 2000.0))

    def toggle(self) -> None:
        self._anchor()
        self.paused = not self.paused

    def jump(self, t: float) -> None:
        self._race, self._wall = max(self(), t), time.monotonic()


class Keyboard:
    """Single-key input without echo. Does nothing when stdin is not a terminal."""

    def __init__(self):
        self.active = False
        if sys.stdin.isatty():
            try:
                import termios
                import tty

                self._termios = termios
                self._saved = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin)
                self.active = True
            except Exception:
                self.active = False

    def read(self) -> str:
        if not self.active or not select.select([sys.stdin], [], [], 0)[0]:
            return ""
        return os.read(sys.stdin.fileno(), 16).decode(errors="ignore")

    def restore(self) -> None:
        if self.active:
            self._termios.tcsetattr(sys.stdin, self._termios.TCSADRAIN, self._saved)


class Screen:
    """Full-screen redraw on the alternate buffer, restored however we exit."""

    def __enter__(self):
        sys.stdout.write("\033[?1049h\033[?25l\033[2J")
        sys.stdout.flush()
        return self

    def __exit__(self, *exc):
        sys.stdout.write("\033[?25h\033[?1049l")
        sys.stdout.flush()

    def draw(self, lines: list[str]) -> None:
        body = "".join(f"\033[{i + 1};1H{line}\033[K" for i, line in enumerate(lines))
        sys.stdout.write(body + f"\033[{len(lines) + 1};1H\033[J")
        sys.stdout.flush()


class DecisionLog:
    """One JSON line per decision, so a race can be reviewed afterwards."""

    def __init__(self, path: str | None):
        self.file = open(path, "a", buffering=1) if path else None

    def write(self, f: Frame) -> None:
        if not self.file:
            return
        self.file.write(json.dumps({
            "t": f.now, "lap": f.lap, "own": f.own.driver if f.own else None,
            "rival": f.rival.driver if f.rival else None, "flag": f.flag,
            "gap": f.gap, "pit_loss": f.pit_loss, "stop_cost_now": f.stop_cost_now,
            "call": dataclasses.asdict(f.call),
            "forecast": dataclasses.asdict(f.forecast),
            "projected": f.projected, "calc_ms": round(f.calc_ms, 2),
            "stale": f.stale_reason,
            "options": [{"label": o.label, "box_lap": o.box_lap, "win": round(o.win, 3), "simulated": round(o.p_ahead, 3),
                         "flag_gap": round(o.exp_gap, 2), "worst": round(o.worst, 2),
                         "rejoin": o.rejoin.position if o.rejoin else None}
                        for o in f.options],
        }) + "\n")

    def write_answer(self, answer: Answer) -> None:
        if self.file:
            self.file.write(json.dumps({"muse_spark": dataclasses.asdict(answer)}) + "\n")

    def close(self) -> None:
        if self.file:
            self.file.close()


class Assistant:
    """What the race loops need from Muse Spark. Does nothing without a copilot."""

    def __init__(self, copilot: Copilot | None, brief: bool, log: DecisionLog):
        self.copilot, self.brief, self.log = copilot, brief, log
        self.answer: Answer | None = None
        self._call = None

    def on_frame(self, frame: Frame) -> Answer | None:
        """Start a brief if the call changed. Returns an answer if one has just arrived."""
        if self.copilot is None:
            return None
        call = (frame.call.kind, frame.call.box_lap)
        if self.brief and frame.options and call != self._call:
            self._call = call
            self.copilot.ask(BRIEF_QUESTION, frame)
        fresh = self.copilot.poll()
        if fresh is not None:
            self.answer = fresh
            self.log.write_answer(fresh)
        return fresh

    def panel(self, frame: Frame) -> list[tuple[str, str]]:
        if self.copilot is None:
            return []
        return panel_text(self.answer, self.copilot.pending, frame.lap)


def _default_pair(drivers: list[str]) -> tuple[str, str]:
    return drivers[1], drivers[0]


def build_feed(a, clock_holder: dict):
    """Returns (feed, engine-ready race info). Raises ValueError with a usable message."""
    priors = load_priors()
    username, password = os.environ.get("OPENF1_USERNAME"), os.environ.get("OPENF1_PASSWORD")
    if a.fabricated:
        return synthetic_feed(a.fabricated, a.seed)
    if a.openf1:
        client = OpenF1Client(a.base_url, username, password)
        if a.rehearse:
            raw = {e: client.get(e, session_key=a.openf1)
                   for e in ("laps", "stints", "pit", "race_control", "drivers")}
            if not raw["laps"]:
                raise ValueError(f"OpenF1 has no laps for session {a.openf1}")
            info, meta = open_session(client, a.openf1, a.laps, priors, "REHEARSAL",
                                      raw_laps=raw["laps"], drivers=raw["drivers"])
            info.start_t = min(_ts_of(r) for r in raw["laps"] if r.get("date_start"))
            clock = Clock(info.start_t, a.speed)
            clock_holder["clock"] = clock
            norm = OpenF1Normalizer()
            norm.add(drivers=raw["drivers"])
            return OpenF1LiveFeed(info, RehearsalClient(raw, clock), a.openf1,
                                  poll_s=max(0.05, a.poll / max(a.speed, 1.0)), clock=clock,
                                  normalizer=norm)
        if not username:
            print("warning: no OPENF1_USERNAME set. OpenF1 live data needs a paid account, so a "
                  "race in progress may return nothing.", file=sys.stderr)
        info, meta = open_session(client, a.openf1, a.laps, priors, "LIVE")
        norm = OpenF1Normalizer()
        norm.add(drivers=meta["drivers"])
        return OpenF1LiveFeed(info, client, meta["session"]["session_key"], poll_s=a.poll,
                              normalizer=norm)
    year, circuit = (int(a.race[0]), a.race[1]) if a.race else DEFAULT_RACE
    if (year, circuit) not in list_bundled_races():
        raise ValueError(f"{year} {circuit} is not bundled. See --list.")
    return bundled_feed(year, circuit)


def _ts_of(row: dict) -> float:
    import pandas as pd

    return pd.Timestamp(row["date_start"]).timestamp()


def _print_answer(answer: Answer, frame: Frame, a) -> None:
    rows = panel_text(answer, False, frame.lap)
    print("\n".join(line for _, text in rows for line in wrap(text, PLAIN_WIDTH)))
    print(flush=True)


def run_plain(engine: LiveEngine, feed, clock, a, log: DecisionLog, helper: Assistant) -> int:
    """Print each new decision as a block. For pipes, logs and slow links."""
    last = None
    question = a.ask
    frame = None
    while True:
        now = clock()
        engine.ingest(feed.poll(now))
        frame = engine.tick(now, feed.health)
        marker = (frame.decision_id, frame.call.headline, frame.finished)
        if marker != last and frame.own_lap >= a.start_lap and frame.decision_id:
            last = marker
            log.write(frame)
            print("\n".join(render(frame, PLAIN_WIDTH, a.plain_height, color=a.color, speed=a.speed)))
            print(flush=True)
            if question and helper.copilot is not None and frame.options:
                helper.copilot.ask(question, frame)
                question = None
        fresh = helper.on_frame(frame)
        if fresh is not None:
            _print_answer(fresh, frame, a)
        if frame.finished or (a.until_lap and frame.own_lap >= a.until_lap) or (
                feed.exhausted and frame.own_lap and clock() > (feed.events[-1].t + 30)):
            if helper.copilot is not None:  # the race is over, so an answer can hold nothing up
                helper.copilot.wait(helper.copilot.client.timeout * 2)
                fresh = helper.copilot.poll()
                if fresh is not None:
                    helper.log.write_answer(fresh)
                    _print_answer(fresh, frame, a)
            return 0
        time.sleep(0.02 if a.speed >= 50 else 0.1)


def _handle_key(keys: str, engine: LiveEngine, feed, clock, state: dict) -> bool:
    """Apply typed characters one by one. Returns False to quit."""
    return all(_handle_char(ch, engine, feed, clock, state) for ch in keys)


def _resume(clock, state: dict) -> None:
    if state.get("resume"):
        state["resume"] = False
        if clock.paused:
            clock.toggle()


def _handle_question(ch: str, engine: LiveEngine, clock, state: dict) -> None:
    if ch in "\r\n":
        question = state["asking"].strip()
        state["asking"] = None
        if question:  # a replay stays paused until the answer arrives
            state["helper"].copilot.ask(question, engine.frame)
        else:
            _resume(clock, state)
    elif ch == "\x1b":
        state["asking"] = None
        _resume(clock, state)
    elif ch in "\x7f\b":
        state["asking"] = state["asking"][:-1]
    elif ch.isprintable():
        state["asking"] += ch


def _handle_char(ch: str, engine: LiveEngine, feed, clock, state: dict) -> bool:
    if state.get("asking") is not None:
        _handle_question(ch, engine, clock, state)
        return True
    if state.get("typing") is None and ch in "aA":
        helper = state.get("helper")
        if helper is None or helper.copilot is None:
            state["note"] = "Muse Spark is off: set MODEL_API_KEY"
        elif not engine.frame.options:
            state["note"] = "no call to ask about yet"
        else:
            state["note"] = ""
            state["asking"] = ""
            if not engine.info.live and not clock.paused:
                clock.toggle()
                state["resume"] = True
        return True
    if state.get("typing") is not None:
        if ch in "\r\n":
            code = state["typing"].strip().upper()
            state["typing"] = None
            try:
                engine.set_opponent(code)
                state["note"] = ""
            except ValueError as e:
                state["note"] = str(e)
        elif ch == "\x1b":
            state["typing"] = None
        elif ch in "\x7f\b":
            state["typing"] = state["typing"][:-1]
        elif ch.isprintable():
            state["typing"] += ch
        return True
    if ch in "qQ\x03":
        return False
    if ch in "tT":
        state["typing"] = ""
    elif not engine.info.live:
        if ch == " ":
            clock.toggle()
        elif ch in "+=":
            clock.set_speed(clock.speed * 2)
        elif ch in "-_":
            clock.set_speed(clock.speed / 2)
        elif ch in "nN":
            # To our next line crossing, where there is a known schedule to jump along.
            events = getattr(feed, "events", [])[getattr(feed, "_i", 0):]
            target = next((e.t for e in events if getattr(e, "driver", None) == engine.own), None)
            if target is not None:
                clock.jump(target)
    return True


def run_screen(engine: LiveEngine, feed, clock, a, log: DecisionLog, helper: Assistant) -> int:
    keyboard = Keyboard()
    state: dict = {"typing": None, "note": "", "asking": None, "resume": False, "helper": helper}
    question = a.ask
    last_id = 0
    last_draw = 0.0
    try:
        with Screen() as screen:
            finished_at = None
            while True:
                now = clock()
                engine.ingest(feed.poll(now))
                frame = engine.tick(now, feed.health)
                if frame.decision_id != last_id:
                    last_id = frame.decision_id
                    log.write(frame)
                    last_draw = 0.0
                if question and helper.copilot is not None and frame.options:
                    helper.copilot.ask(question, frame)
                    question = None
                if helper.on_frame(frame) is not None:
                    last_draw = 0.0
                    if state["asking"] is None and helper.copilot is not None and not helper.copilot.pending:
                        _resume(clock, state)
                key = keyboard.read()
                if key:
                    if not _handle_key(key, engine, feed, clock, state):
                        return 0
                    last_draw = 0.0
                if time.monotonic() - last_draw >= 0.25:
                    w, h = terminal_size()
                    keys = LIVE_KEYS if engine.info.live else KEYS
                    if state["asking"] is not None:
                        keys = f"ask Muse Spark, Enter to send, Esc to cancel: {state['asking']}_"
                    elif state["typing"] is not None:
                        keys = f"new target code, Enter to confirm, Esc to cancel: {state['typing']}_"
                    elif state["note"]:
                        keys = f"{state['note']} | {keys}"
                    if clock.paused:
                        keys = f"PAUSED | {keys}"
                    screen.draw(render(frame, w, h - 1, a.color,
                                       None if engine.info.live else clock.speed, keys,
                                       helper.panel(frame)))
                    last_draw = time.monotonic()
                if frame.finished or (feed.exhausted and now > feed.events[-1].t + 30):
                    finished_at = finished_at or time.monotonic()
                    if time.monotonic() - finished_at > a.hold or not keyboard.active:
                        return 0
                time.sleep(0.03)
    finally:
        keyboard.restore()


def list_races() -> None:
    by_year: dict[int, list[str]] = {}
    for year, circuit in list_bundled_races():
        by_year.setdefault(year, []).append(circuit)
    for year, circuits in by_year.items():
        print(f"{year}: " + ", ".join(circuits))
    print("\nFabricated scenarios: " + ", ".join(f"{k} ({v})" for k, v in SCENARIOS.items()))


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m pitwall.live",
                                 description="Real-time text strategy calls for one car against one rival.")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--race", nargs=2, metavar=("YEAR", "CIRCUIT"), help="bundled real race")
    src.add_argument("--fabricated", choices=sorted(SCENARIOS), help="invented race")
    src.add_argument("--openf1", metavar="SESSION", help="OpenF1 session key, or 'latest'")
    ap.add_argument("--rehearse", action="store_true", help="play a past --openf1 session through the live parser")
    ap.add_argument("--own", help="our car, three-letter code")
    ap.add_argument("--opp", help="rival car, three-letter code")
    ap.add_argument("--speed", type=float, default=20.0, help="race seconds per real second (1 = real time)")
    ap.add_argument("--start-lap", type=int, default=0, help="skip ahead to this lap")
    ap.add_argument("--until-lap", type=int, default=0, help="stop after this lap")
    ap.add_argument("--theta", type=float, default=0.30, help="rival stop chance within 3 laps that raises an alert")
    ap.add_argument("--overtake", type=float, default=1.0, help="s/lap pace advantage needed to pass on track")
    ap.add_argument("--sims", type=int, default=N_SIMS, help="simulated races per option")
    ap.add_argument("--laps", type=int, help="race distance, if not known for this circuit")
    ap.add_argument("--poll", type=float, default=8.0, help="seconds between OpenF1 polls")
    ap.add_argument("--plain", action="store_true", help="print each decision instead of a live screen")
    ap.add_argument("--plain-height", type=int, default=60, help=argparse.SUPPRESS)
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--log", help="append every decision to this JSONL file")
    ap.add_argument("--hold", type=float, default=30.0, help="seconds to keep the final screen open")
    ap.add_argument("--seed", type=int, default=11, help="seed for --fabricated")
    ap.add_argument("--brief", action="store_true", help="Muse Spark explains each new call (uses API credit)")
    ap.add_argument("--ask", metavar="QUESTION", help="ask Muse Spark this at the first decision")
    ap.add_argument("--muse-model", default=MODEL, help="Meta Model API model id")
    ap.add_argument("--list", action="store_true", help="list bundled races and exit")
    ap.add_argument("--drivers", action="store_true", help="list the cars in the chosen race and exit")
    ap.add_argument("--base-url", default="https://api.openf1.org", help=argparse.SUPPRESS)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    if a.list:
        list_races()
        return 0
    a.color = not a.no_color and "NO_COLOR" not in os.environ and (sys.stdout.isatty() or not a.plain)
    holder: dict = {}
    try:
        feed = build_feed(a, holder)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"error: could not reach OpenF1: {e}", file=sys.stderr)
        return 2
    info = feed.info
    if a.drivers:
        print(" ".join(info.drivers))
        feed.close()
        return 0
    default_own, default_opp = (("OWN", "RIV") if a.fabricated
                                else DEFAULT_CARS if not a.race and not a.openf1
                                else _default_pair(info.drivers))
    own, opp = a.own or default_own, a.opp or default_opp
    bad = [c for c in (own, opp) if c not in info.drivers]
    if bad or own == opp:
        message = f"unknown car {bad[0]}" if bad else "pick two different cars"
        print(f"error: {message}. Cars in this race: {' '.join(info.drivers)}", file=sys.stderr)
        feed.close()
        return 2
    engine = LiveEngine(info, own, opp, theta=a.theta, n_sims=a.sims, overtake_delta=a.overtake)
    clock = holder.get("clock") or (Clock(time.time()) if info.live else Clock(info.start_t, a.speed))
    if a.start_lap and hasattr(feed, "events"):
        cue = next((e.t for e in feed.events if getattr(e, "driver", None) == own
                    and e.lap >= a.start_lap - 1), None)
        if cue is not None:
            clock.jump(cue)
    log = DecisionLog(a.log)
    key = load_api_key()
    if not key and (a.brief or a.ask):
        print("warning: MODEL_API_KEY is not set, so Muse Spark is off. Calls are unaffected.",
              file=sys.stderr)
    helper = Assistant(Copilot(MuseClient(key, a.muse_model)) if key else None, a.brief, log)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        if a.plain or not sys.stdout.isatty():
            return run_plain(engine, feed, clock, a, log, helper)
        return run_screen(engine, feed, clock, a, log, helper)
    except KeyboardInterrupt:
        return 0
    finally:
        feed.close()
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())

