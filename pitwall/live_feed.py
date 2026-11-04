"""Where the live terminal gets its data. Every source yields the same events.

    ScheduledFeed   a known race played on a clock (bundled real race, or fabricated)
    OpenF1LiveFeed  polls OpenF1 during a session (paid real-time access)
                    or rehearses a past session through the very same parser

An event only becomes visible at the moment it could really have been known, so a
replay cannot see the future and the engine cannot tell replay from live.

OpenF1 details that shaped the parser, checked against the 2026 Mexico race:
- ``laps`` has a row as soon as a lap starts, with ``lap_duration`` still empty.
- ``pit.lap_number`` is usually the in-lap, but was the out-lap for 9 of 67 stops.
- A lap's end time is the next lap's ``date_start``, or start plus duration.
"""
from __future__ import annotations

import json
import queue
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .openf1 import _neutral_ended, _neutral_kind


@dataclass(frozen=True)
class LapEvent:
    """A car crossed the line and finished a lap."""
    t: float
    driver: str
    lap: int
    lap_time: float
    compound: str
    tyre_age: int | None
    pit_in: bool
    pit_out: bool
    neutral: bool


@dataclass(frozen=True)
class PitEvent:
    """Late news: a lap already reported turns out to be an in-lap."""
    t: float
    driver: str
    lap: int


@dataclass(frozen=True)
class FlagEvent:
    """The race went neutral (kind = SC, VSC or RED) or back to green (kind = None)."""
    t: float
    kind: str | None
    message: str


Event = LapEvent | PitEvent | FlagEvent


@dataclass
class RaceInfo:
    label: str
    source: str              # REAL REPLAY | FABRICATED | LIVE | REHEARSAL
    year: int | None
    circuit: str | None
    laps_total: int
    drivers: list[str]
    start_t: float
    live: bool = False


class FeedHealth:
    """What the feed knows about its own state, shown on screen."""
    ok: bool = True
    last_success_t: float | None = None
    message: str = ""


class ScheduledFeed:
    """Events known in advance and released as the clock reaches them."""

    def __init__(self, info: RaceInfo, events: list[Event]):
        self.info = info
        self.events = sorted(events, key=lambda e: e.t)
        self._i = 0
        self.health = FeedHealth()

    def poll(self, now: float) -> list[Event]:
        out = []
        while self._i < len(self.events) and self.events[self._i].t <= now:
            out.append(self.events[self._i])
            self._i += 1
        self.health.last_success_t = now
        return out

    def next_time(self) -> float | None:
        return self.events[self._i].t if self._i < len(self.events) else None

    @property
    def exhausted(self) -> bool:
        return self._i >= len(self.events)

    def close(self) -> None:
        pass


def _lap_events(race: pd.DataFrame) -> list[LapEvent]:
    """Per-lap rows in the bundled format -> line-crossing events."""
    events = []
    for driver, car in race.groupby("acronym"):
        car = car.sort_values("lap")
        rows = list(car.itertuples())
        for i, r in enumerate(rows):
            nxt = rows[i + 1] if i + 1 < len(rows) and rows[i + 1].lap == r.lap + 1 else None
            if nxt is not None and np.isfinite(nxt.start):
                t = float(nxt.start)
            elif np.isfinite(r.start) and np.isfinite(r.time):
                t = float(r.start) + float(r.time)
            else:
                continue  # the car never finished this lap
            age = int(round(r.age)) if np.isfinite(r.age) else None
            events.append(LapEvent(t, str(driver), int(r.lap), float(r.time), str(r.compound),
                                   age, bool(r.pit_in), bool(r.pit_out), bool(r.neutral)))
    return events


def _flag_events(periods: list[tuple[float, float, str]]) -> list[FlagEvent]:
    names = {"SC": "SAFETY CAR DEPLOYED", "VSC": "VIRTUAL SAFETY CAR DEPLOYED", "RED": "RED FLAG"}
    events = []
    for i, (a, b, kind) in enumerate(periods):
        events.append(FlagEvent(a, kind, names.get(kind, kind)))
        carries_on = i + 1 < len(periods) and periods[i + 1][0] <= b
        if not carries_on:
            events.append(FlagEvent(b, None, "TRACK CLEAR"))
    return events


def feed_from_frame(race: pd.DataFrame, periods: list[tuple[float, float, str]], label: str,
                    source: str, year: int | None, circuit: str | None) -> ScheduledFeed:
    laps = _lap_events(race)
    first = race[race["lap"] == race["lap"].min()]
    start = float(first["start"].min())
    order = first.sort_values("pos")["acronym"].astype(str).tolist()
    info = RaceInfo(label, source, year, circuit, int(race["total_laps"].iloc[0]), order, start)
    return ScheduledFeed(info, [*laps, *_flag_events(periods)])


def bundled_feed(year: int, circuit: str) -> ScheduledFeed:
    from .data import bundled_race
    from .priors import neutral_periods_of

    race = bundled_race(year, circuit)
    if race.empty:
        raise ValueError(f"race not bundled: {year} {circuit}")
    name = str(race["race"].iloc[0])
    return feed_from_frame(race, neutral_periods_of(year, circuit), f"{year} {name} ({circuit})",
                           "REAL REPLAY", year, circuit)


# ---------------------------------------------------------------------------
# OpenF1: rows -> events


def _ts(value) -> float | None:
    return pd.Timestamp(value).timestamp() if value else None


class OpenF1Normalizer:
    """Turns whatever OpenF1 has returned so far into new events, at most once each.

    Safe to feed the same rows repeatedly and in any order, as polling will.
    """

    def __init__(self, grace_s: float = 8.0):
        self.grace_s = grace_s
        self.names: dict[int, str] = {}
        self._laps: dict[int, dict[int, dict]] = {}
        self._stints: dict[int, dict[int, dict]] = {}
        self._pit_rows: set[tuple[int, int]] = set()
        self._sent_laps: dict[tuple[int, int], bool] = {}
        self._control: dict[tuple, dict] = {}
        self._flag_seen = 0
        self._flag_kind: str | None = None
        self._flag_since: float | None = None
        self._neutral: list[tuple[float, float]] = []

    # -- intake --------------------------------------------------------------------
    def add(self, laps=(), stints=(), pits=(), control=(), drivers=()) -> None:
        for r in drivers:
            if r.get("name_acronym"):
                self.names[r["driver_number"]] = r["name_acronym"]
        for r in laps:
            self._laps.setdefault(r["driver_number"], {})[r["lap_number"]] = r
        for r in stints:
            self._stints.setdefault(r["driver_number"], {})[r["stint_number"]] = r
        for r in pits:
            if r.get("lap_number") is not None:
                self._pit_rows.add((r["driver_number"], r["lap_number"]))
        for r in control:
            self._control[(r.get("date"), r.get("message"))] = r

    # -- lookups -------------------------------------------------------------------
    def _name(self, number: int) -> str:
        return self.names.get(number, str(number))

    def _tyre(self, number: int, lap: int) -> tuple[str, int | None]:
        best = None
        for s in self._stints.get(number, {}).values():
            start, end = s.get("lap_start"), s.get("lap_end")
            if start is None or start > lap:
                continue
            if end is not None and lap <= end:
                best = s
                break
            if best is None or start > best["lap_start"]:
                best = s
        if best is None:
            return "UNKNOWN", None
        age0 = best.get("tyre_age_at_start")
        age = None if age0 is None else age0 + lap - best["lap_start"] + 1
        return best.get("compound") or "UNKNOWN", age

    def _is_in_lap(self, number: int, lap: int) -> bool:
        rows = self._laps[number]
        nxt = rows.get(lap + 1)
        if nxt is not None and nxt.get("is_pit_out_lap"):
            return True
        if (number, lap) in self._pit_rows and not rows[lap].get("is_pit_out_lap"):
            return True
        return False

    def _in_neutral(self, start: float | None, end: float | None) -> bool:
        if start is None:
            return False
        end = end if end is not None else start + 90.0
        spans = list(self._neutral)
        if self._flag_kind is not None:
            spans.append((self._flag_since, float("inf")))
        return any(a < end and start < b for a, b in spans)

    # -- output --------------------------------------------------------------------
    def _flag_updates(self, now: float) -> list[FlagEvent]:
        out = []
        ordered = sorted(self._control.values(), key=lambda m: m.get("date") or "")
        for m in ordered[self._flag_seen:]:
            t = _ts(m.get("date"))
            if t is None:
                self._flag_seen += 1
                continue
            if t > now:
                break
            self._flag_seen += 1
            started = _neutral_kind(m)
            if started and started != self._flag_kind:
                if self._flag_kind is not None:
                    self._neutral.append((self._flag_since, t))
                self._flag_since = t
                self._flag_kind = started
                out.append(FlagEvent(t, started, m.get("message") or started))
            elif _neutral_ended(m, self._flag_kind is not None) and self._flag_kind is not None:
                self._neutral.append((self._flag_since, t))
                self._flag_kind = None
                out.append(FlagEvent(t, None, m.get("message") or "TRACK CLEAR"))
        return out

    def events(self, now: float) -> list[Event]:
        out: list[Event] = list(self._flag_updates(now))
        for number, rows in self._laps.items():
            for lap in sorted(rows):
                key = (number, lap)
                r = rows[lap]
                if key in self._sent_laps:
                    if not self._sent_laps[key] and self._is_in_lap(number, lap):
                        self._sent_laps[key] = True
                        out.append(PitEvent(now, self._name(number), lap))
                    continue
                nxt = rows.get(lap + 1)
                start, dur = _ts(r.get("date_start")), r.get("lap_duration")
                next_start = _ts(nxt.get("date_start")) if nxt else None
                if next_start is not None:
                    line = next_start
                elif start is not None and dur is not None:
                    line = start + float(dur)
                else:
                    continue  # still on this lap
                if line > now:
                    continue
                if nxt is None and now - line < self.grace_s:
                    continue  # give the next lap's row, with its pit flag, time to arrive
                if dur is None and start is not None and next_start is not None and lap > 1:
                    dur = next_start - start
                compound, age = self._tyre(number, lap)
                pit_in = self._is_in_lap(number, lap)
                self._sent_laps[key] = pit_in
                out.append(LapEvent(
                    line, self._name(number), lap,
                    float(dur) if dur is not None else float("nan"), compound, age, pit_in,
                    bool(r.get("is_pit_out_lap")) and lap > 1, self._in_neutral(start, line)))
        return sorted(out, key=lambda e: e.t)


# ---------------------------------------------------------------------------
# OpenF1: transport


class OpenF1Client:
    """REST client. Historical data is free; live sessions need OPENF1_USERNAME/PASSWORD."""

    def __init__(self, base_url: str = "https://api.openf1.org", username: str | None = None,
                 password: str | None = None, timeout: float = 8.0):
        self.base_url, self.username, self.password = base_url.rstrip("/"), username, password
        self.timeout = timeout
        self._token: str | None = None
        self._token_until = 0.0

    def _authorise(self) -> None:
        if not self.username or time.time() < self._token_until:
            return
        body = urllib.parse.urlencode({"username": self.username, "password": self.password}).encode()
        req = urllib.request.Request(f"{self.base_url}/token", data=body, headers={
            "Content-Type": "application/x-www-form-urlencoded"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            got = json.load(resp)
        self._token = got["access_token"]
        self._token_until = time.time() + float(got.get("expires_in", 3600)) - 120.0

    def get(self, endpoint: str, **params) -> list[dict]:
        self._authorise()
        query = "&".join(f"{k.replace('__gte', '>')}={urllib.parse.quote(str(v), safe=':+-')}"
                         for k, v in params.items())
        req = urllib.request.Request(f"{self.base_url}/v1/{endpoint}?{query}",
                                     headers={"User-Agent": "pitwall-live"})
        if self._token:
            req.add_header("Authorization", f"Bearer {self._token}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                rows = json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code == 404:  # OpenF1 answers 404 for "no rows yet"
                return []
            if e.code == 401:
                self._token_until = 0.0
            raise
        return rows if isinstance(rows, list) else []


class RehearsalClient:
    """Serves a finished session as if it were running now: rows appear on schedule,
    open laps have no duration and running stints have no end, as in a live session."""

    def __init__(self, raw: dict[str, list[dict]], clock):
        self.raw, self.clock = raw, clock

    def get(self, endpoint: str, **_params) -> list[dict]:
        now = self.clock()
        rows = self.raw.get(endpoint, [])
        if endpoint == "laps":
            out = []
            for r in rows:
                start = _ts(r.get("date_start"))
                if start is None or start > now:
                    continue
                dur = r.get("lap_duration")
                out.append({**r, "lap_duration": dur if dur and start + dur <= now else None})
            return out
        if endpoint == "stints":
            current = {}
            for r in self.raw.get("laps", []):
                start = _ts(r.get("date_start"))
                if start is not None and start <= now:
                    d = r["driver_number"]
                    current[d] = max(current.get(d, 0), r["lap_number"])
            out = []
            for r in rows:
                seen = current.get(r["driver_number"], 0)
                if r.get("lap_start") is not None and r["lap_start"] <= seen:
                    end = r.get("lap_end")
                    out.append({**r, "lap_end": None if end is None or end >= seen else end})
            return out
        if endpoint in ("pit", "race_control"):
            return [r for r in rows if (_ts(r.get("date")) or now + 1) <= now]
        return rows


class OpenF1LiveFeed:
    """Polls in a background thread, so a slow network can never freeze the screen.

    Failures are reported through ``health`` and retried with a growing delay. The
    screen keeps showing the last good state and says how old it is.
    """

    ENDPOINTS = ("laps", "stints", "pit", "race_control")

    def __init__(self, info: RaceInfo, client, session_key, poll_s: float = 8.0,
                 clock=time.time, normalizer: OpenF1Normalizer | None = None):
        self.info, self.client, self.session_key = info, client, session_key
        self.poll_s, self.clock = poll_s, clock
        self.health = FeedHealth()
        self.norm = normalizer or OpenF1Normalizer()
        self._queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="openf1-poll")
        self._thread.start()

    def _poll_once(self) -> None:
        got = {e: self.client.get(e, session_key=self.session_key) for e in self.ENDPOINTS
               if e != "laps"}
        # Only recent laps: the full list grows to hundreds of KB per poll.
        recent = {}
        if self.health.last_success_t is not None:
            since = pd.Timestamp(self.health.last_success_t - 300.0, unit="s", tz="UTC")
            recent["date_start__gte"] = since.isoformat()
        got["laps"] = self.client.get("laps", session_key=self.session_key, **recent)
        self.norm.add(laps=got["laps"], stints=got["stints"], pits=got["pit"],
                      control=got["race_control"])
        for event in self.norm.events(self.clock()):
            self._queue.put(event)

    def _run(self) -> None:
        delay = self.poll_s
        while not self._stop.is_set():
            try:
                self._poll_once()
                self.health.ok = True
                self.health.last_success_t = self.clock()
                self.health.message = ""
                delay = self.poll_s
            except Exception as e:  # any failure must leave the screen running
                self.health.ok = False
                self.health.message = f"{type(e).__name__}: {e}"[:80]
                delay = min(delay * 1.5, 30.0)
            self._stop.wait(delay)

    def poll(self, now: float) -> list[Event]:
        out = []
        while True:
            try:
                out.append(self._queue.get_nowait())
            except queue.Empty:
                return out

    def next_time(self) -> float | None:
        return None

    @property
    def exhausted(self) -> bool:
        return False

    def close(self) -> None:
        self._stop.set()


def open_session(client, session: str, laps_total: int | None, priors, source: str,
                 raw_laps: list[dict] | None = None, drivers=None) -> tuple[RaceInfo, dict]:
    """Look up a session and build its RaceInfo. ``session`` is a key or 'latest'."""
    rows = client.get("sessions", session_key=session)
    if not rows:
        raise ValueError(f"OpenF1 has no session {session!r}")
    s = rows[-1]
    if s.get("session_name") != "Race":
        raise ValueError(f"session {s.get('session_key')} is {s.get('session_name')!r}, not a Race")
    circuit = s.get("circuit_short_name")
    if laps_total is None and raw_laps:
        laps_total = max(r["lap_number"] for r in raw_laps)
    if laps_total is None:
        laps_total = priors.laps_by_circuit.get(circuit)
    if laps_total is None:
        raise ValueError(f"race distance unknown for {circuit}: pass --laps")
    drivers = drivers if drivers is not None else client.get("drivers", session_key=s["session_key"])
    label = f"{s.get('year')} {s.get('country_name')} ({circuit})"
    info = RaceInfo(label, source, s.get("year"), circuit, int(laps_total),
                    [d["name_acronym"] for d in drivers if d.get("name_acronym")],
                    _ts(s.get("date_start")) or time.time(), live=source == "LIVE")
    return info, {"session": s, "drivers": drivers}


__all__ = ["Event", "FeedHealth", "FlagEvent", "LapEvent", "OpenF1Client", "OpenF1LiveFeed",
           "OpenF1Normalizer", "PitEvent", "RaceInfo", "RehearsalClient", "ScheduledFeed",
           "bundled_feed", "feed_from_frame", "open_session"]
