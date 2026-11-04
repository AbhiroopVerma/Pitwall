"""Turns an engine Frame into text. No figures, no dependencies, no logic about racing.

Laid out for someone with a radio in one hand: the call and the words to say come
first, in the largest contrast, and everything below only backs them up. Plain mode
prints the same content with no cursor movement, for logs and slow links.
"""
from __future__ import annotations

import math
import re
import shutil

from .live_engine import Call, Frame, Option

MIN_WIDTH = 60
RESET = "\033[0m"
STYLES = {
    "bold": "\033[1m", "dim": "\033[2m",
    "red": "\033[1;97;41m", "yellow": "\033[1;30;43m", "green": "\033[1;30;42m",
    "blue": "\033[1;97;44m", "grey": "\033[1;97;100m",
    "t_red": "\033[1;91m", "t_yellow": "\033[93m", "t_green": "\033[92m",
}
CALL_STYLE = {"BOX": "red", "STANDBY": "yellow", "STAY": "green", "PIT": "blue", "NONE": "grey"}
FLAG_STYLE = {"SC": "yellow", "VSC": "yellow", "RED": "red", None: "green"}
FLAG_TEXT = {"SC": " SAFETY CAR ", "VSC": " VIRTUAL SC ", "RED": " RED FLAG ", None: " GREEN "}
ALERT_STYLE = {"CRIT": "t_red", "WARN": "t_yellow", "INFO": "dim"}
ANSI = re.compile(r"\033\[[0-9;]*m")


class Paint:
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def __call__(self, style: str, text: str) -> str:
        return f"{STYLES[style]}{text}{RESET}" if self.enabled else text


def visible_len(text: str) -> int:
    return len(ANSI.sub("", text))


def clip(text: str, width: int) -> str:
    """Cut to width, counting only visible characters."""
    if visible_len(text) <= width:
        return text
    out, seen, i = [], 0, 0
    while i < len(text) and seen < width - 1:
        m = ANSI.match(text, i)
        if m:
            out.append(m.group())
            i = m.end()
            continue
        out.append(text[i])
        seen += 1
        i += 1
    joined = "".join(out) + "~"
    return joined + RESET if "\033[" in joined else joined


def wrap(text: str, width: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > width:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}" if cur else word
    return lines + ([cur] if cur else [])


def lap_time(t: float | None) -> str:
    if t is None or not math.isfinite(t):
        return "--:--.---"
    return f"{int(t // 60)}:{t % 60:06.3f}"


def signed(v: float | None, digits: int = 1) -> str:
    return "  -  " if v is None or not math.isfinite(v) else f"{v:+.{digits}f}"


def _bar(width: int, char: str = "-") -> str:
    return char * width


def _gap_words(gap: float | None, rival: str) -> str:
    if gap is None or not math.isfinite(gap):
        return f"gap to {rival} unknown"
    return f"{abs(gap):.1f}s {'BEHIND' if gap > 0 else 'AHEAD OF'} {rival}"


def _header(f: Frame, paint: Paint, width: int, speed: float | None) -> list[str]:
    info = f.info
    source = info.source if not speed or info.live else f"{info.source} x{speed:g}"
    left = f"PITWALL  {info.label}"
    right = source
    line1 = paint("bold", left) + " " * max(1, width - len(left) - len(right)) + right
    lap_now = min(f.own_lap + 1, info.laps_total) if not f.finished else info.laps_total
    flag = paint(FLAG_STYLE.get(f.flag, "grey"), FLAG_TEXT.get(f.flag, f" {f.flag} "))
    eta = f"  line in {f.seconds_to_line:>3.0f}s" if f.seconds_to_line is not None and not f.finished else ""
    pos = f"P{f.own.position}" if f.own and f.own.position else ""
    line2 = paint("bold", f"LAP {lap_now}/{info.laps_total}") + f"  {flag}  "
    if f.own and f.rival:
        line2 += f"{f.own.driver} {pos}  vs  {f.rival.driver} P{f.rival.position}{eta}"
    return [line1, line2]


def _call_box(call: Call, paint: Paint, width: int) -> list[str]:
    style = CALL_STYLE.get(call.kind, "grey")
    head = f"  {call.headline}"
    lines = [paint(style, " " * width), paint(style, head.ljust(width)[:width]), paint(style, " " * width)]
    if call.radio:
        lines.append(paint("bold", f'RADIO  "{call.radio}"'))
    tags = []
    if call.confidence:
        tags.append(f"confidence {call.confidence}")
    if call.since_lap and call.kind in ("BOX", "STANDBY", "STAY"):
        tags.append(f"call since lap {call.since_lap}")
    if tags:
        lines.append("       " + ", ".join(tags))
    for part in wrap(call.why, width - 7):
        lines.append(f"       {part}")
    return lines


def rival_stop_outcome(f: Frame) -> str:
    """Where the rival would come out if they stopped now. Gap + means we trail them,
    and their stop moves them back by its cost, so the gap falls by that much."""
    if f.gap is None or not f.rival or not f.own or f.rival.in_pit or f.own.in_pit:
        return ""
    after = f.gap - f.stop_cost_now
    where = f"{after:.1f}s ahead of us" if after > 0 else f"{-after:.1f}s behind us"
    return f"If {f.rival.driver} boxes this lap they come out {where}, on fresh tyres."


def _battle(f: Frame, paint: Paint, width: int) -> list[str]:
    if not f.own or not f.rival:
        return []
    o, r, fc = f.own, f.rival, f.forecast
    gap_line = paint("bold", _gap_words(f.gap, r.driver))
    if f.gap is not None and f.gap_line is not None and abs(f.gap - f.gap_line) > 0.05:
        gap_line += f"  (line gap {f.gap_line:+.1f}s, stop in progress counted)"
    if f.projected:
        gap_line += "  [rival lap estimated]"
    lines = [gap_line]
    trend = []
    if f.gap_trend is not None:
        word = "closing" if (f.gap_trend < 0) == (f.gap is not None and f.gap > 0) else "opening"
        trend.append(f"{word} {abs(f.gap_trend):.2f}s/lap")
    if f.pace_delta is not None:
        word = "faster" if f.pace_delta < 0 else "slower"
        trend.append(f"our pace {abs(f.pace_delta):.2f}s/lap {word}")
    cost = f"stop costs {f.stop_cost_now:.1f}s"
    if f.flag:
        cost += f" now (green {f.pit_loss:.1f}s)"
    else:
        cost += f" ({f.pit_loss_source})"
    trend.append(cost)
    lines.extend(wrap("; ".join(trend), width))
    lines.append("")
    lines.append(paint("dim", f"{'':<5}{'TYRE':<14}{'LOST TO WEAR':<20}{'LAST':<11}{'PACE':<11}STOPS"))
    for car in (o, r):
        tyre = f"{car.compound[:6]} {car.age}"
        wear = ("  -  " if car.penalty is None else "none visible yet" if car.penalty < 0.05
                else f"{car.penalty:+.2f}s, {car.slope:+.3f}/lap")
        must = " must stop" if car.must_stop else ""
        state = " IN PIT" if car.in_pit else "" if car.running else " STOPPED"
        lines.append(f"{car.driver:<5}{tyre:<14}{wear:<20}{lap_time(car.last_lap):<11}"
                     f"{lap_time(car.pace):<11}{car.stops}{must}{state}")
    lines.append("")
    head = f"{r.driver} STOP  this lap {fc.p_this_lap:>3.0%}   within 3 {fc.p3:>3.0%}   within 5 {fc.p5:>3.0%}"
    if fc.likely_lap and not r.in_pit:
        head += f"   likely lap {fc.likely_lap}"
    lines.append(paint("t_yellow", head) if fc.p3 >= 0.3 else head)
    if fc.p3 >= 0.4:
        lines.append(paint("dim", "Readings this high ran hot on unseen races: 40-50% came true 25%."))
    outcome = rival_stop_outcome(f)
    if outcome:
        lines.append(outcome)
    return lines


def _options(options: list[Option], chosen_lap, kind: str, paint: Paint, width: int) -> list[str]:
    if not options:
        return []
    lines = [paint("dim", f"  {'OPTION':<20}{'WIN':>5}{'TIME':>9}{'STOPS':>7}  REJOIN")]
    reference = next((o for o in options if o.box_lap == chosen_lap), options[0])
    ordered = sorted(options, key=lambda o: (o.offset is None, o.offset or 0))
    for o in ordered:
        mark = ">" if o.box_lap == chosen_lap else " "
        name = (f"box end lap {o.box_lap}" if o.offset is not None
                else f"stay, stop ~{o.box_lap}" if o.box_lap else "stay to flag")
        rejoin = ""
        if o.rejoin is not None:
            r = o.rejoin
            near = (f"traffic {r.road_ahead} {r.road_gap:.1f}s" if r.traffic else "clear")
            rejoin = f"P{r.position:<3}{near}"
            if r.ahead_to_stop:
                rejoin += f", {r.ahead_to_stop} ahead to stop"
        delta = "call" if o is reference else f"{o.exp_gap - reference.exp_gap:+.1f}s"
        row = f"{mark} {name:<20}{o.win:>5.0%}{delta:>9}{len(o.stops):>7}  {rejoin}"
        lines.append(paint("bold", row) if mark == ">" else row)
    lines.append(paint("dim", "  WIN = chance of beating the rival. TIME = at the flag, against the call."))
    return lines


def _field_table(f: Frame, paint: Paint, n: int) -> list[str]:
    if not f.field_rows or not f.own:
        return []
    own = f.own.driver
    rival = f.rival.driver if f.rival else ""
    rows = f.field_rows
    index = next((i for i, r in enumerate(rows) if r.driver == own), 0)
    start = max(0, min(index - n // 2, len(rows) - n))
    shown = rows[start:start + n]
    extra = [r for r in rows if r.driver == rival and r not in shown]
    lines = [paint("dim", f"{'POS':<5}{'CAR':<6}{'GAP':>9}  {'TYRE':<11}{'LAST':<11}STOPS")]
    for r in shown + extra:
        gap = "--" if r.driver == own else (f"{r.gap:+.1f}s" if r.gap is not None else "?")
        status = " IN PIT" if r.in_pit else "" if r.running else " OUT"
        row = (f"P{r.position:<4}{r.driver:<6}{gap:>9}  {r.compound[:6] + ' ' + str(r.age):<11}"
               f"{lap_time(r.last_lap):<11}{r.stops}{status}")
        if r.driver == own:
            row = paint("bold", row)
        elif r.driver == rival:
            row = paint("t_yellow", row)
        lines.append(row)
    return lines


def _alerts(f: Frame, paint: Paint, n: int) -> list[str]:
    return [paint(ALERT_STYLE.get(a.level, "dim"), f"L{a.lap:<3} {a.text}") for a in f.alerts[-n:]][::-1]


def _footer(f: Frame, paint: Paint, width: int, keys: str) -> list[str]:
    bits = []
    health = f.health
    if health is not None and not health.ok:
        bits.append(paint("t_red", f"FEED DOWN: {health.message}"))
    elif health is not None and health.last_success_t is not None and f.info.live:
        age = f.now - health.last_success_t
        bits.append(paint("t_red", f"FEED STALE {age:.0f}s") if age > 30 else f"feed {age:.0f}s old")
    if f.pace_error is not None:
        bits.append(f"next-lap pace error {f.pace_error:.2f}s (last {f.pace_error_n})")
    if f.rival_stops_seen:
        bits.append(f"rival stops flagged {f.rival_stops_warned}/{f.rival_stops_seen}")
    bits.append(f"calc {f.calc_ms:.0f}ms")
    lines = wrap(" | ".join(bits), width) if not any("\033" in b for b in bits) else [" | ".join(bits)]
    if f.info.source in ("REAL REPLAY", "REHEARSAL"):
        lines.append(paint("dim", "Replay: cars do what really happened, not what is called here."))
    if f.stale_reason:
        lines.insert(0, paint("t_red", f"STALE: {f.stale_reason}"))
    if keys:
        lines.append(paint("dim", keys))
    return lines


def _panel(rows: list[tuple[str, str]], paint: Paint, width: int) -> list[str]:
    return [paint(style, part) for style, text in rows for part in wrap(text, width)]


def render(f: Frame, width: int = 100, height: int = 50, color: bool = True,
           speed: float | None = None, keys: str = "",
           panel: list[tuple[str, str]] | None = None) -> list[str]:
    """The whole screen as lines. Sections are dropped from the bottom if it will not fit."""
    width = max(MIN_WIDTH, min(width, 110))
    paint = Paint(color)
    rule = paint("dim", _bar(width))
    if f.info.source == "FABRICATED":
        banner = [paint("t_yellow", "FABRICATED RACE: not real data")]
    else:
        banner = []
    top = _header(f, paint, width, speed) + banner + _call_box(f.call, paint, width)
    battle = _battle(f, paint, width)
    options = _options(f.options, f.call.box_lap, f.call.kind, paint, width)
    tyres = [t for t in wrap("TYRES  " + f.tyre_note, width)] if f.tyre_note else []
    footer = _footer(f, paint, width, keys)
    lines = list(top)
    for section in (_panel(panel or [], paint, width), options + tyres, battle):
        if section:
            lines += [rule] + section
    used = len(lines) + len(footer) + 1
    room = height - used
    field_n = max(0, min(9, room - 8))
    alerts_n = max(0, min(6, room - field_n - 3))
    field_rows = _field_table(f, paint, field_n) if field_n >= 3 else []
    if field_rows:
        lines += [rule] + field_rows
    alert_rows = _alerts(f, paint, alerts_n)
    if alert_rows:
        lines += [rule] + alert_rows
    lines += [rule] + footer
    return [clip(l, width) for l in lines]


def terminal_size() -> tuple[int, int]:
    size = shutil.get_terminal_size((100, 50))
    return size.columns, size.lines
