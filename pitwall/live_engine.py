"""The live race engine: events in, a stable pit-wall call out.

It never sleeps and never prints, so the same code runs in the terminal, in the
scoring harness and in the tests. On each line crossing it runs the full pipeline:

    laps so far -> tyre models -> rival stop forecast -> simulate every option to the
    flag -> rank -> keep the call stable -> say it in plain words

What makes it a live tool rather than a lap-by-lap report:

- **It always has a plan.** Our own stop plan is kept current even when the rival is
  no threat. The threat detector only raises alerts.
- **It recalculates on news.** A safety car or a stop mid-lap updates the call at once.
- **It does not wait.** If the rival is behind us on the road, their lap is projected
  after ``opp_wait_s`` and corrected when they cross the line, so a call never describes
  a lap we have already finished for longer than that.
- **Calls hold.** A strategist cannot use a call that changes on Monte Carlo noise. A
  call is replaced only if the new one is better by ``hysteresis``, or the race changed.
- **Calls name a lap.** "Box +1" one lap and "box now" the next are the same call,
  so it is tracked as the lap to box on.
- **A failure never ends the session.** If a calculation fails, the last call stays on
  screen marked stale, with the reason.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field

import numpy as np

from .demo import N_SIMS, WINDOW, decide_on_replay
from .live_feed import Event, FeedHealth, FlagEvent, LapEvent, PitEvent, RaceInfo
from .live_field import CarRow, FieldState, Rejoin
from .optimizer import utility
from .pit_model import PitHazardModel, load_models
from .priors import RacePriors, load_priors
from .simulator import action_label
from .state import LapRecord, Replay

HYSTERESIS = 0.03    # utility a new call must gain to replace the current one
COMMIT_UTILITY = 0.02  # a stop must beat staying out by this much, or by COMMIT_SECONDS,
COMMIT_SECONDS = 1.0   # since staying out keeps every later option open
OPP_WAIT_S = 3.0     # race seconds to wait for the rival before projecting their lap
THETA = 0.30         # P(rival stops within 3 laps) that raises the stop-window alert
SLOW_LAP_S = 1.5     # lap this far off prediction raises an alert
MAX_ALERTS = 40


@dataclass
class Option:
    label: str
    offset: int | None       # None = stay out past the action window
    box_lap: int | None      # the lap we box on under this option, None = no more stops
    p_ahead: float           # as simulated; this ranks the options
    exp_gap: float           # + = ahead of the rival at the flag
    worst: float             # bad-case gap (5th percentile)
    utility: float
    stops: tuple
    rejoin: Rejoin | None
    win: float = -1.0        # calibrated against real results; this is what is displayed


@dataclass
class Alert:
    lap: int
    level: str   # CRIT | WARN | INFO
    text: str


@dataclass
class Call:
    kind: str = "NONE"   # BOX | STANDBY | STAY | PIT | NONE
    headline: str = "NO CALL YET"
    radio: str = ""
    why: str = ""
    box_lap: int | None = None
    confidence: str = ""
    since_lap: int = 0
    held: bool = False   # current call kept although another option is slightly ahead


@dataclass
class CarPanel:
    driver: str
    position: int = 0
    lap: int = 0
    compound: str = "?"
    age: int = 0
    last_lap: float = float("nan")
    pace: float | None = None
    stops: int = 0
    penalty: float | None = None      # seconds lost to tyre age now
    slope: float | None = None        # seconds lost per extra lap on this tyre
    in_pit: bool = False
    must_stop: bool = False
    running: bool = True


@dataclass
class RivalForecast:
    p_this_lap: float = 0.0
    p3: float = 0.0
    p5: float = 0.0
    likely_lap: int | None = None
    threat: bool = False
    threat_reason: str = ""


@dataclass
class Frame:
    info: RaceInfo
    now: float
    lap: int = 0               # last lap the decision is based on
    own_lap: int = 0
    finished: bool = False
    flag: str | None = None
    own: CarPanel | None = None
    rival: CarPanel | None = None
    gap: float | None = None   # + = we trail, once any stop in progress is paid for
    gap_line: float | None = None
    gap_trend: float | None = None  # seconds per lap, - = we are gaining
    pace_delta: float | None = None  # our pace minus theirs, - = we are faster
    projected: bool = False    # rival's latest lap is estimated, not yet measured
    pit_loss: float = 0.0
    pit_loss_source: str = ""
    stop_cost_now: float = 0.0
    forecast: RivalForecast = field(default_factory=RivalForecast)
    call: Call = field(default_factory=Call)
    options: list[Option] = field(default_factory=list)
    tyre_note: str = ""
    field_rows: list[CarRow] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)
    pace_error: float | None = None   # own and rival next-lap forecast error so far
    pace_error_n: int = 0
    rival_stops_seen: int = 0
    rival_stops_warned: int = 0
    calc_ms: float = 0.0
    stale_reason: str = ""
    seconds_to_line: float | None = None
    decision_id: int = 0
    health: FeedHealth | None = None
    error_trace: str = ""


class CallTracker:
    """Holds the call unless something clearly better appears."""

    def __init__(self, margin: float = HYSTERESIS):
        self.margin = margin
        self.box_lap: int | None = None
        self.has_call = False
        self.since = 0

    def reset(self) -> None:
        self.has_call = False

    def choose(self, ranked: list[Option], lap: int, force: bool = False) -> tuple[Option, bool]:
        """Pick from options sorted best first. Returns (option, held)."""
        best = self._earn_the_stop(ranked)
        kept = None if force or not self.has_call else self._same_call(ranked, lap)
        if kept is not None and best.utility - kept.utility < self.margin:
            chosen, held = kept, kept is not best
        else:
            chosen, held = best, False
        if not self.has_call or not self._same_target(chosen, lap):
            self.since = lap
        self.box_lap = chosen.box_lap
        self.has_call = True
        return chosen, held

    @staticmethod
    def _earn_the_stop(ranked: list[Option]) -> Option:
        """A stop is irreversible and waiting is not, so near-ties go to staying out."""
        best = ranked[0]
        if best.offset is None:
            return best
        stay = next((o for o in ranked if o.offset is None), None)
        if (stay is not None and best.utility - stay.utility < COMMIT_UTILITY
                and best.exp_gap - stay.exp_gap < COMMIT_SECONDS):
            return stay
        return best

    def _same_target(self, option: Option, lap: int) -> bool:
        return option is self._same_call([option], lap)

    def _same_call(self, options: list[Option], lap: int) -> Option | None:
        for o in options:
            if o.box_lap == self.box_lap and (o.offset is not None or self.box_lap is None):
                return o
        # A planned stop past the action window can drift a lap or two without being a
        # new call. Only the stay-out option can represent it.
        if self.box_lap is not None and self.box_lap > lap + 1 + WINDOW:
            return next((o for o in options if o.offset is None), None)
        return None


def _confidence(chosen: Option, ranked: list[Option]) -> str:
    others = [o for o in ranked if o is not chosen]
    if not others:
        return "ONLY OPTION"
    if chosen.p_ahead >= 0.98 or chosen.p_ahead <= 0.02:
        # The fight is settled either way, so only lost time separates the options.
        margin = chosen.exp_gap - max(o.exp_gap for o in others)
        return "HIGH" if margin >= 2.0 else "MEDIUM" if margin >= 0.5 else "LOW"
    margin = chosen.utility - max(o.utility for o in others)
    return "HIGH" if margin >= 0.08 else "MEDIUM" if margin >= 0.03 else "LOW"


class LiveEngine:
    def __init__(self, info: RaceInfo, own: str, opp: str, theta: float = THETA,
                 n_sims: int = N_SIMS, overtake_delta: float = 1.0,
                 opp_wait_s: float = OPP_WAIT_S, hysteresis: float = HYSTERESIS,
                 pit_model: PitHazardModel | None = None, priors: RacePriors | None = None):
        self.info, self.own, self.opp = info, own, opp
        self.theta, self.n_sims, self.overtake_delta = theta, n_sims, overtake_delta
        self.opp_wait_s = opp_wait_s
        self.pit_model = pit_model if pit_model is not None else load_models()[0]
        self.priors = priors if priors is not None else load_priors()
        self.prior_pit_loss = self.priors.pit_loss(info.circuit)
        self.field = FieldState()
        self.tracker = CallTracker(hysteresis)
        self.flag_kind: str | None = None
        self.alerts: list[Alert] = []
        self.frame = Frame(info, info.start_t)
        self.decision_id = 0
        self.chequered = False
        self._decided: tuple | None = None
        self._dirty = False
        self._force = False
        self._reason = ""
        self._prev_call_kind = None
        self._prev_trigger = False
        self._predicted: dict[tuple[str, int], float] = {}
        self._errors: list[float] = []
        self._warned_at: dict[int, float] = {}
        self._rival_stops = 0
        self._rival_warned = 0
        self._last_box_call: int | None = None
        self._miss_noted = False
        self._was_ahead: bool | None = None
        self._out: dict | None = None

    # -- intake --------------------------------------------------------------------
    def ingest(self, events: list[Event]) -> None:
        for e in events:
            if isinstance(e, LapEvent):
                self._on_lap(e)
            elif isinstance(e, PitEvent):
                self._on_late_pit(e)
            elif isinstance(e, FlagEvent):
                self._on_flag(e)

    def _alert(self, level: str, text: str) -> None:
        self.alerts.append(Alert(self.field.last_lap(self.own), level, text))
        del self.alerts[:-MAX_ALERTS]

    def _on_flag(self, e: FlagEvent) -> None:
        if e.kind == self.flag_kind:
            return
        self.flag_kind = e.kind
        self._dirty = self._force = True
        self._reason = "flag"
        if e.kind is None:
            self._alert("WARN", "GREEN: racing resumes, normal stop cost again")
        else:
            words = {"SC": "SAFETY CAR", "VSC": "VIRTUAL SAFETY CAR", "RED": "RED FLAG"}
            self._alert("CRIT", f"{words.get(e.kind, e.kind)} deployed: stop cost has changed")

    def _on_late_pit(self, e: PitEvent) -> None:
        self.field.mark_in_lap(e.driver, e.lap)
        if e.driver in (self.own, self.opp):
            self._announce_stop(e.driver, e.lap)
            self._dirty = self._force = True

    def _announce_stop(self, driver: str, lap: int) -> None:
        rec = self.field.cars.get(driver, {}).get(lap)
        tyre = f"{rec.compound} {rec.age} laps old" if rec else ""
        if driver == self.opp:
            warned = any(l in self._warned_at for l in range(lap - 3, lap))
            self._rival_stops += 1
            self._rival_warned += int(warned)
            note = "forecast had flagged it" if warned else "forecast did not flag it"
            self._alert("CRIT", f"{driver} BOXED end of lap {lap}, came off {tyre} ({note})")
        else:
            self._alert("WARN", f"{driver} boxed end of lap {lap}, came off {tyre}")

    def _on_lap(self, e: LapEvent) -> None:
        self.field.add(e)
        predicted = self._predicted.pop((e.driver, e.lap), None)
        clean = not (e.pit_in or e.pit_out or e.neutral) and np.isfinite(e.lap_time)
        if predicted is not None and clean:
            error = e.lap_time - predicted
            self._errors.append(abs(error))
            del self._errors[:-20]
            if e.driver == self.own and error > SLOW_LAP_S:
                self._alert("WARN", f"{e.driver} lap {e.lap}: {e.lap_time:.3f}, "
                                    f"{error:.1f}s slower than forecast. Traffic or problem?")
        if e.driver in (self.own, self.opp):
            # A rival a lap further on changes nothing until we get there too, unless it
            # stops, which the call must respond to.
            if e.driver == self.own or e.lap <= self.field.last_lap(self.own) or e.pit_in:
                self._dirty = True
            if e.pit_in:
                self._announce_stop(e.driver, e.lap)
                self._force = True
            if e.pit_out and e.driver == self.opp:
                self._alert("WARN", f"{e.driver} out-lap done, on {e.compound}, now "
                                    f"P{self.field.position(e.driver)}")
        if e.lap >= self.info.laps_total:
            self.chequered = True

    def set_opponent(self, code: str) -> None:
        if code == self.own or code not in self.field.cars:
            raise ValueError(f"cannot target {code}")
        self.opp = code
        self._rival_stops = self._rival_warned = 0
        self._warned_at.clear()
        self._dirty = self._force = True
        self._prev_trigger = False
        self._was_ahead = None
        self._alert("INFO", f"Target changed to {code}")

    # -- decision ------------------------------------------------------------------
    def _records(self, lap: int, projected: bool) -> tuple[list[LapRecord], list[LapRecord]]:
        own_laps = self.field.cars[self.own]
        opp_laps = self.field.cars[self.opp]

        def record(name: str, role: str, n: int, r, gap: float) -> LapRecord:
            return LapRecord(n, role, r.compound, r.age, r.lap_time, 0, gap,
                             safety_car=r.neutral, pitted=r.pit_in, pit_out=r.pit_out)

        own_recs, opp_recs, gap = [], [], 0.0
        for n in sorted(set(own_laps) | set(opp_laps)):
            if n > lap:
                break
            a, b = own_laps.get(n), opp_laps.get(n)
            if a is not None and b is not None:
                gap = a.t - b.t
            if a is not None:
                own_recs.append(record(self.own, "OWN", n, a, gap))
            if b is not None:
                opp_recs.append(record(self.opp, "OPP", n, b, -gap))
        if projected:
            prev = opp_laps[max(opp_laps)]
            pace = self.field.pace(self.opp) or self.field.pace(self.own) or prev.lap_time
            age = 1 if prev.pit_in else prev.age + 1
            line = prev.t + pace * (lap - max(opp_laps))
            gap = own_laps[lap].t - line
            own_recs[-1].gap_to_rival = gap
            opp_recs.append(LapRecord(lap, "OPP", prev.compound, age, pace, 0, -gap,
                                      safety_car=self.flag_kind is not None))
        return own_recs, opp_recs

    def _ready_lap(self, now: float) -> tuple[int, bool] | None:
        own_last = self.field.last_lap(self.own)
        if own_last == 0 or self.field.last_lap(self.opp) == 0:
            return None
        if own_last in self.field.cars[self.opp]:
            return own_last, False
        # We have crossed and the rival has not. Wait a little, then estimate their lap.
        if now - self.field.line_time(self.own, own_last) >= self.opp_wait_s:
            return own_last, True
        # Meanwhile, use the latest lap both cars have finished if it is newer than the
        # last decision. Polling can deliver several laps at once.
        common = max(set(self.field.cars[self.own]) & set(self.field.cars[self.opp]), default=0)
        if common and (self._decided is None or self._decided[0] < common):
            return common, False
        return None

    def _options(self, out: dict, lap: int, pit_loss: float, stop_costs, now: float,
                 gap_now: float) -> list[Option]:
        rec = out["recommendation"]
        by_offset = {r.our_pit_offset: r for r in out["results"]}
        options = []
        for offset, _ in rec.ranked:
            r = by_offset[offset]
            if offset is not None:
                box = lap + 1 + offset
                cost = stop_costs[offset] if stop_costs and offset < len(stop_costs) else pit_loss
            else:
                box = r.stops[0] if r.stops else None
                cost = pit_loss
            rejoin = self.field.rejoin(self.own, box, cost, now) if box is not None else None
            win = self.priors.win_chance(r.p_ahead, -gap_now, self.info.laps_total - lap)
            options.append(Option(action_label(offset), offset, box, r.p_ahead, r.exp_gap, r.risk,
                                  utility(r, pit_loss), tuple(r.stops), rejoin, win))
        return options

    def _tyre_note(self, box_lap: int | None, own_compound: str, must_stop: bool) -> str:
        if box_lap is None:
            return ""
        left = self.info.laps_total - box_lap
        parts = []
        for compound in ("HARD", "MEDIUM", "SOFT"):
            if must_stop and compound == own_compound:
                continue
            limit = self.priors.stint_limit(self.info.circuit, compound)
            if limit is None:
                continue
            word = "reaches flag" if left <= limit else "needs another stop"
            parts.append(f"{compound} {word} ({limit} lap norm)")
        return f"{left} laps left after that stop: " + ", ".join(parts) if parts else ""

    def _say(self, kind: str, box_lap: int | None, lap: int, rival_stopped: bool,
             threat: bool) -> tuple[str, str]:
        """(headline, radio message) for this call."""
        opp = self.opp
        flag = {"SC": "safety car", "VSC": "VSC"}.get(self.flag_kind or "")
        if kind == "BOX":
            head = f"BOX BOX BOX  -  THIS LAP (end of lap {box_lap})"
            if flag:
                return head, f"{flag.capitalize()}, cheap stop. Box this lap, box this lap. Confirm."
            if rival_stopped:
                return head, f"{opp} has boxed. Box this lap, box this lap. Confirm."
            return head, "Box this lap, box this lap. Confirm."
        if kind == "STANDBY":
            n = box_lap - (lap + 1)
            head = f"STAND BY  -  BOX END OF LAP {box_lap} ({n} lap{'s' if n != 1 else ''} after this)"
            return head, f"Plan is box end of lap {box_lap}. Stay in the window till then."
        if kind == "PIT":
            head = "OUT-LAP  -  BUILD TYRES, " + (f"NEXT STOP AROUND LAP {box_lap}"
                                                  if box_lap else "NO MORE STOPS PLANNED")
            return head, "Good stop. Build the tyres, then push. Gap to follow."
        head = "STAY OUT  -  " + (f"NEXT STOP AROUND LAP {box_lap}" if box_lap
                                  else "NO MORE STOPS, RUN TO THE FLAG")
        if flag:
            return head, f"Stay out, stay out under the {flag}. Hold position."
        if rival_stopped:
            return head, f"{opp} has boxed, we stay out. Push now, you have clean air."
        if threat:
            return head, f"Stay out. {opp} could stop soon, keep the pace up."
        if box_lap:
            return head, f"Stay out. Plan is around lap {box_lap}. Tell us how the tyres feel."
        return head, "Stay out to the flag. Bring it home."

    def _why(self, chosen: Option, ranked: list[Option], held: bool) -> str:
        others = [o for o in ranked if o is not chosen]
        text = f"Win chance {chosen.win:.0%}"
        if others:
            o = others[0]
            what = f"box lap {o.box_lap}" if o.offset is not None else "staying out"
            text += (f"; next best {what} {o.win:.0%}, "
                     f"{o.exp_gap - chosen.exp_gap:+.1f}s at the flag")
        if chosen.offset is not None and chosen.rejoin is not None:
            r = chosen.rejoin
            clear = (f"traffic: {r.road_ahead} {r.road_gap:.1f}s ahead" if r.traffic
                     else "clear air")
            text += f"; rejoin P{r.position}, {clear}"
        if held:
            text += "; holding call, the difference is too small to change it"
        return text

    def _decide(self, now: float, lap: int, projected: bool) -> None:
        self.decision_id += 1
        frame = self.frame
        own_recs, opp_recs = self._records(lap, projected)
        replay = Replay(self.info.laps_total, own_recs, opp_recs, self.prior_pit_loss)
        pit_loss = self.field.pit_loss(self.prior_pit_loss)
        stop_costs = self.priors.stop_costs(pit_loss, self.flag_kind)
        out = decide_on_replay(
            replay, lap, self.theta, self.pit_model, False, self.overtake_delta, self.n_sims,
            always_simulate=True, pit_loss=pit_loss, stop_costs=stop_costs,
            neutral=True if self.flag_kind else None, pace_shrink=self.priors.pace_shrink,
            pace_sd=self.priors.pace_sd, wear_prior=self.priors.wear_prior(self.info.circuit),
            prior_strength=self.priors.prior_strength,
            stint_limits=self.priors.stint_limits(self.info.circuit),
            pace_tau=self.priors.pace_tau)
        self._out = out
        snap = replay.state_at(lap)
        snap.pit_loss = pit_loss
        frame.lap, frame.projected = lap, projected
        frame.pit_loss = pit_loss
        frame.pit_loss_source = (f"measured, {len(self.field.stop_costs)} stops"
                                 if len(self.field.stop_costs) >= 3 else "circuit history")
        frame.stop_cost_now = stop_costs[0] if stop_costs else pit_loss
        frame.gap = snap.effective_gap
        frame.gap_line = snap.gap_own_behind_opp
        frame.rival.must_stop = snap.opp.must_stop
        frame.own.must_stop = snap.own.must_stop
        self._fill_trend(frame, lap)

        trig, fc = out["trigger"], out.get("forecast")
        if fc is not None:
            dist = fc.pit_dist[:-1]
            top = int(np.argmax(dist))
            frame.forecast = RivalForecast(
                fc.hazards[0], fc.p_within[3], fc.p_within[5],
                lap + 1 + top if dist[top] > 0 else None, trig.active, trig.reason)
            if fc.p_within[3] >= self.theta:
                self._warned_at[lap] = fc.p_within[3]
            if trig.active and not self._prev_trigger and not snap.opp.just_pitted:
                self._alert("WARN", f"{self.opp} stop window: {fc.p_within[3]:.0%} within 3 laps")
            self._prev_trigger = trig.active
        est, own_model = out.get("est"), out.get("own_model")
        if est is not None:
            frame.rival.penalty, frame.rival.slope = est.penalty, est.slope
            if not snap.opp.just_pitted:
                self._predicted[(self.opp, lap + 1)] = est.forecast[0]
        if own_model is not None:
            frame.own.penalty = own_model.penalty_at(snap.own.tyre_age, snap.own.compound)
            frame.own.slope = own_model.slope_at(snap.own.tyre_age, snap.own.compound)
            if not snap.own.just_pitted:
                self._predicted[(self.own, lap + 1)] = own_model.pace(
                    snap.own.next_age, lap + 1, snap.own.compound)

        if "recommendation" not in out:
            frame.options, frame.tyre_note = [], ""
            frame.call = Call("NONE", "NO CALL YET", "", trig.reason)
            self.tracker.reset()
            return

        ranked = self._options(out, lap, pit_loss, stop_costs, now, snap.effective_gap)
        force = self._force or snap.own.just_pitted or snap.opp.just_pitted
        chosen, held = self.tracker.choose(ranked, lap, force)
        kind = ("PIT" if snap.own.just_pitted else
                "BOX" if chosen.offset == 0 else
                "STANDBY" if chosen.offset is not None else "STAY")
        head, radio = self._say(kind, chosen.box_lap, lap, snap.opp.just_pitted, trig.active)
        frame.call = Call(kind, head, radio, self._why(chosen, ranked, held), chosen.box_lap,
                          _confidence(chosen, ranked), self.tracker.since, held)
        frame.options = ranked
        frame.tyre_note = self._tyre_note(chosen.box_lap, snap.own.compound, snap.own.must_stop)
        self._note_call_change(kind, chosen, lap, snap)
        ahead = snap.effective_gap < 0
        if self._was_ahead is not None and ahead != self._was_ahead:
            self._alert("WARN", f"Now {'AHEAD of' if ahead else 'BEHIND'} {self.opp} on effective gap")
        self._was_ahead = ahead

    def _note_call_change(self, kind: str, chosen: Option, lap: int, snap) -> None:
        missed = self._last_box_call
        if missed is not None and lap >= missed:
            stopped = self.field.stops(self.own, missed) > self.field.stops(self.own, missed - 1)
            if not stopped and not self._miss_noted:
                self._alert("WARN", f"Box call for lap {missed} was not taken")
                self._miss_noted = True
            if stopped:
                self._miss_noted = False
        if kind != "BOX":
            self._miss_noted = False
        self._last_box_call = chosen.box_lap if kind == "BOX" else None
        if kind != self._prev_call_kind and kind in ("BOX", "STANDBY", "STAY"):
            level = "CRIT" if kind == "BOX" else "INFO"
            words = {"BOX": f"CALL: box this lap ({chosen.box_lap})",
                     "STANDBY": f"CALL: stand by, box end of lap {chosen.box_lap}",
                     "STAY": "CALL: stay out"}
            self._alert(level, words[kind])
        self._prev_call_kind = kind

    def _fill_trend(self, frame: Frame, lap: int) -> None:
        own, opp = self.field.cars[self.own], self.field.cars[self.opp]
        gaps = []
        for n in range(lap, lap - 5, -1):
            a, b = own.get(n), opp.get(n)
            if (a is None or b is None or a.pit_in or a.pit_out or b.pit_in or b.pit_out
                    or a.neutral or b.neutral):
                break
            gaps.append(a.t - b.t)
        frame.gap_trend = (gaps[0] - gaps[-1]) / (len(gaps) - 1) if len(gaps) >= 3 else None
        a, b = self.field.pace(self.own, n=3), self.field.pace(self.opp, n=3)
        frame.pace_delta = a - b if a is not None and b is not None else None

    # -- output --------------------------------------------------------------------
    def _panel(self, driver: str, now: float, old: CarPanel | None) -> CarPanel:
        laps = self.field.cars.get(driver)
        if not laps:
            return CarPanel(driver)
        last = max(laps)
        r = laps[last]
        panel = CarPanel(driver, self.field.position(driver, now), last, r.compound, r.age,
                         r.lap_time, self.field.pace(driver), self.field.stops(driver), None, None,
                         r.pit_in, False, self.field.is_running(driver, now))
        if old is not None and old.driver == driver:
            panel.penalty, panel.slope, panel.must_stop = old.penalty, old.slope, old.must_stop
        return panel

    def _retarget_if_lost(self, now: float) -> None:
        if not self.field.cars.get(self.opp) or self.chequered:
            return
        if self.field.is_running(self.opp, now) or not self.field.is_running(self.own, now):
            return
        order = [d for d in self.field.order(now) if self.field.is_running(d, now) and d != self.own]
        if not order:
            return
        lost = self.opp
        mine = self.field.position(self.own, now)
        ahead = [d for d in order if self.field.position(d, now) < mine]
        self.set_opponent(ahead[-1] if ahead else order[0])
        self._alert("CRIT", f"{lost} has stopped running. Target is now {self.opp}")

    def tick(self, now: float, health: FeedHealth | None = None) -> Frame:
        """Bring the frame up to ``now``. Cheap when nothing has changed."""
        f = self.frame
        self._retarget_if_lost(now)
        f.now, f.flag, f.health = now, self.flag_kind, health
        f.own_lap = self.field.last_lap(self.own)
        f.own = self._panel(self.own, now, f.own)
        f.rival = self._panel(self.opp, now, f.rival)
        f.field_rows = self.field.table(self.own, now) if self.field.cars.get(self.own) else []
        f.alerts = self.alerts
        if self._errors:
            f.pace_error, f.pace_error_n = float(np.mean(self._errors)), len(self._errors)
        f.rival_stops_seen, f.rival_stops_warned = self._rival_stops, self._rival_warned
        if self.field.cars.get(self.own):
            pace = self.field.pace(self.own)
            line = self.field.line_time(self.own, f.own_lap)
            f.seconds_to_line = max(0.0, line + pace - now) if pace else None
        f.finished = self.chequered and f.own_lap >= self.info.laps_total or (
            self.chequered and self.field.is_running(self.own, now) is False)
        if self.chequered and f.own and self.field.cars.get(self.own):
            leader_done = [d for d, laps in self.field.cars.items()
                           if self.info.laps_total in laps]
            if leader_done and self.field.line_time(self.own, f.own_lap) >= min(
                    self.field.line_time(d, self.info.laps_total) for d in leader_done):
                f.finished = True
        if f.finished:
            f.call = Call("NONE", "CHEQUERED FLAG", "", self._result_text())
            f.options, f.tyre_note = [], ""
            return f
        if self.field.cars.get(self.own) and not self.field.is_running(self.own, now):
            f.call = Call("NONE", f"{self.own} HAS STOPPED RUNNING", "", "No laps coming in.")
            f.options = []
            return f
        if self.flag_kind == "RED":
            f.call = Call("NONE", "RED FLAG  -  NO CALL",
                          "Red flag. Slowly back to the pit lane.",
                          "Tyres can normally be changed under a red flag at no time cost; "
                          "the simulator does not model the restart.")
            f.options, f.tyre_note = [], ""
            return f
        ready = self._ready_lap(now)
        if ready is None:
            if not self._decided:
                f.call = Call("NONE", "NO CALL YET", "", "Waiting for both cars to set a lap.")
            return f
        lap, projected = ready
        if not self._dirty and self._decided == (lap, projected, self.opp):
            return f
        started = time.perf_counter()
        try:
            self._decide(now, lap, projected)
            f.stale_reason = ""
        except Exception as e:  # keep the last call on screen rather than dying mid-race
            f.stale_reason = f"calculation failed, showing lap {f.lap}: {type(e).__name__}: {e}"
            f.error_trace = traceback.format_exc()
        f.calc_ms = (time.perf_counter() - started) * 1000.0
        f.decision_id = self.decision_id
        self._decided = (lap, projected, self.opp)
        self._dirty = self._force = False
        return f

    def _result_text(self) -> str:
        gap = self.field.gap_to(self.own, self.opp)
        if gap is None:
            return "Race over."
        word = "ahead of" if gap < 0 else "behind"
        return (f"{self.own} finished P{self.field.position(self.own)}, {abs(gap):.1f}s {word} "
                f"{self.opp} (P{self.field.position(self.opp)}) on the last lap both completed.")


