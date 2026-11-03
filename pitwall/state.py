"""Shared race-state representation (Module 1 output, consumed by all modules).

Timing convention: a snapshot for lap ``t`` describes the instant the car
crosses the line at the END of lap ``t``. Every prediction made from it is
about lap ``t + 1`` or later.

Pit-stop timing convention (matches real timing data): the timing line sits
near the pit box, so a stop costs a little on the in-lap and most of its time
on the out-lap. ``lap_time`` is always the raw time of that lap.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Measured on 12 real stops (2023 Bahrain and Spain, 2024 Hungary): the in-lap
# carries about 15% of the loss and the out-lap the rest.
PIT_IN_SHARE = 0.15
WET_COMPOUNDS = {"INTERMEDIATE", "WET"}
DRY_COMPOUNDS = {"SOFT", "MEDIUM", "HARD"}


@dataclass
class LapRecord:
    lap: int
    driver: str
    compound: str          # SOFT / MEDIUM / HARD / INTERMEDIATE / WET / UNKNOWN
    tyre_age: int          # laps on this tyre including this lap (1 = fresh)
    lap_time: float        # raw seconds, NaN if not recorded
    position: int = 0
    gap_to_rival: float = 0.0   # at END of this lap; + = this driver is BEHIND
    track_temp: float = 35.0
    safety_car: bool = False    # safety car, VSC or red flag during this lap
    pitted: bool = False   # pitted at END of this lap (this is the in-lap)
    pit_out: bool = False  # this lap is the out-lap of a stop
    private_wear: float | None = None  # SYNTHETIC, own car only


@dataclass
class CarSnapshot:
    driver: str
    compound: str
    tyre_age: int
    last_lap_time: float
    stops_done: int        # stops completed up to and including the end of this lap
    just_pitted: bool = False   # boxed at the end of this lap
    must_stop: bool = False     # dry race and has not used a second dry compound
    private_wear: float | None = None  # SYNTHETIC own-car-only signal, never set for opponent

    @property
    def next_age(self) -> int:
        """Tyre age on the next lap: a car that just boxed starts over."""
        return 1 if self.just_pitted else self.tyre_age + 1


@dataclass
class RaceSnapshot:
    """Everything any module may use at decision point 'lap' (all data <= lap)."""
    lap: int
    laps_total: int
    own: CarSnapshot
    opp: CarSnapshot
    gap_own_behind_opp: float  # measured at the line, + = we trail the opponent
    track_temp: float = 35.0
    safety_car: bool = False
    pit_loss: float = 21.0     # known at this lap: prior, then observed stops
    history_own: list = field(default_factory=list)  # LapRecords with lap <= current
    history_opp: list = field(default_factory=list)

    @property
    def laps_left(self) -> int:
        return self.laps_total - self.lap

    @property
    def effective_gap(self) -> float:
        """Gap once any stop in progress has been paid for.

        A car that just boxed has only paid the in-lap share, so the line gap
        flatters it by the remaining out-lap share.
        """
        pending = (1.0 - PIT_IN_SHARE) * self.pit_loss
        return (self.gap_own_behind_opp
                + (pending if self.own.just_pitted else 0.0)
                - (pending if self.opp.just_pitted else 0.0))


def must_stop(history: list[LapRecord]) -> bool:
    """Dry-race rule: two different dry compounds are required.

    Returns False when the rule cannot be applied (wet running, unknown tyres).
    """
    compounds = {r.compound for r in history}
    if compounds & WET_COMPOUNDS or not compounds <= DRY_COMPOUNDS:
        return False
    return len(compounds) < 2 and not any(r.pitted for r in history)


def observed_pit_loss(history: list[LapRecord], lap: int) -> list[float]:
    """Stop costs measurable from one car's laps up to ``lap`` (no future)."""
    clean = [r.lap_time for r in history
             if not r.pitted and not r.pit_out and not r.safety_car
             and r.lap > 1 and np.isfinite(r.lap_time)]
    if len(clean) < 3:
        return []
    ref = float(np.median(clean))
    by_lap = {r.lap: r for r in history}
    costs = []
    for r in history:
        out = by_lap.get(r.lap + 1)
        if (not r.pitted or out is None or out.lap > lap
                or r.safety_car or out.safety_car):
            continue
        if np.isfinite(r.lap_time) and np.isfinite(out.lap_time):
            cost = r.lap_time + out.lap_time - 2.0 * ref
            if 10.0 < cost < 45.0:  # anything else is not a normal green-flag stop
                costs.append(cost)
    return costs


class Replay:
    """Lap-by-lap replayer with no-future-leakage guarantee (Spec §4)."""

    def __init__(self, laps_total: int, own_laps: list[LapRecord],
                 opp_laps: list[LapRecord], pit_loss: float = 21.0):
        self.laps_total = laps_total
        self._own = {r.lap: r for r in own_laps}
        self._opp = {r.lap: r for r in opp_laps}
        self._own_list = sorted(own_laps, key=lambda r: r.lap)
        self._opp_list = sorted(opp_laps, key=lambda r: r.lap)
        self.pit_loss = pit_loss  # prior, used until a stop has been observed

    def available_laps(self) -> list[int]:
        return sorted(set(self._own) & set(self._opp))

    def state_at(self, lap: int, include_private_wear: bool = True) -> RaceSnapshot:
        """Build snapshot using ONLY laps <= `lap`. Raises if lap not replayed yet."""
        if lap not in self._own or lap not in self._opp:
            raise KeyError(f"lap {lap} not available for both cars")
        ho = [r for r in self._own_list if r.lap <= lap]
        hp = [r for r in self._opp_list if r.lap <= lap]
        o, p = self._own[lap], self._opp[lap]
        own = CarSnapshot(o.driver, o.compound, o.tyre_age, o.lap_time,
                          sum(1 for r in ho if r.pitted), just_pitted=o.pitted,
                          must_stop=must_stop(ho),
                          private_wear=(o.private_wear if include_private_wear else None))
        opp = CarSnapshot(p.driver, p.compound, p.tyre_age, p.lap_time,
                          sum(1 for r in hp if r.pitted), just_pitted=p.pitted,
                          must_stop=must_stop(hp),
                          private_wear=None)  # asymmetric info: never for opponent
        costs = observed_pit_loss(ho, lap) + observed_pit_loss(hp, lap)
        return RaceSnapshot(lap, self.laps_total, own, opp,
                            gap_own_behind_opp=o.gap_to_rival,
                            track_temp=o.track_temp, safety_car=o.safety_car,
                            pit_loss=float(np.median(costs)) if costs else self.pit_loss,
                            history_own=ho, history_opp=hp)
