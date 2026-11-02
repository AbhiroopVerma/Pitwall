"""Shared race-state representation (Module 1 output, consumed by all modules)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LapRecord:
    lap: int
    driver: str
    compound: str          # SOFT / MEDIUM / HARD
    tyre_age: int          # laps on current tyre at START of this lap (1 = fresh)
    lap_time: float        # seconds, includes pit loss if pitted this lap
    position: int = 0
    gap_to_rival: float = 0.0   # + means this driver is BEHIND rival (seconds)
    track_temp: float = 35.0
    safety_car: bool = False
    pitted: bool = False   # pitted at END of this lap


@dataclass
class CarSnapshot:
    driver: str
    compound: str
    tyre_age: int
    last_lap_time: float
    stops_done: int
    private_wear: float | None = None  # SYNTHETIC own-car-only signal, never set for opponent


@dataclass
class RaceSnapshot:
    """Everything any module may use at decision point 'lap' (all data <= lap)."""
    lap: int
    laps_total: int
    own: CarSnapshot
    opp: CarSnapshot
    gap_own_behind_opp: float  # + = we trail the opponent
    track_temp: float = 35.0
    safety_car: bool = False
    pit_loss: float = 21.0
    history_own: list = field(default_factory=list)  # LapRecords with lap <= current
    history_opp: list = field(default_factory=list)


class Replay:
    """Lap-by-lap replayer with no-future-leakage guarantee (Spec §4)."""

    def __init__(self, laps_total: int, own_laps: list[LapRecord],
                 opp_laps: list[LapRecord], pit_loss: float = 21.0):
        self.laps_total = laps_total
        self._own = {r.lap: r for r in own_laps}
        self._opp = {r.lap: r for r in opp_laps}
        self._own_list = sorted(own_laps, key=lambda r: r.lap)
        self._opp_list = sorted(opp_laps, key=lambda r: r.lap)
        self.pit_loss = pit_loss

    def available_laps(self) -> list[int]:
        return sorted(set(self._own) & set(self._opp))

    def state_at(self, lap: int, include_private_wear: bool = True) -> RaceSnapshot:
        """Build snapshot using ONLY laps <= `lap`. Raises if lap not replayed yet."""
        if lap not in self._own or lap not in self._opp:
            raise KeyError(f"lap {lap} not available for both cars")
        ho = [r for r in self._own_list if r.lap <= lap]
        hp = [r for r in self._opp_list if r.lap <= lap]
        o, p = self._own[lap], self._opp[lap]
        stops_own = sum(1 for r in ho if r.pitted)
        stops_opp = sum(1 for r in hp if r.pitted)
        own = CarSnapshot(o.driver, o.compound, o.tyre_age, o.lap_time,
                          stops_own, private_wear=(o.tyre_age * 1.0 if include_private_wear else None))
        opp = CarSnapshot(p.driver, p.compound, p.tyre_age, p.lap_time,
                          stops_opp, private_wear=None)  # asymmetric info: never for opponent
        return RaceSnapshot(lap, self.laps_total, own, opp,
                            gap_own_behind_opp=o.gap_to_rival,
                            track_temp=o.track_temp, safety_car=o.safety_car,
                            pit_loss=self.pit_loss,
                            history_own=ho, history_opp=hp)
