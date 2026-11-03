"""The whole field: order, gaps, measured stop cost and where a stop would rejoin.

The 1v1 simulator only knows two cars. The question on the pit wall is also "who am
I behind when I come out", and that needs all of them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .live_feed import LapEvent

TRAFFIC_S = 2.0      # another car this close ahead on the road will cost us time
RETIRED_LAPS = 3.0   # no line crossing for this many laps means the car has stopped
MIN_FIELD_STOPS = 3  # measured stops needed before trusting them over the prior


@dataclass
class CarLap:
    t: float
    lap_time: float
    compound: str
    age: int
    pit_in: bool
    pit_out: bool
    neutral: bool


@dataclass
class Rejoin:
    box_lap: int
    position: int
    ahead: str | None        # car ahead in race order after the stop
    ahead_gap: float | None
    behind: str | None
    behind_gap: float | None
    road_ahead: str | None   # nearest car in front on the road, maybe a lapped one
    road_gap: float | None
    ahead_to_stop: int = 0   # cars ahead of us then that have not made a stop yet

    @property
    def traffic(self) -> bool:
        return self.road_gap is not None and self.road_gap <= TRAFFIC_S


@dataclass
class CarRow:
    position: int
    driver: str
    laps: int
    gap: float | None        # to the reference car; + = this car is behind it
    compound: str
    age: int
    last_lap: float
    stops: int
    in_pit: bool
    running: bool


class FieldState:
    def __init__(self):
        self.cars: dict[str, dict[int, CarLap]] = {}
        self.stop_costs: list[float] = []
        self._scored: set[tuple[str, int]] = set()

    # -- intake --------------------------------------------------------------------
    def add(self, e: LapEvent) -> None:
        laps = self.cars.setdefault(e.driver, {})
        prev = laps.get(e.lap - 1)
        if e.tyre_age is not None:
            age = max(int(e.tyre_age), 1)
        elif prev is None or prev.pit_in:
            age = 1
        else:
            age = prev.age + 1
        laps[e.lap] = CarLap(e.t, e.lap_time, e.compound, age, e.pit_in, e.pit_out, e.neutral)
        self._score_stop(e.driver, e.lap - 1)

    def mark_in_lap(self, driver: str, lap: int) -> None:
        rec = self.cars.get(driver, {}).get(lap)
        if rec is not None:
            rec.pit_in = True

    def _score_stop(self, driver: str, in_lap: int) -> None:
        """Once a car finishes its out-lap, measure what the stop cost against its pace."""
        laps = self.cars[driver]
        i, o = laps.get(in_lap), laps.get(in_lap + 1)
        if (i is None or o is None or not i.pit_in or i.neutral or o.neutral
                or (driver, in_lap) in self._scored):
            return
        self._scored.add((driver, in_lap))
        ref = self.pace(driver, before=in_lap)
        if ref is not None and np.isfinite(i.lap_time) and np.isfinite(o.lap_time):
            cost = i.lap_time + o.lap_time - 2.0 * ref
            if 10.0 < cost < 45.0:
                self.stop_costs.append(float(cost))

    # -- per car -------------------------------------------------------------------
    def last_lap(self, driver: str) -> int:
        laps = self.cars.get(driver)
        return max(laps) if laps else 0

    def line_time(self, driver: str, lap: int) -> float | None:
        rec = self.cars.get(driver, {}).get(lap)
        return rec.t if rec else None

    def pace(self, driver: str, before: int | None = None, n: int = 5) -> float | None:
        """Median of the last ``n`` clean laps."""
        laps = self.cars.get(driver, {})
        out = []
        for lap in sorted(laps, reverse=True):
            r = laps[lap]
            if before is not None and lap >= before:
                continue
            if (lap > 1 and not r.pit_in and not r.pit_out and not r.neutral
                    and np.isfinite(r.lap_time)):
                out.append(r.lap_time)
                if len(out) == n:
                    break
        return float(np.median(out)) if len(out) >= 2 else None

    def stops(self, driver: str, up_to: int | None = None) -> int:
        return sum(1 for lap, r in self.cars.get(driver, {}).items()
                   if r.pit_in and (up_to is None or lap <= up_to))

    def is_running(self, driver: str, now: float) -> bool:
        laps = self.cars.get(driver)
        if not laps:
            return False
        pace = self.pace(driver) or 100.0
        return now - laps[max(laps)].t <= RETIRED_LAPS * pace

    def pit_loss(self, prior: float) -> float:
        if len(self.stop_costs) >= MIN_FIELD_STOPS:
            return float(np.median(self.stop_costs))
        return prior

    # -- order ---------------------------------------------------------------------
    def order(self, now: float | None = None) -> list[str]:
        """Race order: most laps first, then whoever finished that lap earliest."""
        running = []
        stopped = []
        for d, laps in self.cars.items():
            if not laps:
                continue
            last = max(laps)
            key = (-last, laps[last].t)
            (running if now is None or self.is_running(d, now) else stopped).append((key, d))
        return [d for _, d in sorted(running)] + [d for _, d in sorted(stopped)]

    def position(self, driver: str, now: float | None = None) -> int:
        order = self.order(now)
        return order.index(driver) + 1 if driver in order else 0

    def gap_to(self, driver: str, reference: str) -> float | None:
        """Time at the line on the last lap both finished. + = driver is behind."""
        a, b = self.cars.get(driver), self.cars.get(reference)
        if not a or not b:
            return None
        common = min(max(a), max(b))
        while common > 0 and (common not in a or common not in b):
            common -= 1
        return a[common].t - b[common].t if common else None

    def table(self, reference: str, now: float) -> list[CarRow]:
        rows = []
        for i, d in enumerate(self.order(now), 1):
            laps = self.cars[d]
            r = laps[max(laps)]
            rows.append(CarRow(i, d, max(laps), self.gap_to(d, reference), r.compound, r.age,
                               r.lap_time, self.stops(d), r.pit_in, self.is_running(d, now)))
        return rows

    # -- rejoin --------------------------------------------------------------------
    def rejoin(self, driver: str, box_lap: int, stop_cost: float, now: float) -> Rejoin | None:
        """Where we would be at the end of the out-lap if we box at the end of ``box_lap``.

        Every other car holds its recent pace and makes no stop. The stop cost already
        includes the slow out-lap, so this is measured at that lap's end.
        """
        laps = self.cars.get(driver)
        if not laps:
            return None
        cur = max(laps)
        pace = self.pace(driver)
        if pace is None:
            return None
        out_lap = box_lap + 1
        t_own = laps[cur].t + (out_lap - cur) * pace + stop_cost
        at_flag: list[tuple[float, str]] = []
        road_ahead, road_gap = None, None
        for d, other in self.cars.items():
            if d == driver or not self.is_running(d, now):
                continue
            last = max(other)
            p = self.pace(d) or pace
            t_same_lap = other[last].t + (out_lap - last) * p
            at_flag.append((t_same_lap - t_own, d))
            # Road position: where this car is, whatever lap it is on.
            n = round((t_own - other[last].t) / p)
            crossing = other[last].t + n * p
            ahead_by = t_own - crossing          # this car crossed that long before us
            if ahead_by < 0:
                ahead_by += p
            if road_gap is None or ahead_by < road_gap:
                road_ahead, road_gap = d, float(ahead_by)
        in_front = sorted(((g, d) for g, d in at_flag if g < 0), reverse=True)
        behind = sorted((g, d) for g, d in at_flag if g >= 0)
        return Rejoin(box_lap, len(in_front) + 1,
                      in_front[0][1] if in_front else None,
                      float(-in_front[0][0]) if in_front else None,
                      behind[0][1] if behind else None,
                      float(behind[0][0]) if behind else None,
                      road_ahead, road_gap,
                      sum(1 for _, d in in_front if self.stops(d) == 0))
