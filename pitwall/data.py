"""Race data: synthetic generator, bundled real races, FastF1 loader, CSV cache.

Every source produces LapRecords with the same conventions (see ``state.py``).
"""
from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .state import PIT_IN_SHARE, LapRecord, Replay

COMPOUND_PACE = {"SOFT": -0.6, "MEDIUM": 0.0, "HARD": 0.5}
COMPOUND_DEG = {"SOFT": 0.12, "MEDIUM": 0.08, "HARD": 0.05}
DEFAULT_PIT_LOSS = 22.0  # prior until this race has shown a stop
BUNDLE = Path(__file__).resolve().parent / "assets" / "openf1_laps.csv.gz"


def true_lap_time(base: float, compound: str, age: int, lap: int, laps_total: int) -> float:
    """Noise-free generator physics. Models never see this function."""
    fuel = 1.2 * (laps_total - lap) / laps_total
    return base + COMPOUND_PACE[compound] + COMPOUND_DEG[compound] * age + 0.004 * age ** 2 + fuel


def true_wear(compound: str, age: int) -> float:
    return COMPOUND_DEG[compound] * age + 0.004 * age ** 2


def generate_race(seed: int = 7, laps_total: int = 40, pit_loss: float = 21.0,
                  opp_pit_laps: tuple = (14, 28), own_pit_laps: tuple = (15, 29),
                  base_own: float = 93.0, base_opp: float = 92.9,
                  start_gap: float = 1.5,
                  compounds: tuple = ("MEDIUM", "HARD", "SOFT")) -> Replay:
    """Build a deterministic SYNTHETIC race.

    A lap in pit_laps is the in-lap: it is driven on the old tyre and carries the
    in-lap share of the stop; the next lap is the out-lap and carries the rest,
    as real timing data does. The gap is the running sum of the lap times.

    The own car also gets a SYNTHETIC private wear signal: the true wear plus
    small noise. It stands in for team telemetry and is never measured data.
    """
    rng = np.random.default_rng(seed)
    own_recs, opp_recs = [], []
    for car, pits, base, recs in (("OWN", own_pit_laps, base_own, own_recs),
                                  ("OPP", opp_pit_laps, base_opp, opp_recs)):
        age, comp_idx, out_lap = 0, 0, False
        for lap_no in range(1, laps_total + 1):
            age += 1
            comp = compounds[min(comp_idx, len(compounds) - 1)]
            lt = true_lap_time(base, comp, age, lap_no, laps_total) + rng.normal(0, 0.25)
            is_pit = lap_no in pits
            if is_pit:
                lt += PIT_IN_SHARE * pit_loss
            if out_lap:
                lt += (1.0 - PIT_IN_SHARE) * pit_loss
            wear = None
            if car == "OWN":
                wear = true_wear(comp, age) + float(rng.normal(0, 0.05))
            recs.append(LapRecord(
                lap=lap_no, driver=car, compound=comp, tyre_age=age, lap_time=float(lt),
                track_temp=38.0 + 2 * float(np.sin(lap_no / 8)), pitted=is_pit,
                pit_out=out_lap, private_wear=wear))
            out_lap = is_pit
            if is_pit:
                age, comp_idx = 0, comp_idx + 1
    gap = start_gap
    for o, p in zip(own_recs, opp_recs):
        gap += o.lap_time - p.lap_time
        o.gap_to_rival, p.gap_to_rival = gap, -gap
    return Replay(laps_total, own_recs, opp_recs, pit_loss)


# ---------------------------------------------------------------------------
# Frames -> records. Pure mapping, shared by the bundled data and FastF1.


def records_from_frames(own, opp, weather_time=None, weather_temp=None,
                        default_temp=float("nan"), default_pit_loss=DEFAULT_PIT_LOSS,
                        laps_total=None):
    """Normalized frames -> (laps_total, own_recs, opp_recs, pit_loss prior).

    Times stay raw: in-laps, out-laps and missing laps are flagged, never edited,
    and no statistic from later laps is used. ``laps_total`` should be the official
    race distance; without it, the longer of the two drivers' races is used.
    """
    laps = sorted(set(own["lap"]) & set(opp["lap"]))
    if not laps:
        raise ValueError("no overlapping laps for the two drivers")
    own_by = {int(r["lap"]): r for _, r in own.iterrows()}
    opp_by = {int(r["lap"]): r for _, r in opp.iterrows()}

    def temp_at(start):
        if weather_time is None or weather_temp is None or not len(weather_time):
            return default_temp
        wt = np.asarray(weather_time, float)
        temps = np.asarray(weather_temp, float)
        i = int(np.searchsorted(wt, float(start), side="right")) - 1
        for j in range(min(i, len(temps) - 1), -1, -1):  # latest valid reading so far
            if np.isfinite(temps[j]):
                return float(temps[j])
        return default_temp

    def age_of(r, hist):
        a = float(r["age"])
        if np.isfinite(a):
            return max(int(round(a)), 1)
        if not hist or hist[-1].pitted:
            return 1
        return hist[-1].tyre_age + 1

    def line_time(by, lap):
        """When this car crossed the line at the end of ``lap``."""
        r, nxt = by[lap], by.get(lap + 1)
        if nxt is not None and np.isfinite(float(nxt["start"])):
            return float(nxt["start"])
        return float(r["start"]) + float(r["time"])

    def neutral(r):
        return any(c in str(r["status"]) for c in "4567")

    own_recs, opp_recs = [], []
    prev_gap, prev_pos = 0.0, {"OWN": 0, "OPP": 0}

    def pos_of(r, role):
        p = float(r["pos"])
        if np.isfinite(p):
            prev_pos[role] = int(p)
        return prev_pos[role]

    for l in laps:
        ro, rp = own_by[l], opp_by[l]
        gap = line_time(own_by, l) - line_time(opp_by, l)
        gap = gap if np.isfinite(gap) else prev_gap
        prev_gap = gap
        so = float(ro["start"])
        temp = temp_at(so) if np.isfinite(so) else default_temp
        own_recs.append(LapRecord(
            lap=l, driver="OWN", compound=str(ro["compound"]),
            tyre_age=age_of(ro, own_recs), lap_time=float(ro["time"]),
            position=pos_of(ro, "OWN"), gap_to_rival=gap, track_temp=temp,
            safety_car=neutral(ro), pitted=bool(ro["pit_in"]), pit_out=bool(ro["pit_out"])))
        opp_recs.append(LapRecord(
            lap=l, driver="OPP", compound=str(rp["compound"]),
            tyre_age=age_of(rp, opp_recs), lap_time=float(rp["time"]),
            position=pos_of(rp, "OPP"), gap_to_rival=-gap, track_temp=temp,
            safety_car=neutral(rp), pitted=bool(rp["pit_in"]), pit_out=bool(rp["pit_out"])))
    if laps_total is None:
        laps_total = int(max(max(own["lap"]), max(opp["lap"])))
    return int(laps_total), own_recs, opp_recs, float(default_pit_loss)


# ---------------------------------------------------------------------------
# Bundled real races (OpenF1, see openf1.py). Offline and needs no login.


@lru_cache(maxsize=1)
def bundled_laps():
    import pandas as pd

    df = pd.read_csv(BUNDLE)
    df["status"] = df["neutral"].map({True: "4", False: "1"})
    df["acronym"] = df["acronym"].astype(str)
    return df


def has_bundled_data() -> bool:
    return BUNDLE.exists()


def list_bundled_races() -> list[tuple[int, str]]:
    df = bundled_laps()
    order = df.groupby(["year", "circuit"], sort=False)["start"].min().sort_values()
    return [(int(y), str(c)) for y, c in order.index]


def bundled_race(year: int, circuit: str):
    df = bundled_laps()
    return df[(df["year"] == year) & (df["circuit"] == circuit)]


def bundled_drivers(year: int, circuit: str) -> list[str]:
    race = bundled_race(year, circuit)
    return race[race["lap"] == 1].sort_values("pos")["acronym"].tolist() \
        or sorted(race["acronym"].unique())


def driver_frame(race, acronym: str):
    frame = race[race["acronym"] == acronym].sort_values("lap").reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"no laps for driver {acronym}")
    return frame


def load_bundled_replay(year: int, circuit: str, own: str, opp: str) -> Replay:
    race = bundled_race(year, circuit)
    if race.empty:
        raise ValueError(f"race not bundled: {year} {circuit}")
    laps_total, own_r, opp_r, prior = records_from_frames(
        driver_frame(race, own), driver_frame(race, opp),
        laps_total=int(race["total_laps"].iloc[0]))
    return Replay(laps_total, own_r, opp_r, prior)


# ---------------------------------------------------------------------------
# FastF1 (optional: current FastF1 needs an F1 account via F1_USERNAME/F1_PASSWORD).


def default_cache_dir() -> Path:
    return Path(__file__).resolve().parents[1] / ".data_cache"


def normalize_fastf1_frame(df):
    """FastF1 laps frame -> normalized columns (pure, no network)."""
    import pandas as pd

    need = {"LapNumber", "Compound", "TyreLife", "LapTime", "Position",
            "TrackStatus", "PitInTime", "PitOutTime", "LapStartTime"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"missing FastF1 columns: {sorted(missing)}")
    out = pd.DataFrame({
        "lap": df["LapNumber"].astype(int),
        "compound": df["Compound"].fillna("UNKNOWN").astype(str).str.upper(),
        "age": pd.to_numeric(df["TyreLife"], errors="coerce"),
        "time": pd.to_timedelta(df["LapTime"]).dt.total_seconds(),
        "pos": pd.to_numeric(df["Position"], errors="coerce"),
        "status": df["TrackStatus"].fillna("1").astype(str),
        "pit_in": df["PitInTime"].notna(),
        "pit_out": df["PitOutTime"].notna(),
        "start": pd.to_timedelta(df["LapStartTime"]).dt.total_seconds(),
    })
    out = out.sort_values("lap").reset_index(drop=True)
    out.loc[out["lap"] == out["lap"].min(), "pit_out"] = False  # leaving the grid
    return out


def _cache_key(year, gp, own_drv, opp_drv) -> str:
    slug = lambda s: "".join(c if c.isalnum() else "_" for c in str(s)).lower()
    return f"fastf1_{year}_{slug(gp)}_{slug(own_drv)}_vs_{slug(opp_drv)}"


def save_replay_csv(replay: Replay, pit_loss: float, meta: dict, csv_path) -> None:
    last = max(replay.available_laps())
    snap = replay.state_at(last)
    recs = list(snap.history_own) + list(snap.history_opp)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lap", "driver", "compound", "tyre_age", "lap_time",
                    "position", "gap_to_rival", "track_temp", "safety_car",
                    "pitted", "pit_out"])
        for r in recs:
            w.writerow([r.lap, r.driver, r.compound, r.tyre_age,
                        f"{r.lap_time:.3f}", r.position,
                        f"{r.gap_to_rival:.3f}", f"{r.track_temp:.2f}",
                        int(r.safety_car), int(r.pitted), int(r.pit_out)])
    Path(str(csv_path)).with_suffix(".json").write_text(
        json.dumps({**meta, "laps_total": replay.laps_total,
                    "pit_loss": pit_loss}))


def load_replay_csv(csv_path) -> Replay:
    meta = json.loads(Path(str(csv_path)).with_suffix(".json").read_text())
    own, opp = [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            r = LapRecord(
                lap=int(row["lap"]), driver=row["driver"],
                compound=row["compound"], tyre_age=int(row["tyre_age"]),
                lap_time=float(row["lap_time"]), position=int(row["position"]),
                gap_to_rival=float(row["gap_to_rival"]),
                track_temp=float(row["track_temp"]),
                safety_car=bool(int(row["safety_car"])),
                pitted=bool(int(row["pitted"])),
                pit_out=bool(int(row.get("pit_out", 0) or 0)))
            (own if r.driver == "OWN" else opp).append(r)
    return Replay(int(meta["laps_total"]), own, opp, float(meta["pit_loss"]))


def load_fastf1_replay(year, gp, own_drv, opp_drv, cache_dir=None,
                       offline_ok=True) -> Replay:
    """Fetch a real race via FastF1 (needs network + F1 credentials).

    Falls back to the CSV cache when offline_ok and an entry exists, else re-raises.
    """
    if own_drv == opp_drv:
        raise ValueError("pick two different drivers")
    cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()
    (cache_dir / "fastf1").mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / (_cache_key(year, gp, own_drv, opp_drv) + ".csv")
    try:
        import fastf1

        fastf1.Cache.configure(cache_dir=str(cache_dir / "fastf1"))
        s = fastf1.get_session(year, gp, "R")
        s.load(laps=True, weather=True, telemetry=False)
        laps = s.laps.pick_drivers([own_drv, opp_drv])
        if laps.empty:
            raise ValueError("FastF1 returned no laps for those drivers")
        own_n = normalize_fastf1_frame(laps.pick_drivers([own_drv]))
        opp_n = normalize_fastf1_frame(laps.pick_drivers([opp_drv]))
        wt, wtemp = None, None
        try:
            import pandas as pd

            w = s.weather_data
            wt = pd.to_timedelta(w["Time"]).dt.total_seconds().to_numpy()
            wtemp = pd.to_numeric(w["TrackTemp"], errors="coerce").to_numpy()
        except Exception:
            pass
        official = getattr(s, "total_laps", None)
        laps_total, own_r, opp_r, prior = records_from_frames(
            own_n, opp_n, wt, wtemp, laps_total=int(official) if official else None)
        replay = Replay(laps_total, own_r, opp_r, prior)
        save_replay_csv(replay, prior,
                        {"year": year, "gp": gp, "own": own_drv, "opp": opp_drv},
                        csv_path)
        return replay
    except Exception:
        if (offline_ok and csv_path.exists()
                and csv_path.with_suffix(".json").exists()):
            return load_replay_csv(csv_path)
        raise
