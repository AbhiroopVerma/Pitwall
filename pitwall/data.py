"""Synthetic race generator + replay plumbing (Module 1).

Real FastF1/OpenF1 wiring plugs in here later: any loader only needs to
return (laps_total, own_laps, opp_laps) as LapRecord lists. The demo uses a
clearly-labelled SYNTHETIC generator so the prototype runs offline.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .state import LapRecord, Replay

COMPOUND_PACE = {"SOFT": -0.6, "MEDIUM": 0.0, "HARD": 0.5}
COMPOUND_DEG = {"SOFT": 0.12, "MEDIUM": 0.08, "HARD": 0.05}


def generate_race(seed: int = 7, laps_total: int = 40, pit_loss: float = 21.0,
                  opp_pit_laps: tuple = (14, 28), own_pit_laps: tuple = (15, 29),
                  base_own: float = 93.0, base_opp: float = 92.9) -> Replay:
    """Build a deterministic 2-stop synthetic race.

    A lap listed in pit_laps is driven on the OLD tyre and its time includes
    pit_loss (pit at END of that lap); the next lap starts a fresh stint.
    Exactly laps_total laps per car are produced.
    """
    rng = np.random.default_rng(seed)
    comps = ["MEDIUM", "HARD", "SOFT"]
    own_recs, opp_recs = [], []
    gap = 1.5  # we start 1.5 s behind
    for car, pits, base, recs in (("OWN", own_pit_laps, base_own, own_recs),
                                  ("OPP", opp_pit_laps, base_opp, opp_recs)):
        age, comp_idx = 0, 0
        for lap_no in range(1, laps_total + 1):
            age += 1
            comp = comps[min(comp_idx, 2)]
            fuel = 1.2 * (laps_total - lap_no) / laps_total
            deg = COMPOUND_DEG[comp] * age + 0.004 * age ** 2
            lt = base + COMPOUND_PACE[comp] + deg + fuel + rng.normal(0, 0.25)
            is_pit = lap_no in pits
            if is_pit:
                lt += pit_loss
            gap += float(rng.normal(0.05, 0.3))
            recs.append(LapRecord(
                lap=lap_no, driver=car, compound=comp, tyre_age=age,
                lap_time=lt, gap_to_rival=(gap if car == "OWN" else -gap),
                track_temp=38.0 + 2 * np.sin(lap_no / 8), pitted=is_pit))
            if is_pit:
                age, comp_idx = 0, comp_idx + 1
    for o, p in zip(own_recs, opp_recs):
        p.gap_to_rival = -o.gap_to_rival
    return Replay(laps_total, own_recs, opp_recs, pit_loss)


# ---------------------------------------------------------------------------
# Real-data loading (Module 1 extension, spec section 4).
#
# fastf1/pandas are imported lazily so the synthetic core stays light.
# Live fetch needs network + F1 credentials (current FastF1); on any failure
# the caller should fall back to generate_race(). Mapping + cache are pure
# and unit-tested on fabricated frames (see test_pitwall.py).


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
    return out.sort_values("lap").reset_index(drop=True)


def records_from_frames(own, opp, weather_time=None, weather_temp=None,
                        default_temp=35.0, default_pit_loss=21.0):
    """Normalized frames -> (laps_total, own_recs, opp_recs, pit_loss).

    Pure mapping, no network. Out-lap times are replaced by the driver's
    clean-lap median (documented approximation); only status "1" laps count
    as clean pace; gaps come from lap-start deltas.
    """
    laps = sorted(set(own["lap"]) & set(opp["lap"]))
    if not laps:
        raise ValueError("no overlapping laps for the two drivers")
    own_by = {int(r["lap"]): r for _, r in own.iterrows()}
    opp_by = {int(r["lap"]): r for _, r in opp.iterrows()}

    def clean_median(n):
        ts = [float(r["time"]) for _, r in n.iterrows()
              if not r["pit_in"] and not r["pit_out"]
              and str(r["status"]) == "1" and np.isfinite(float(r["time"]))]
        return float(np.median(ts)) if ts else 95.0

    med = {"OWN": clean_median(own), "OPP": clean_median(opp)}

    def temp_at(start):
        if weather_time is None or weather_temp is None or not len(weather_time):
            return default_temp
        wt = np.asarray(weather_time, float)
        i = int(np.searchsorted(wt, float(start)))
        return float(np.asarray(weather_temp, float)[min(i, len(wt) - 1)])

    own_recs, opp_recs, diffs = [], [], []
    prev_gap, prev_pos = 0.0, {"OWN": 0, "OPP": 0}
    for l in laps:
        ro, rp = own_by[l], opp_by[l]

        def age_of(r, hist):
            a = float(r["age"])
            if np.isfinite(a):
                return max(int(round(a)), 1)
            return hist[-1].tyre_age + 1 if hist else 1

        def time_of(r, role):
            t = float(r["time"])
            if r["pit_out"] or not np.isfinite(t):
                return med[role]
            return t

        def pos_of(r, role):
            p = float(r["pos"])
            if np.isfinite(p):
                prev_pos[role] = int(p)
            return prev_pos[role]

        so, sp = float(ro["start"]), float(rp["start"])
        gap = float(so - sp) if np.isfinite(so) and np.isfinite(sp) else prev_gap
        prev_gap = gap
        temp = temp_at(so) if np.isfinite(so) else default_temp
        own_recs.append(LapRecord(
            lap=l, driver="OWN", compound=str(ro["compound"]),
            tyre_age=age_of(ro, own_recs), lap_time=time_of(ro, "OWN"),
            position=pos_of(ro, "OWN"), gap_to_rival=gap, track_temp=temp,
            safety_car=str(ro["status"]) != "1", pitted=bool(ro["pit_in"])))
        opp_recs.append(LapRecord(
            lap=l, driver="OPP", compound=str(rp["compound"]),
            tyre_age=age_of(rp, opp_recs), lap_time=time_of(rp, "OPP"),
            position=pos_of(rp, "OPP"), gap_to_rival=-gap, track_temp=temp,
            safety_car=str(rp["status"]) != "1", pitted=bool(rp["pit_in"])))
        for r, role in ((ro, "OWN"), (rp, "OPP")):
            if bool(r["pit_in"]) and np.isfinite(float(r["time"])):
                diffs.append(float(r["time"]) - med[role])
    pit_loss = float(np.median(diffs)) if diffs else default_pit_loss
    pit_loss = min(35.0, max(15.0, pit_loss))
    laps_total = int(max(max(own["lap"]), max(opp["lap"])))
    return laps_total, own_recs, opp_recs, pit_loss


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
                    "pitted"])
        for r in recs:
            w.writerow([r.lap, r.driver, r.compound, r.tyre_age,
                        f"{r.lap_time:.3f}", r.position,
                        f"{r.gap_to_rival:.3f}", f"{r.track_temp:.2f}",
                        int(r.safety_car), int(r.pitted)])
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
                pitted=bool(int(row["pitted"])))
            (own if r.driver == "OWN" else opp).append(r)
    return Replay(int(meta["laps_total"]), own, opp, float(meta["pit_loss"]))


def load_fastf1_replay(year, gp, own_drv, opp_drv, cache_dir=None,
                       offline_ok=True) -> Replay:
    """Fetch a real race via FastF1 (needs network + F1 credentials).

    Falls back to the CSV cache when offline_ok and an entry exists,
    else re-raises. Callers should fall back to generate_race().
    """
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
        laps_total, own_r, opp_r, ploss = records_from_frames(
            own_n, opp_n, wt, wtemp)
        replay = Replay(laps_total, own_r, opp_r, ploss)
        save_replay_csv(replay, ploss,
                        {"year": year, "gp": gp, "own": own_drv, "opp": opp_drv},
                        csv_path)
        return replay
    except Exception:
        if (offline_ok and csv_path.exists()
                and csv_path.with_suffix(".json").exists()):
            return load_replay_csv(csv_path)
        raise
