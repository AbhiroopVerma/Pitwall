"""Fetch historical race laps from OpenF1 into a compact bundled dataset.

No login is needed for historical data. Run once, the result is checked in so
the dashboard and the training script work offline::

    python -m pitwall.openf1 --years 2023 2024 2025

Responses are cached per race in ``.data_cache/openf1`` so a run can resume.
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://api.openf1.org/v1/"
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".data_cache" / "openf1"
BUNDLE = Path(__file__).resolve().parent / "assets" / "openf1_laps.csv.gz"
NEUTRAL = Path(__file__).resolve().parent / "assets" / "openf1_neutral.json"
SPACING_S = 2.2  # the free tier allows about 30 requests a minute


def _get(endpoint: str, **params) -> list[dict]:
    url = BASE + endpoint + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "pitwall-openf1"})
    delay = 10.0
    for attempt in range(6):
        time.sleep(SPACING_S)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code == 404:  # OpenF1 answers 404 for "no rows"
                return []
            if e.code not in (429, 500, 502, 503, 504) or attempt == 5:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == 5:
                raise
        time.sleep(delay)
        delay *= 1.5
    return []


def _cached(key: int, endpoint: str) -> list[dict]:
    path = CACHE / f"{key}_{endpoint}.json.gz"
    if path.exists():
        with gzip.open(path, "rt") as f:
            return json.load(f)
    rows = _get(endpoint, session_key=key)
    if not rows:  # a session that has not run yet must not be cached as empty
        return rows
    CACHE.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump(rows, f)
    return rows


def _neutral_kind(message: dict) -> str | None:
    """SC, VSC or RED if this race-control message starts a neutral period."""
    text = (message.get("message") or "").upper()
    if message.get("flag") == "RED":
        return "RED"
    if message.get("category") == "SafetyCar" and "DEPLOYED" in text:
        return "VSC" if "VIRTUAL" in text or "VSC" in text else "SC"
    return None


def _neutral_ended(message: dict, is_open: bool) -> bool:
    text = (message.get("message") or "").upper()
    return ((message.get("category") == "SafetyCar"
             and ("ENDING" in text or "IN THIS LAP" in text))
            or (message.get("flag") == "GREEN" and "PIT EXIT" not in text and is_open)
            or "RACE WILL RESUME" in text)


def neutral_periods(messages: list[dict]) -> list[tuple[float, float, str]]:
    """Safety car, VSC and red flag periods as (start, end, kind) epoch seconds.

    A VSC upgraded to a safety car, or a safety car stopped by a red flag, splits
    into consecutive periods so the kind always says what is running now.
    """
    out, opened, kind = [], None, None
    rank = {"VSC": 0, "SC": 1, "RED": 2}
    for m in sorted(messages, key=lambda r: r.get("date") or ""):
        if not m.get("date"):
            continue
        t = pd.Timestamp(m["date"]).timestamp()
        started = _neutral_kind(m)
        if started and opened is None:
            opened, kind = t, started
        elif started and rank[started] > rank[kind]:
            out.append((opened, t, kind))
            opened, kind = t, started
        elif _neutral_ended(m, opened is not None) and opened is not None:
            out.append((opened, t, kind))
            opened, kind = None, None
    if opened is not None:
        out.append((opened, opened + 300.0, kind))
    return out


def neutral_intervals(messages: list[dict]) -> list[tuple[float, float]]:
    """Neutral periods as (start, end), whatever their kind."""
    return [(a, b) for a, b, _ in neutral_periods(messages)]


def race_frame(session: dict) -> pd.DataFrame:
    key = session["session_key"]
    laps = _cached(key, "laps")
    stints = _cached(key, "stints")
    control = _cached(key, "race_control")
    if not laps:
        return pd.DataFrame()
    neutral = neutral_intervals(control)
    by_driver_stints: dict[int, list[dict]] = {}
    for s in stints:
        by_driver_stints.setdefault(s["driver_number"], []).append(s)
    out_laps = {(r["driver_number"], r["lap_number"]) for r in laps
                if r.get("is_pit_out_lap") and r["lap_number"] > 1}
    rows = []
    for r in laps:
        d, n = r["driver_number"], r["lap_number"]
        stint = next((s for s in by_driver_stints.get(d, [])
                      if s.get("lap_start") is not None and s.get("lap_end") is not None
                      and s["lap_start"] <= n <= s["lap_end"]), None)
        age = None
        if stint and stint.get("tyre_age_at_start") is not None:
            age = stint["tyre_age_at_start"] + n - stint["lap_start"] + 1
        start = pd.Timestamp(r["date_start"]).timestamp() if r.get("date_start") else None
        dur = r.get("lap_duration")
        is_neutral = False
        if start is not None:
            end = start + (dur or 90.0)
            is_neutral = any(a < end and start < b for a, b in neutral)
        rows.append({
            "year": session["year"], "race": session.get("country_name"),
            "circuit": session.get("circuit_short_name"), "driver": d,
            "acronym": None, "lap": n,
            "compound": (stint or {}).get("compound") or "UNKNOWN",
            "age": age, "time": dur, "start": start,
            "pit_in": (d, n + 1) in out_laps, "pit_out": (d, n) in out_laps,
            "neutral": is_neutral})
    df = pd.DataFrame(rows)
    df["pos"] = df.groupby("lap")["start"].rank(method="first")
    df["total_laps"] = int(df["lap"].max())
    names = {x["driver_number"]: x.get("name_acronym")
             for x in _cached(key, "drivers")}
    df["acronym"] = df["driver"].map(names).fillna(df["driver"].astype(str))
    return df


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025, 2026])
    ap.add_argument("--out", default=str(BUNDLE))
    a = ap.parse_args(argv)
    frames, periods = [], {}
    for year in a.years:
        sessions = _get("sessions", year=year, session_name="Race")
        for s in sorted(sessions, key=lambda r: r["date_start"]):
            try:
                df = race_frame(s)
            except Exception as e:
                print(f"skip {year} {s.get('country_name')}: {type(e).__name__}: {e}", flush=True)
                continue
            print(f"{year} {s.get('country_name'):20s} {len(df):5d} laps", flush=True)
            if len(df):
                frames.append(df)
                periods[f"{year}|{s.get('circuit_short_name')}"] = [
                    [round(x, 3), round(y, 3), k]
                    for x, y, k in neutral_periods(_cached(s["session_key"], "race_control"))]
    all_laps = pd.concat(frames, ignore_index=True)
    all_laps["time"] = all_laps["time"].round(3)
    all_laps["start"] = all_laps["start"].round(3)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    all_laps.to_csv(a.out, index=False, compression="gzip")
    NEUTRAL.write_text(json.dumps(periods, indent=0))
    print(f"wrote {len(all_laps)} laps to {a.out}", flush=True)


if __name__ == "__main__":
    main()
