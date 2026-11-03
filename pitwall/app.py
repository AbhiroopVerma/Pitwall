"""Module 8 — interactive pit-wall dashboard (Streamlit, spec section 11).

Thin UI over the engine: every number shown comes from ``decide_on_replay`` or
the stored evaluation results. Run from repo root::

    .venv/bin/streamlit run pitwall/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from pitwall.data import (bundled_drivers, generate_race, has_bundled_data,
                          list_bundled_races, load_bundled_replay, load_fastf1_replay)
from pitwall.demo import decide_on_replay
from pitwall.evaluate import OUT as STRATEGY_EVAL
from pitwall.owncar import OwnCarModel
from pitwall.pit_model import WEIGHTS, load_models, rows_for_replay
from pitwall.simulator import action_label
from pitwall.state import PIT_IN_SHARE

OPP_PITS = (14, 28)  # synthetic demo race
OWN_PITS = (15, 29)
HORIZON = 8
LAYOUT = dict(height=320, margin=dict(l=10, r=10, t=40, b=10))

st.set_page_config(page_title="PITWALL — F1 strategy intelligence", layout="wide")
st.title("PITWALL — opponent prediction + counter-strategy")
st.caption("1v1 decision support. Alternative outcomes are simulated counterfactuals, "
           "never observed history. Observe → Estimate → Predict → Detect → Simulate "
           "→ Optimise → Recommend.")


@st.cache_resource
def get_models():
    return load_models()


def fmt_time(t: float) -> str:
    return f"{t:.2f}s" if np.isfinite(t) else "not recorded"


# ---------------------------------------------------------------- sidebar ---
SYNTH, BUNDLED, FF1 = "Synthetic demo", "Historical race (bundled)", "FastF1 (needs F1 login)"
st.sidebar.header("Race selection")
sources = ([BUNDLED] if has_bundled_data() else []) + [SYNTH, FF1]
source = st.sidebar.radio("Data source", sources)
own_name, opp_name = "Our car", "Opponent"
real_data = False
pit, baseline = get_models()

if source == BUNDLED:
    races = list_bundled_races()
    race = st.sidebar.selectbox("Race", races, index=0,
                                format_func=lambda r: f"{r[0]} {r[1]}")
    drivers = bundled_drivers(*race)
    own_name = st.sidebar.selectbox("Our driver", drivers,
                                    index=drivers.index("ALO") if "ALO" in drivers else 1)
    others = [d for d in drivers if d != own_name]
    opp_name = st.sidebar.selectbox("Opponent", others,
                                    index=others.index("VER") if "VER" in others else 0)
    replay = load_bundled_replay(race[0], race[1], own_name, opp_name)
    label = f"REAL {race[0]} {race[1]}: {own_name} (us) vs {opp_name} (OpenF1 timing)"
    real_data = True
elif source == SYNTH:
    seed = st.sidebar.selectbox("Demo race (seed varies noise only)", [7, 1, 42], index=0)
    replay = generate_race(seed, opp_pit_laps=OPP_PITS, own_pit_laps=OWN_PITS)
    label = "SYNTHETIC demo race"
else:
    f_year = st.sidebar.number_input("Season", 2018, 2026, 2023, 1)
    f_gp = st.sidebar.text_input("Grand Prix", "Bahrain")
    f_own = st.sidebar.text_input("Our driver code", "ALO").upper().strip()
    f_opp = st.sidebar.text_input("Opponent driver code", "VER").upper().strip()
    try:
        with st.spinner("Loading FastF1 data (or CSV cache)…"):
            replay = load_fastf1_replay(f_year, f_gp, f_own, f_opp)
        label = f"REAL {f_year} {f_gp}: {f_own} (us) vs {f_opp} (FastF1)"
        own_name, opp_name = f_own, f_opp
        real_data = True
    except Exception as e:
        st.warning(f"FastF1 load failed ({type(e).__name__}: {e}); "
                   f"showing the synthetic demo race instead.")
        label = "SYNTHETIC demo race (FastF1 failed)"
        replay = generate_race(7, opp_pit_laps=OPP_PITS, own_pit_laps=OWN_PITS)

decision_laps = [l for l in replay.available_laps() if l < replay.laps_total] \
    or replay.available_laps()
if len(decision_laps) > 1:
    lap = st.sidebar.select_slider("Replay lap", options=decision_laps,
                                   value=min(decision_laps, key=lambda l: abs(l - 13)),
                                   help="Models see laps <= this lap only (no future leakage).")
else:
    lap = decision_laps[0]
if decision_laps[-1] < replay.laps_total - 1:
    st.sidebar.warning(f"Replay ends at lap {decision_laps[-1]} of {replay.laps_total}: "
                       f"one of the cars did not continue past it.")
theta = st.sidebar.slider("Trigger threshold θ (3-lap pit prob.)", 0.05, 0.95, 0.30, 0.05)
overtake_delta = st.sidebar.slider(
    "Pace advantage needed to overtake (s/lap)", 0.25, 3.0, 1.0, 0.25,
    help="Assumption, varies by circuit: low for Monza, high for Monaco.")

# ------------------------------------------------------------- engine run ---
snap = replay.state_at(lap)
st.sidebar.caption(f"{replay.laps_total}-lap race · pit loss so far {snap.pit_loss:.1f}s "
                   f"(prior until a stop is observed).")
with st.spinner("Running PITWALL pipeline…"):
    out = decide_on_replay(replay, lap, theta, pit, verbose=False,
                           overtake_delta=overtake_delta)
trig = out["trigger"]
st.caption(f"Data source: **{label}**.")

st.header(f"{trig.status}: {trig.reason}")
c1, c2, c3 = st.columns(3)
c1.metric("P(opp pits within 3)", f"{out['p3']:.0%}")
c2.metric("Threshold θ", f"{theta:.0%}")
c3.metric("Gap after stops in progress (we trail +)", f"{snap.effective_gap:+.1f}s")

tab_race, tab_opp, tab_strat, tab_val = st.tabs(
    ["Race state", "Opponent", "Counter-strategy", "Validation"])

# --------------------------------------------------------------- race tab ---
with tab_race:
    left, right = st.columns(2)
    for col, title, car, hist in ((left, f"{own_name} (us)", snap.own, snap.history_own),
                                  (right, f"{opp_name} (opponent)", snap.opp, snap.history_opp)):
        with col:
            st.subheader(title)
            pos = f"P{hist[-1].position} · " if hist[-1].position else ""
            st.write(f"{pos}Compound **{car.compound}**, age **{car.tyre_age}** · "
                     f"stops done **{car.stops_done}**"
                     + (" · **boxed this lap**" if car.just_pitted else ""))
            st.write(f"Last lap **{fmt_time(car.last_lap_time)}**"
                     + (" · must still use a second dry compound" if car.must_stop else ""))
    temp = f"Track temp **{snap.track_temp:.1f}°C** · " if np.isfinite(snap.track_temp) else ""
    st.write(f"{temp}Safety car / VSC / red flag **{snap.safety_car}** · "
             f"pit loss **{snap.pit_loss:.1f}s** · line gap **{snap.gap_own_behind_opp:+.1f}s**")
    if snap.own.just_pitted or snap.opp.just_pitted:
        st.caption(f"A car that just boxed has paid only {PIT_IN_SHARE:.0%} of the stop at the "
                   f"line, so the gap above the tabs includes the rest.")
    st.divider()
    st.subheader("Actual outcome (uses future, on request only)")
    if st.button("Reveal when the opponent actually pitted next", key=f"reveal_{label}_{lap}"):
        nxt = next((r.lap for r in replay._opp_list if r.pitted and r.lap > lap), None)
        st.warning(f"Opponent next pitted at lap **{nxt}**. Compare with the Opponent tab."
                   if nxt else "The opponent did not pit again in this replay.")
    else:
        st.caption("Hidden until pressed, so the replay stays leakage-free.")

# ----------------------------------------------------------- opponent tab ---
with tab_opp:
    est, fc = out["est"], out["forecast"]
    if est is None or fc is None:
        st.info(f"{trig.reason}. Advance the replay.")
    else:
        st.subheader("Tyre-related degradation (public data only)")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Pace penalty", f"{est.penalty:+.2f}s")
        d2.metric("Degradation slope", f"{est.slope:+.3f}s/lap")
        d3.metric("Next-lap forecast", f"{est.forecast[0]:.2f}s ± {est.std:.2f}")
        d4.metric("Useful extension", f"+{est.max_extra_laps} laps")
        st.caption("Tyre-*related pace loss*, not physical wear or temperature (spec §5). "
                   "Fuel is an assumed 0.04 s/lap because lap times cannot separate it "
                   "from tyre age.")

        hist = [r for r in snap.history_opp[-12:]
                if not r.pitted and not r.pit_out and not r.safety_car
                and np.isfinite(r.lap_time)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[r.lap for r in hist], y=[r.lap_time for r in hist],
                                 mode="lines+markers", name="observed clean laps"))
        fx = list(range(lap + 1, lap + 1 + len(est.forecast)))
        fig.add_trace(go.Scatter(x=fx, y=est.forecast, mode="lines+markers",
                                 error_y=dict(type="data", array=est.forecast_std),
                                 name="stay-out forecast ± 1 std"))
        fig.update_layout(title="Opponent pace: recent laps + 5-lap forecast",
                          xaxis_title="Lap", yaxis_title="Lap time (s)", **LAYOUT)
        st.plotly_chart(fig, width="stretch")

        st.subheader("Pit-stop probability forecast")
        labels = [f"L{lap + j}" for j in range(1, len(fc.hazards) + 1)]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=labels, y=fc.hazards,
                              name="conditional hazard h (pit that lap | not yet stopped)"))
        fig2.add_trace(go.Bar(x=labels, y=fc.pit_dist[:-1],
                              name="first-pit probability (unconditional)"))
        fig2.update_layout(title="Next-5-lap pit forecast (hazards ≠ distribution)",
                           yaxis_title="Probability", barmode="group", **LAYOUT)
        st.plotly_chart(fig2, width="stretch")
        p1, p3, p5 = fc.p_within[1], fc.p_within[3], fc.p_within[5]
        st.write(f"P(pit within 1) **{p1:.0%}** · within 3 **{p3:.0%}** · "
                 f"within 5 **{p5:.0%}** · P(no pit in window) **{fc.pit_dist[-1]:.0%}**")

        st.subheader("Our car pace")
        own = out.get("own_model")
        if own is None:
            st.info("Not enough own-car laps to fit the pace model yet.")
        else:
            steps = min(HORIZON, replay.laps_total - lap)
            stay = own.predict_stay(snap.own.next_age - 1, lap, steps)
            fresh = own.predict_fresh(lap, steps)
            xs = list(range(lap + 1, lap + 1 + steps))
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=xs, y=stay, mode="lines+markers",
                                      name="stay out (ageing tyres)"))
            fig3.add_trace(go.Scatter(x=xs, y=fresh, mode="lines+markers",
                                      name="new tyres, same compound pace, excluding stop time"))
            fig3.update_layout(title="Own-car pace projection",
                               xaxis_title="Lap", yaxis_title="Lap time (s)", **LAYOUT)
            st.plotly_chart(fig3, width="stretch")
            if own.uses_private_signal:
                st.caption(f"Own-car curve fitted to a SYNTHETIC private wear signal, standing "
                           f"in for team telemetry (spec §7). Lap noise ±{own.std_:.2f}s.")
            else:
                st.caption(f"No private telemetry on historical races, so the own-car model uses "
                           f"public lap times too. Lap noise ±{own.std_:.2f}s.")

# ----------------------------------------------------------- strategy tab ---
with tab_strat:
    if not trig.active or "results" not in out:
        st.info("Monitoring only — no counter-strategy simulation. "
                "Lower θ or advance the replay to trigger the engine.")
    else:
        st.subheader("Candidate responses (Monte Carlo to the flag, 1000 sims/action)")
        rec = out["recommendation"]
        rank = {o: i for i, (o, _) in enumerate(rec.ranked, 1)}
        rows = [{"action": action_label(r.our_pit_offset),
                 "P(ahead at flag)": round(r.p_ahead, 3),
                 "expected gap (s)": round(r.exp_gap, 2),
                 "worst 5% (s)": round(r.risk, 2),
                 "planned stops (lap)": ", ".join(map(str, r.stops)) or "none",
                 "utility rank": rank[r.our_pit_offset]}
                for r in out["results"]]
        st.dataframe(rows, width="stretch", hide_index=True)
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            x=[r["action"] for r in rows], y=[r["P(ahead at flag)"] for r in rows],
            marker_color=["#2ca02c" if r["utility rank"] == 1 else "#1f77b4" for r in rows],
            name="P(ahead at flag)"))
        fig4.update_layout(title="P(ahead at flag) per response (green = recommended)",
                           yaxis_title="Probability", height=300,
                           margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig4, width="stretch")
        st.subheader("Recommendation")
        st.success(rec.explanation)
        st.caption("Utility U = P(ahead) − 0.25·deficit − 0.10·tail risk, with both penalties "
                   "scaled by the pit loss. Assumes: no reaction from the opponent, no other "
                   "traffic, no future safety cars, and a next compound of unknown pace "
                   "(±0.4 s). Outcomes are simulated counterfactuals (spec §12).")

# --------------------------------------------------------- validation tab ---
with tab_val:
    metrics = json.loads(WEIGHTS.read_text())["metrics"]
    t = metrics["test"]
    st.subheader("Pit-stop model on unseen real races")
    st.write(f"Trained on {metrics['train']['races']} dry races ({metrics['train']['years']}); "
             f"tested on {t['races']} later dry races ({t['years']}), "
             f"{t['laps']} car-laps and {t['stops']} stops.")
    st.dataframe([{"predictor": n, "Brier": round(t[k]["brier"], 4),
                   "Log loss": round(t[k]["log_loss"], 4)}
                  for n, k in (("PITWALL hazard model", "model"),
                               ("Tyre-age-only baseline", "baseline"),
                               ("Constant base rate", "base_rate"))],
                 width="stretch", hide_index=True)
    w = t["window_model"]
    st.write(f"P(stop within 3 laps) at θ={w['theta']:.0%}: warned before "
             f"**{w['stop_recall']:.0%}** of {w['stops_scored']} stops, and fired on "
             f"**{w['false_alarm_rate']:.0%}** of laps with no stop coming.")
    cal = w["calibration"]
    figc = go.Figure()
    figc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                              line=dict(dash="dash")))
    figc.add_trace(go.Scatter(x=[c["predicted"] for c in cal], y=[c["actual"] for c in cal],
                              mode="lines+markers", name="model",
                              text=[f"n={c['n']}" for c in cal]))
    figc.update_layout(title="Calibration of P(stop within 3 laps), unseen races",
                       xaxis_title="Predicted", yaxis_title="Actual", **LAYOUT)
    st.plotly_chart(figc, width="stretch")

    if STRATEGY_EVAL.exists():
        ev = json.loads(STRATEGY_EVAL.read_text())
        st.subheader("Strategy comparison (controlled simulated races, spec §12)")
        st.dataframe([{"policy": n, "P(ahead at flag)": round(r["p_ahead"], 3),
                       "mean gap (s)": round(r["mean_gap"], 2),
                       "mean stops": round(r["mean_stops"], 2),
                       "PITWALL gain (s)": round(r["pitwall_gain_s"], 2),
                       "gain 95% CI": f"{r['pitwall_gain_s_ci95'][0]:+.1f} to "
                                      f"{r['pitwall_gain_s_ci95'][1]:+.1f}"}
                      for n, r in ev["table"].items()], width="stretch", hide_index=True)
        st.caption(f"{ev['races']} paired synthetic races. Simulated counterfactuals: they test "
                   f"decisions in a world that shares the simulator's hold-up assumption.")

    st.subheader("This replay so far")
    rows = rows_for_replay(replay)
    show_all = st.checkbox("Show the whole race (uses future)", value=False)
    if rows:
        X = np.array([r[1] for r in rows])
        hz, bl = pit.hazards(X), baseline.hazards(X)
        laps = np.array([r[0] for r in rows])
        y = np.array([r[3] for r in rows])
        mask = np.ones(len(laps), bool) if show_all else laps <= lap
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=laps[mask], y=hz[mask], mode="lines+markers",
                                  name="model hazard"))
        fig5.add_trace(go.Scatter(x=laps[mask], y=bl[mask], mode="lines",
                                  name="tyre-age-only baseline"))
        stop_mask = mask & (y == 1.0)
        fig5.add_trace(go.Scatter(x=laps[stop_mask], y=[1.0] * int(stop_mask.sum()),
                                  mode="markers", marker=dict(size=12, symbol="x"),
                                  name="actual stop"))
        fig5.update_layout(title="Per-lap pit hazard vs actual stops",
                           xaxis_title="Lap", yaxis_title="P(pit at end of lap)", **LAYOUT)
        st.plotly_chart(fig5, width="stretch")
    if source == BUNDLED and race[0] in metrics["train"]["years"]:
        st.caption(f"{race[0]} was used to train the hazard model, so this chart is in-sample. "
                   f"Pick a {metrics['test']['years'][0]} race for a fair view.")
    if not real_data:
        st.caption("Synthetic race: the hazard model was trained on real races, so this "
                   "chart is an out-of-domain view, not a score.")
