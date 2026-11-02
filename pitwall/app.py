"""Module 8 — interactive pit-wall dashboard (Streamlit, spec section 11).

Thin UI over the frozen engine (Modules 1-7): every number shown comes from
``decide_at`` / ``backtest`` or the same public model APIs. Run from repo root::

    .venv/bin/streamlit run pitwall/app.py
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from pitwall.data import generate_race, load_fastf1_replay
from pitwall.degradation import DegradationEstimator
from pitwall.demo import backtest, build_models, decide_on_replay
from pitwall.owncar import OwnCarModel
from pitwall.pit_model import (PitHazardModel, age_only_baseline, brier_score,
                               log_loss)

OPP_PITS = (14, 28)  # demo-race stop laps (must match decide_at internals)
OWN_PITS = (15, 29)
HORIZON = 8

st.set_page_config(page_title="PITWALL — F1 strategy intelligence", layout="wide")
st.title("PITWALL — opponent prediction + counter-strategy")
st.caption(
    "1v1 proof of concept. Counterfactual outcomes are simulated, "
    "never observed history. Observe → Estimate → Predict → Detect → "
    "Simulate → Optimise → Recommend."
)


@st.cache_resource
def get_model() -> PitHazardModel:
    _, pit = build_models(1)
    return pit


def scored_rows(replay, pit) -> list[tuple]:
    """Per-lap hazard predictions vs actual stops (mirrors demo.backtest)."""
    rows: list[tuple] = []
    for lap in replay.available_laps():
        snap = replay.state_at(lap)
        hist = list(snap.history_opp)
        if len([r for r in hist if not r.pitted and not r.safety_car]) < 6:
            continue
        deg = DegradationEstimator().fit(hist, replay.laps_total)
        est = deg.estimate(hist, replay.laps_total, lap)
        f = PitHazardModel.feats(snap.opp.tyre_age, lap, replay.laps_total,
                                 est.penalty, -snap.gap_own_behind_opp,
                                 snap.opp.stops_done)
        rec = next(r for r in hist if r.lap == lap)
        rows.append((lap, snap.opp.tyre_age, pit.hazard(f),
                     age_only_baseline(snap.opp.tyre_age),
                     1.0 if rec.pitted else 0.0))
    return rows


@st.cache_data
def per_lap_scores(seed: int) -> list[tuple]:
    demo = generate_race(seed, opp_pit_laps=OPP_PITS, own_pit_laps=OWN_PITS)
    return scored_rows(demo, get_model())


def actual_opp_pits(replay) -> list[int]:
    """Actual historical stop laps. Future info: only call for reveal/backtest."""
    return [r.lap for r in replay._opp_list if r.pitted]


# ---------------------------------------------------------------- sidebar ---
st.sidebar.header("Race selection")
source = st.sidebar.radio("Data source", ["Synthetic demo", "FastF1 (real)"])
if source == "Synthetic demo":
    seed = st.sidebar.selectbox("Demo race (seed varies noise only)",
                                [7, 1, 42], index=0)
else:
    f_year = st.sidebar.number_input("Season", 2018, 2025, 2023, 1)
    f_gp = st.sidebar.text_input("Grand Prix", "Bahrain")
    f_own = st.sidebar.text_input("Our driver code", "ALO").upper().strip()
    f_opp = st.sidebar.text_input("Opponent driver code", "VER").upper().strip()

# ------------------------------------------------------------- replay load ---
pit = get_model()
label = "SYNTHETIC demo race"
if source == "Synthetic demo":
    replay = generate_race(seed, opp_pit_laps=OPP_PITS, own_pit_laps=OWN_PITS)
else:
    try:
        with st.spinner("Loading FastF1 data (or CSV cache)…"):
            replay = load_fastf1_replay(f_year, f_gp, f_own, f_opp)
        label = f"FASTF1 {f_year} {f_gp}: {f_own} (us) vs {f_opp}"
    except Exception as e:
        st.warning(f"FastF1 load failed ({type(e).__name__}: {e}); "
                   f"using the synthetic demo race instead.")
        label = "SYNTHETIC demo race (FastF1 fallback)"
        replay = generate_race(7, opp_pit_laps=OPP_PITS, own_pit_laps=OWN_PITS)

max_lap = max(2, replay.laps_total - 1)
lap = st.sidebar.slider("Replay lap", 1, max_lap, min(13, max_lap),
                        help="Models see laps <= this lap only (no future leakage).")
theta = st.sidebar.slider("Trigger threshold θ (3-lap pit prob.)",
                          0.05, 0.95, 0.30, 0.05)
st.sidebar.caption(f"{replay.laps_total}-lap race · "
                   f"pit loss {replay.pit_loss:.0f}s.")

# ------------------------------------------------------------- engine run ---
snap = replay.state_at(lap)
with st.spinner("Running PITWALL pipeline…"):
    out = decide_on_replay(replay, lap, theta, pit, verbose=False)
trig = out["trigger"]
st.caption(f"Data source: **{label}**.")

st.header(f"{trig.status}: {trig.reason}")
c1, c2, c3 = st.columns(3)
c1.metric("P(opp pits within 3)", f"{out['p3']:.0%}")
c2.metric("Threshold θ", f"{theta:.0%}")
c3.metric("Gap (we trail +)", f"{snap.gap_own_behind_opp:+.1f}s")

tab_race, tab_opp, tab_strat, tab_val = st.tabs(
    ["Race state", "Opponent", "Counter-strategy", "Validation"])

# --------------------------------------------------------------- race tab ---
with tab_race:
    left, right = st.columns(2)
    with left:
        st.subheader("Our car (OWN)")
        st.write(f"Compound **{snap.own.compound}**, age **{snap.own.tyre_age}** · "
                 f"stops done **{snap.own.stops_done}**")
        st.write(f"Last lap **{snap.own.last_lap_time:.2f}s** · "
                 f"private wear signal: {snap.own.private_wear} (synthetic)")
    with right:
        st.subheader("Opponent (OPP)")
        st.write(f"Compound **{snap.opp.compound}**, age **{snap.opp.tyre_age}** · "
                 f"stops done **{snap.opp.stops_done}**")
        st.write(f"Last lap **{snap.opp.last_lap_time:.2f}s** · "
                 f"private wear signal: {snap.opp.private_wear} (never available)")
    st.write(f"Track temp **{snap.track_temp:.1f}°C** · "
             f"safety car **{snap.safety_car}** · pit loss **{snap.pit_loss:.0f}s**")
    st.divider()
    st.subheader("Historical reveal (uses future — demo only)")
    if st.button("Reveal when the opponent actually pitted next",
                 key=f"reveal_{label}_{lap}"):
        nxt = next((p for p in actual_opp_pits(replay) if p > lap), None)
        st.warning(f"Opponent actually pitted next at lap **{nxt}**. "
                   f"Compare with the forecast in the Opponent tab.")
    else:
        st.caption("Hidden until pressed, so the replay stays leakage-free.")

# ----------------------------------------------------------- opponent tab ---
with tab_opp:
    est = out["est"]
    fc = out["forecast"]
    if est is None or fc is None:
        st.info("Not enough clean laps yet — model not ready. Advance the replay.")
    else:
        st.subheader("Tyre-related degradation (public data only)")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Pace penalty", f"{est.penalty:+.2f}s")
        d2.metric("Degradation slope", f"{est.slope:+.3f}s/lap")
        d3.metric("Next-lap forecast", f"{est.forecast[0]:.2f}s ± {est.std:.2f}")
        d4.metric("Useful extension", f"+{est.max_extra_laps} laps")
        st.caption("Tyre-*related pace loss*, not physical wear/temperature (spec §5).")

        hist = [r for r in snap.history_opp[-12:] if not r.pitted and not r.safety_car]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[r.lap for r in hist],
                                 y=[r.lap_time for r in hist],
                                 mode="lines+markers", name="observed clean laps"))
        fig.add_trace(go.Scatter(x=list(range(lap + 1, lap + 1 + len(est.forecast))),
                                 y=est.forecast, mode="lines+markers",
                                 name="stay-out forecast"))
        fig.update_layout(title="Opponent pace: recent laps + 5-lap forecast",
                          xaxis_title="Lap", yaxis_title="Lap time (s)",
                          height=320, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")

        st.subheader("Pit-stop probability forecast")
        labels = [f"L{lap + j}" for j in range(1, len(fc.hazards) + 1)]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=labels, y=fc.hazards,
                              name="conditional hazard h (pit now | survived)"))
        fig2.add_trace(go.Bar(x=labels, y=fc.pit_dist[:-1],
                              name="first-pit probability (unconditional)"))
        fig2.update_layout(title="Next-5-lap pit forecast (hazards ≠ distribution)",
                           yaxis_title="Probability", barmode="group",
                           height=320, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig2, width="stretch")
        p1, p3, p5 = fc.p_within[1], fc.p_within[3], fc.p_within[5]
        st.write(f"P(pit within 1) **{p1:.0%}** · within 3 **{p3:.0%}** · "
                 f"within 5 **{p5:.0%}** · P(no pit in window) "
                 f"**{fc.pit_dist[-1]:.0%}**")

        st.subheader("Our car pace (richer + synthetic private signal)")
        try:
            own = OwnCarModel().fit(snap.history_own, replay.laps_total)
            stay = own.predict_stay(snap.own.tyre_age, lap, replay.laps_total,
                                    snap.track_temp, HORIZON)
            fresh = own.predict_fresh(lap, replay.laps_total,
                                      snap.track_temp, HORIZON)
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=list(range(lap + 1, lap + 1 + HORIZON)),
                                      y=stay, mode="lines+markers",
                                      name="stay out (ageing tyres)"))
            fig3.add_trace(go.Scatter(x=list(range(lap + 1, lap + 1 + HORIZON)),
                                      y=fresh, mode="lines+markers",
                                      name="pit now, then fresh tyres"))
            fig3.update_layout(title="Own-car 8-lap pace projection",
                               xaxis_title="Lap", yaxis_title="Lap time (s)",
                               height=320, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig3, width="stretch")
            st.caption(f"Own-car forecast std ±{own.std_:.2f}s. Private wear is "
                       f"synthetic and labelled as such (spec §7).")
        except ValueError:
            st.info("Not enough own-car laps to fit the pace model yet.")

# ----------------------------------------------------------- strategy tab ---
with tab_strat:
    if not trig.active or "results" not in out:
        st.info("Monitoring only — no counter-strategy simulation. "
                "Lower θ or advance the replay to trigger the engine.")
    else:
        st.subheader("Candidate responses (Monte Carlo, 1000 sims/action)")
        rec = out["recommendation"]
        rows = [{"action": f"pit +{r.our_pit_offset}" if r.our_pit_offset else "pit now",
                 "P(finish ahead)": round(r.p_ahead, 3),
                 "expected gap (s)": round(r.exp_gap, 2),
                 "worst-5% (s)": round(r.risk, 2),
                 "utility rank": next(i for i, (o, _) in enumerate(rec.ranked, 1)
                                      if o == r.our_pit_offset)}
                for r in out["results"]]
        st.dataframe(rows, width="stretch", hide_index=True)
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            x=[r["action"] for r in rows],
            y=[r["P(finish ahead)"] for r in rows],
            marker_color=["#2ca02c" if r["utility rank"] == 1 else "#1f77b4"
                          for r in rows],
            name="P(finish ahead)"))
        fig4.update_layout(title="P(finish ahead) per response (green = recommended)",
                           yaxis_title="Probability", height=300,
                           margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig4, width="stretch")
        st.subheader("Recommendation")
        st.success(rec.explanation)
        st.caption("Utility U = P(ahead) − 0.15·deficit − 0.10·tail-risk. "
                   "Alternative outcomes are simulated counterfactuals (spec §12).")

# --------------------------------------------------------- validation tab ---
with tab_val:
    if source == "Synthetic demo":
        st.subheader("Held-out backtest (synthetic race)")
        with contextlib.redirect_stdout(io.StringIO()):
            agg = backtest(seed)
        st.write(f"{agg['n']} decision laps · Brier model "
                 f"**{agg['brier_model']:.4f}** vs baseline "
                 f"**{agg['brier_base']:.4f}** · LogLoss model "
                 f"**{agg['ll_model']:.4f}** vs baseline "
                 f"**{agg['ll_base']:.4f}**")
        rows = per_lap_scores(seed)
    else:
        st.subheader("Real-race scoring (FastF1 laps, synthetic-trained model)")
        rows = scored_rows(replay, pit)
        y = np.array([r[4] for r in rows])
        pm = np.array([r[2] for r in rows])
        pb = np.array([r[3] for r in rows])
        st.write(f"{len(rows)} decision laps · Brier model "
                 f"**{brier_score(y, pm):.4f}** vs baseline "
                 f"**{brier_score(y, pb):.4f}** · LogLoss model "
                 f"**{log_loss(y, pm):.4f}** vs baseline "
                 f"**{log_loss(y, pb):.4f}**")
        st.caption("Out-of-domain check: hazard model trained on synthetic "
                   "races, scored on real laps.")
    laps = [r[0] for r in rows]
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=laps, y=[r[2] for r in rows],
                              mode="lines+markers", name="model hazard"))
    fig5.add_trace(go.Scatter(x=laps, y=[r[3] for r in rows],
                              mode="lines", name="age-only baseline"))
    stops = [r[0] for r in rows if r[4] == 1.0]
    fig5.add_trace(go.Scatter(x=stops, y=[1.0] * len(stops), mode="markers",
                              marker=dict(size=12, symbol="x"),
                              name="actual stop"))
    fig5.update_layout(title="Per-lap pit hazard vs actual stops",
                       xaxis_title="Lap", yaxis_title="P(pit at end of lap)",
                       height=320, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig5, width="stretch")
    by_lap = {r[0]: r[2] for r in rows}
    actual = actual_opp_pits(replay)
    hits = sum(max(by_lap.get(s - d, 0.0) for d in (1, 2, 3)) >= 0.20
               for s in actual)
    npos = sum(r[4] for r in rows)
    st.write(f"Pit-window recall: **{hits}/{len(actual)}** stops had a ≥20% "
             f"hazard in the preceding 3 laps.")
    st.caption(f"{int(npos)} positive stops in {len(rows)} decision laps — "
               f"indicative, not conclusive. Train/test split is by race, "
               f"never by lap (spec §6).")
