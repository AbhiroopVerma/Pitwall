# PITWALL — proof of concept

1v1 F1 opponent-prediction + counter-strategy engine for the hackathon
(26 Oct – 4 Nov 2026). Prototype only: **synthetic data**, numpy-only,
no network calls. FastF1/OpenF1 loaders plug into `pitwall/data.py` later.

## Pipeline (spec §§3–10)

```
Replay → DegradationEstimator → PitHazardModel → ThreatDetector
    → OwnCarModel → Monte Carlo simulator → optimiser → recommendation
```

| Step | File | Math |
|---|---|---|
| Replay (no future leakage, opp has no private signal) | `state.py`, `data.py` | history `<= lap` only |
| Opp tyre/pace `L = b·[1,age,age²,fuel,temp]`, ridge | `degradation.py` | `D(age)=b₁·age+b₂·age²`, slope, forecast ± std |
| Opp pit hazard `h=σ(β·X)`, `P(≤k)=1−∏(1−h)` | `pit_model.py` | GD logistic, projected ages |
| Trigger `p≥θ` or loss `≥λ` | `threshold.py` | rule-based |
| Own pace (extra SYNTHETIC wear signal) | `owncar.py` | stay-out vs fresh-tyre forecasts |
| Monte Carlo over opp pit-lap distribution | `simulator.py` | undercut/overcut emerge |
| `U=w₁P(ahead)−w₂deficit−w₃risk`, ranked + explanation | `optimizer.py` | argmax expected utility |

## Run (project-local env)

```
source .venv/bin/activate               # from repo root
python -m pitwall.demo                  # lap 13: TRIGGERED, full recommendation
python -m pitwall.demo --lap 10         # mid-stint: MONITOR, no simulation
python -m pitwall.demo --backtest       # held-out scoring vs age-only baseline
```

## Dashboard (Module 8, spec §11)

```
source .venv/bin/activate               # from repo root
streamlit run pitwall/app.py            # open http://localhost:8501
```

Four tabs (Race state / Opponent / Counter-strategy / Validation), lap
replay slider, θ threshold slider, one-click historical reveal (gated so
the replay stays leakage-free), and a Synthetic/FastF1 source toggle.

## Real data (FastF1, optional)

`pitwall/data.py:load_fastf1_replay` maps FastF1 laps → `LapRecord`s
(out-laps sanitized to stint median, gaps from lap-start deltas, pit loss
estimated from in-laps) with a CSV cache in `.data_cache/`. Live fetch
needs network + F1 credentials in current FastF1; without them the
dashboard shows a warning and falls back to synthetic. Mapping, cache,
and the replay-agnostic pipeline are unit-tested on fabricated frames.

## Demo scenarios (offline-safe)

1. Lap 13, θ=0.30 → TRIGGERED, recommend pit now (cover undercut).
2. Lap 10 → MONITOR, no simulation (cheap monitoring path).
3. `--backtest` → Brier/LogLoss beat the age-only baseline.
4. Dashboard → Validation tab → per-lap hazard chart + pit-window recall.
5. Dashboard → FastF1 toggle → graceful fallback warning (no network).

Default run replays the demo race up to lap 13 (opponent actually pits at 14):
threat triggers, Monte Carlo compares pit now/+1/+2/+3, optimiser recommends
covering the undercut. `--backtest` scores the hazard model (Brier ≈ 0.001,
LogLoss ≈ 0.02) vs an age-only baseline on a held-out race.

## Key assumptions / limits

- Synthetic generator stands in for FastF1/OpenF1; replace `generate_race`
  with a real loader returning `LapRecord`s — downstream code is unchanged.
- Degradation is tyre-*related pace loss*, not physical wear/temperature.
- Fresh-tyre pace after a stop is estimated from recent clean laps (−0.3 s);
  traffic/rejoin effects are noise only.
- Counterfactuals are *simulated*, never observed history (§12 of spec).
