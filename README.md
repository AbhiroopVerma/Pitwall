# PITWALL

1v1 F1 opponent-prediction and counter-strategy engine for the hackathon
(26 Oct – 4 Nov 2026). It predicts when a chosen opponent will pit from public
timing data, then simulates our options to the flag and recommends one.

Decision support only. Alternative outcomes are simulated, never observed history.

## Pipeline

```
Replay → DegradationEstimator → PitHazardModel → ThreatDetector
    → OwnCarModel → Monte Carlo simulator → optimiser → recommendation
```

| Step | File | Method |
|---|---|---|
| Replay, sees laps `<= t` only | `state.py`, `data.py` | Raw timing, with stop cost learned from stops already seen |
| Tyre and pace | `degradation.py` | Stint baselines + non-negative `D(age)`, fuel as a stated assumption |
| Opponent pit hazard | `pit_model.py` | Logistic `h=σ(β·X)` from earlier laps, `P(≤k)=1−∏(1−h)` |
| Trigger | `threshold.py` | Opponent just stopped, or `p≥θ`, or gap-aware loss `≥λ` |
| Own pace | `owncar.py` | Same model, plus a private wear signal where one exists (SYNTHETIC in the demo race) |
| Simulation | `simulator.py` | Lap by lap to the flag, with on-track hold-ups, stop plans and shared random draws |
| Decision | `optimizer.py` | `U = P(ahead) − 0.25·deficit − 0.10·risk`, penalties scaled by pit loss |

## Run

```
source .venv/bin/activate
streamlit run pitwall/app.py                  # dashboard, http://localhost:8501

python -m pitwall.demo                        # synthetic race, lap 13: TRIGGERED
python -m pitwall.demo --lap 10               # synthetic race, lap 10: MONITOR
python -m pitwall.demo --race 2024 Hungaroring --own PIA --opp NOR --lap 16
python -m pitwall.demo --backtest             # scores on unseen real races

pip install -e ".[dev]" && pytest             # tests
```

All of this works offline. The repo bundles 70 races (2023–2025, about 77k laps)
from [OpenF1](https://openf1.org), which needs no login. Rebuild the data or the
results with:

```
python -m pitwall.openf1          # refetch the bundled races (about 15 min)
python -m pitwall.train_hazard    # retrain and re-score the hazard model
python -m pitwall.evaluate        # rerun the strategy comparison
```

The FastF1 option in the dashboard needs an F1 account (`F1_USERNAME`,
`F1_PASSWORD`). If it fails, the dashboard says so and clearly labels the
synthetic race it falls back to.

## Results

### Predicting stops, on races the model never saw

Trained on 41 dry races from 2023–24, tested on 22 dry races from 2025
(23,764 car-laps, 685 stops).

| Per-lap predictor | Brier | Log loss |
|---|---|---|
| PITWALL hazard model | 0.0274 | 0.1199 |
| Tyre-age-only baseline | 0.0278 | 0.1274 |
| Constant base rate | 0.0280 | 0.1308 |

For P(stop within 3 laps) at θ=30%, it warned before 34% of 636 stops and fired on
6.8% of laps with no stop coming. Probabilities are reasonably calibrated up to
about 30%, then run high: predictions of 40–50% came true 25% of the time.

**How to read this:** the edge over the baselines is real but small. Pit timing is
hard to predict from one car's public data, and most stops arrive without much
warning. An earlier version reported a Brier of 0.001, but that came from a data
leak: a feature already contained the stop being predicted.

### Choosing a strategy, in 300 simulated races

Each policy drove the same randomised races with the same noise (spec §12).

| Policy | P(ahead at flag) | Mean gap | Mean stops |
|---|---|---|---|
| PITWALL | 70% | +17.2 s | 2.24 |
| PITWALL ignoring the opponent forecast | 70% | +16.9 s | 2.32 |
| Tyre-age rule | 50% | +4.4 s | 2.00 |
| Reactive (cover their stop) | 44% | +2.6 s | 1.85 |
| Fixed one-stop | 35% | −9.8 s | 1.00 |

PITWALL beats every rule-based policy, by 20 to 34 points of P(ahead), and its stop
count matches the true optimum (2.2 stops, checked on 40 of these races).

**How to read this:**
- **The opponent forecast adds no measurable benefit here:** 0 points, 95% CI ±1.7.
  The gain comes from modelling our own tyres and planning to the flag. The opponent
  in these races does not react to us, which limits what predicting them can add.
- **The test is favourable to PITWALL:** these races share the simulator's hold-up
  rule, and PITWALL gets the synthetic wear signal.

So these results show good decisions inside this model's world, not proof that it
would win real races.

## Assumptions and limits

- **Fuel:** assumed at 0.04 s/lap, because in a first stint lap times cannot
  separate fuel burn from tyre age.
- **Compounds:** relative wear rates are assumed (SOFT 1.4, MEDIUM 1.0, HARD 0.7).
  The next compound is unknown, so a new stint gets a ±0.4 s pace offset.
- **Overtaking:** a car needs an assumed 1.0 s/lap advantage to pass, adjustable in
  the dashboard. Real circuits differ a lot.
- **Opponent behaviour:** the opponent does not react to our stop, and the hazard
  model does not use the gap to us.
- **Not modelled:** other traffic and future safety cars. Safety cars do feed the
  hazard model, but a stop under one still costs a full pit loss in the simulation.
- **Dry only:** in wet conditions the engine stays on MONITOR and says why.
- **Absolute gaps run wide:** current pace is held to the flag, so a large pace
  difference compounds. In the 2023 Bahrain race, ALO vs VER forecasts about −100 s
  at lap 13, while the real gap was about −38 s. Compare options against each other
  rather than reading the number as a finish gap.
- **Own-car telemetry:** there is none on historical races, so both cars use public
  lap times there. The private wear signal exists only in the synthetic race.
- **Replays of 2023–24 races are in-sample** for the hazard model, and the dashboard
  says so. Use a 2025 race for a fair view.
- **Data:** OpenF1 data is unofficial. Check its terms before any use beyond this
  project.
