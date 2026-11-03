# PITWALL

1v1 F1 opponent-prediction and counter-strategy engine for the hackathon
(26 Oct – 4 Nov 2026). It predicts when a chosen opponent will pit from public
timing data, then simulates our options to the flag and recommends one.

Decision support only. Alternative outcomes are simulated, never observed history.

## Live terminal

A text-only strategy screen for one car against one rival, updating as the race runs.
The call and the words to say to the driver come first; everything below backs them up.

```
source .venv/bin/activate

python -m pitwall.live                                   # real 2026 Mexico race, LEC vs VER
python -m pitwall.live --race 2025 Monza --own LEC --opp NOR --start-lap 25
python -m pitwall.live --fabricated safety-car           # or: undercut, rival-out
python -m pitwall.live --openf1 11404 --rehearse --own LEC --opp VER   # past race, live parser
python -m pitwall.live --openf1 latest --own LEC --opp NOR             # a race happening now

python -m pitwall.live --list                            # every race available
python -m pitwall.live --plain --log race.jsonl          # no full screen; keep a decision log
```

`--speed` sets race seconds per real second. The default is 20 (a lap takes about 4 s), and
`--speed 1` is true real time. While running: `space` pause, `+`/`-` speed, `n` next lap,
`t` change rival, `q` quit.

| Source | Data | Needs |
|---|---|---|
| `--race` | 89 real races, 2023 to Mexico 2026, bundled | Nothing, works offline |
| `--fabricated` | Invented 20-car race, labelled on screen | Nothing |
| `--openf1 KEY --rehearse` | Real OpenF1 rows released on schedule through the live parser | Internet |
| `--openf1 latest` | The session running now, polled every 8 s | Paid OpenF1 account: `OPENF1_USERNAME`, `OPENF1_PASSWORD`, plus `--laps N` for a new circuit |

### What it does each lap

| Step | File | Why |
|---|---|---|
| Events, released only when they could be known | `live_feed.py` | The engine cannot tell replay from live, and cannot see the future |
| Whole field: order, stop cost, rejoin position | `live_field.py` | "Who am I behind when I come out?" takes all the cars |
| Full pipeline on every line crossing and on any flag | `live_engine.py` | About 5 ms, so a safety car changes the call at once |
| Stable calls, tracked by the lap to box on | `live_engine.py` | A stop must beat staying out by 2 points or 1 s, and a held call needs 3 points to change |
| Plain-language screen | `live_view.py` | No figures and no dependencies, with a plain mode for logs |
| Values learned from 2023-24 | `priors.py` | Stop cost and wear by circuit, safety car and VSC stop costs, normal stint lengths |

If the feed drops, the screen keeps the last state and shows the outage. If a calculation
fails, the last call stays up marked `STALE`. If the rival retires, the target moves to the
next car and says so.

### How good is it, on real races it never saw

Each race was played through the same engine as the screen, then compared with what really
happened: 21 dry races from 2025 and 18 from 2026, with four pairs of cars per race, chosen
by position at lap 5. Settings were fitted on 2023-24 only. Rerun with
`python -m pitwall.live_eval`.

| Measure | 2025 | 2026 |
|---|---|---|
| **Win chance**, Brier score (lower is better) | **0.111** | **0.129** |
| Gap alone, a fair baseline | 0.129 | 0.170 |
| "Current order holds" | 0.155 | 0.219 |
| Coin flip | 0.250 | 0.250 |
| Next-lap pace, mean error | 0.47 s | 0.48 s |
| Pace 10 laps ahead on the same tyres, mean error | 0.69 s | 0.67 s |
| Rejoin position within 2 places (mean error) | 76% (1.6) | 75% (1.8) |
| Rival stops flagged in the 3 laps before | 35% | 43% |
| Call changes per race | 4.7 | 5.0 |

When the screen says a win chance, this is how often it came true:

| Screen says | 0-10% | 10-30% | 30-50% | 50-70% | 70-90% | 90-100% |
|---|---|---|---|---|---|---|
| 2025 | 3% | 16% | 38% | 71% | 86% | 98% |
| 2026 | 4% | 29% | 46% | 67% | 81% | 95% |

Calculation takes about 5 ms per decision at 1000 simulations per option (worst 58 ms over
353 decisions), against a lap of about 80 s.

**How to read this:**
- **Win chance is the strong part.** It is scored for the option the team really took, since
  the others never happened. That shows the forecast is sound, not that a call was the best one.
- **It loses to the simple gap baseline in the last 10 laps:** 0.057 against 0.042 in 2025,
  0.039 against 0.026 in 2026. Near the flag, trust the gap.
- **Absolute finish gaps are not reliable.** The mean error is about 14 s, no better than
  assuming the gap stays as it is (11 s in 2025, 15 s in 2026), so the screen only shows time
  between options.
- **Most rival stops come without warning.** More than half are not flagged in advance.
- **Calls cannot be scored directly.** Real teams made other choices, so nothing here proves
  a call would have gained a place. The plan's stop lap is a median 4 laps from the team's
  real stop, measured 5 laps before it.
- **The test years are not perfectly clean.** Nothing was fitted on them, but looking at their
  scores led to the win-chance blend and the race-long pace gap.
- **In a replay the cars do what really happened.** The screen says so, and repeated box calls
  or "not taken" alerts are expected there.

### Before using it in a real race

- **Live access is not proven in a running session.** The parser has run on real OpenF1 rows
  from finished 2026 races and on a local fake server with outages and logins. No race was
  running during this work, so the first live session should be a rehearsal (next: Brazil, 8 Nov).
- **Polling adds delay.** Up to 8 s, plus OpenF1's own. Check it before relying on it for a call
  near the pit entry.
- **2026 cars are outside the training data.** Tyre and hazard settings come from 2023-24,
  though the 2026 scores held up.
- **A real team has better data.** Team telemetry, tyre sets and radio are not used, except
  for the unused private-wear slot in `owncar.py`.

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

pip install -e ".[dev]" && pytest test_pitwall.py test_live.py   # tests
```

All of this works offline. The repo bundles 89 races (2023 to Mexico 2026, about 100k laps)
from [OpenF1](https://openf1.org), which needs no login. Rebuild the data or the
results with:

```
python -m pitwall.openf1          # refetch the bundled races (about 15 min)
python -m pitwall.train_hazard    # retrain and re-score the hazard model
python -m pitwall.evaluate        # rerun the strategy comparison
python -m pitwall.priors          # relearn circuit values from 2023-24
python -m pitwall.live_eval       # rescore the live terminal on real races
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
- **Not modelled:** future safety cars, and the way one bunches the field up. In the live
  terminal a stop under a current safety car or VSC is cheaper (67% and 81% of the green
  cost, measured), and rejoin position counts traffic, but the race simulation is still
  two cars. The dashboard and demo keep a full pit loss.
- **Tyre cliff:** drivers manage pace, so lap times often show little wear. In the live
  terminal, each lap past the circuit's normal stint length costs an assumed extra 0.25 s.
- **Tyre choice:** the terminal checks which compounds normally reach the flag. It does
  not know which sets a team has left, and it does not race one compound against another.
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
