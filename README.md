# PITWALL

**Real-time race strategy decisions under uncertainty, from public timing data.**

PITWALL follows one Formula 1 car against one rival. On every lap it predicts when the rival
will stop, simulates each of our options to the flag, and gives the race engineer one stable
call and the words to say to the driver. The engineer can then ask Meta's Muse Spark about that
call and get a short answer built only from PITWALL's numbers, with every number checked.

```
LAP 49/71   GREEN   LEC P4  vs  VER P1  line in  27s

  STAND BY  -  BOX END OF LAP 52 (3 laps after this)

RADIO  "Plan is box end of lap 52. Stay in the window till then."
       confidence LOW, call since lap 48
       Win chance 45%; next best box lap 51 44%, -0.3s at the flag; rejoin P18, clear air

  OPTION                WIN     TIME  STOPS  REJOIN
  box end lap 49        42%    -1.2s      1  P18 clear
  box end lap 50        43%    -0.7s      1  P18 clear
  box end lap 51        44%    -0.3s      1  P18 clear
> box end lap 52        45%     call      1  P18 clear
  stay to flag          16%    -0.7s      0

VER STOP  this lap  2%   within 3  5%   within 5 10%   likely lap 53

MUSE SPARK explains only, the call above is PITWALL's
Q (lap 49, 1 lap ago): Why not box now, and where would we rejoin?
Boxing now leaves us 26.6 s behind VER. Plan is box end of lap 52 to rejoin P18 in clear air.
```
*Real output from a replay of the 2026 Mexico race, trimmed and rearranged, with the answer shown
below the call data rather than directly under the call. It is decision support only: the
cars in a replay do what really happened, and options that were not taken are simulated.*

## Contents

- [Meta Global AI Developer Hackathon](#meta-global-ai-developer-hackathon)
- [Run it](#run-it)
- [Research method](#research-method)
- [Results](#results)
- [Threats to validity](#threats-to-validity)
- [Reproduce the research](#reproduce-the-research)
- [Code map](#code-map)
- [Assumptions and limits](#assumptions-and-limits)

## Meta Global AI Developer Hackathon

**Problem statement:** Build for Your World. The domain problem is making a stop decision within
seconds, when the rival's plan is hidden and each option can only be judged by simulation.

| Requirement | Status | Evidence |
|---|---|---|
| Uses Meta's models | Done | Muse Spark (`muse-spark-1.3`) through the Meta Model API, in [`pitwall/copilot.py`](pitwall/copilot.py) |
| Public repository | Done | [AbhiroopVerma/Pitwall](https://github.com/AbhiroopVerma/Pitwall) |
| New work, built during the event | Done | Git history below, all dated 1 to 4 Nov 2026 |
| Three-minute demo video | Not linked yet | |
| Rights to data and code | To confirm | Data comes from [OpenF1](https://openf1.org), which is unofficial. Its terms need checking |

### How Muse Spark is used

The engine makes the call and Muse Spark explains it, so a strategist can question a number in
plain words without looking away to read a table. Facts flow one way, so no answer can change a call:

| Safeguard | How it works |
|---|---|
| Facts only | The model sees the current race state as JSON and must reply `NOT IN DATA` to anything else |
| Sums supplied | "If we box now" and "if the rival boxes now" are worked out by PITWALL, because the model got one wrong in testing: 19.8 s behind instead of 26.0 s |
| Numbers checked | An answer quoting a number not in the facts gets one retry, then is withheld and labelled on screen |
| Never in the way | Requests run on a background thread. A slow reply, an API failure or a missing key leaves the calls unaffected |
| Clearly labelled | Answers sit under a heading that says the call is PITWALL's, and show how many laps old they are |

### What was built during the hackathon

| Commit | Date | Work |
|---|---|---|
| `a4ecb9e` | 1 Nov | Repository created |
| `20c0283` | 2 Nov | First 1v1 engine: tyre model, hazard model, simulator, optimiser, dashboard |
| `7a0e376` | 3 Nov | Training on real races, season-based testing, leakage fixes, fairer simulator |
| `c695087` | 3 Nov | Live terminal, learned circuit values, real-race scoring, algorithm paper |
| `d8d1a91` | 4 Nov | Packaging metadata |
| Latest | 4 Nov | Muse Spark integration, its safeguards and its evaluation |

Built with Muse Spark, numpy, pandas, Streamlit, Plotly and FastF1, using data from OpenF1. Code was
written with AI assistance from Claude Code. The maths is written up in
[`docs/pitwall_algorithm.pdf`](docs/pitwall_algorithm.pdf).

## Run it

Needs Python 3.10 or newer. Race data is bundled, so the replays work offline.

```bash
git clone https://github.com/AbhiroopVerma/Pitwall.git && cd Pitwall
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m pitwall.live                                  # real 2026 Mexico race, LEC vs VER
python -m pitwall.live --fabricated safety-car          # or: undercut, rival-out
python -m pitwall.live --race 2025 Monza --own LEC --opp NOR --start-lap 25
python -m pitwall.live --list                           # every bundled race

pytest test_pitwall.py test_live.py test_copilot.py     # 138 tests, no API key needed
```

`--speed` sets race seconds per real second. The default is 20, and `--speed 1` is true real
time. While it runs: `a` ask Muse Spark, `space` pause, `+`/`-` speed, `n` next lap, `t` change
rival, `q` quit.

To turn on Muse Spark, create a key in the Meta Model API dashboard and either export it or put it
in a `.env` file, which git ignores. Everything else works without one.

```bash
export MODEL_API_KEY="your-key"
python -m pitwall.live --fabricated safety-car --brief   # explains each new call, uses API credit
python -m pitwall.live --plain --start-lap 49 --until-lap 50 --ask "Why not box now?"
```

In a replay, pressing `a` pauses the race until the answer arrives. In a live session the race
carries on, and the answer says how many laps old it is.

| Source | Data | Needs |
|---|---|---|
| `--race` | 89 real races, bundled | Nothing |
| `--fabricated` | Invented 20-car race, labelled on screen | Nothing |
| `--openf1 KEY --rehearse` | Real OpenF1 rows released on schedule through the live parser | Internet |
| `--openf1 latest` | The session running now, polled every 8 s | Paid OpenF1 account (`OPENF1_USERNAME`, `OPENF1_PASSWORD`), plus `--laps N` for a new circuit |

Also available: `--plain --log race.jsonl` prints each decision and keeps a log, and
`streamlit run pitwall/app.py` opens the dashboard.

## Research method

### Questions

1. **Prediction:** from public timing alone, how well can we predict a rival's stop?
2. **Forecasting:** is the win chance on screen trustworthy, and does it beat simple rules?
3. **Decision:** does planning to the flag beat rule-based strategies?
4. **Explanation:** can a language model answer questions about a call accurately enough to use on
   a pit wall, and refuse what it can't know?

Real races can answer the first two. The third needs simulation, because each race shows only the
choice the team really made. The fourth is tested on facts from real races.

### Design

| Stage | What was done |
|---|---|
| Data | Lap, stint and race-control records from OpenF1 for 89 races |
| Split | By season: fit on 2023-24, test on 2025 and 2026 |
| Models | Interpretable statistical models and Monte Carlo simulation make every forecast and call. Muse Spark only explains them |
| Baselines | Every result is compared with a simple alternative |
| Replay | Test races run through the production engine, one event at a time |
| Reporting | Weak results are published next to strong ones |

### Data

| Season | Races | Car-laps | Stops |
|---|---:|---:|---:|
| 2023 | 22 | 24,479 | 912 |
| 2024 | 24 | 26,629 | 825 |
| 2025 | 24 | 26,265 | 815 |
| 2026, to Mexico | 19 | 22,180 | 779 |
| **Total, 26 circuits** | **89** | **99,553** | **3,331** |

- **Stops:** a lap counts as a stop when OpenF1 marks the next lap as a pit-out lap.
- **Neutral laps:** a lap is neutral if it overlaps a safety car, VSC or red flag period from
  race-control messages. Those laps are left out of pace fitting.
- **Clean laps:** pace models drop lap 1, in-laps, out-laps, neutral laps, and laps over 3 s from
  their stint's median after fuel correction.
- **Dry races only:** wet races follow different logic, so they are excluded, and in wet conditions
  the engine stays on MONITOR. The cut-off varies by study, as the table shows.

| Study | Keeps a race if | Fit | Test |
|---|---|---|---|
| Hazard model | Under 5% of laps on wet tyres | 41 races | 22 races, 2025 |
| Circuit values | No laps on wet tyres | 2023-24, dry races | Used in the live scoring |
| Live terminal | Under 2% of laps on wet tyres | 41 races | 21 in 2025, 18 in 2026 |

### Preventing leakage

An early version scored a Brier of 0.001, which was too good to be true: one feature already
contained the stop being predicted. After that, these controls went in:

- **Features use earlier laps only.** The stop count excludes the stop being predicted.
- **Events are released only when knowable.** The engine reads a feed and cannot tell a replay from
  a live race.
- **Tests enforce it.** `test_changing_future_laps_cannot_change_a_decision` and
  `test_engine_output_does_not_depend_on_the_future` alter later data and require the same output.
- **Rival pairs are chosen at lap 5.** Choosing neighbours at the finish would use the result to pick
  the test. Which car counts as "ours" alternates.

### Models

**Tyre and pace (`degradation.py`).** Each car gets a fit from its own clean laps:

```
lap time - fuel = stint baseline + b1·a + b2·a²,   a = age × compound factor / 10,   b1, b2 ≥ 0
```

A baseline per stint keeps a compound change from being read as wear, and the non-negative terms rule
out tyres that get faster with age. In the live terminal, a wear curve learned for each circuit steers
the fit while a stint is too short to show its own wear (ridge strength 10).

**Rival stop hazard (`pit_model.py`).** A discrete-time logistic model, fitted with L2-regularised
Newton steps on 43,719 car-laps and 1,402 stops:

```
h(t) = P(stop at end of lap t | still running) = σ(β·X)
P(stop within k laps) = 1 − ∏ (1 − h)
```

Features: tyre age and age², compound and its interaction with age, race progress and its square,
final five laps, first stop still pending, stops made, wear penalty and slope, and neutral period
with its interaction with age. The loss is unweighted so outputs remain real probabilities.

**Race simulation (`simulator.py`).** For each option, 1,000 races are rolled lap by lap to the flag:

- The rival's next stop is drawn from the hazard forecast.
- After the first choice, each car follows its cheapest remaining plan, found by dynamic programming.
- Lap noise, the unknown next compound (±0.4 s) and wear-slope error are sampled.
- A car that catches another is held up unless it is 1.0 s/lap faster. Cars can swap places in the pits.
- All options share the same random draws, so differences between them are not Monte Carlo noise.

**Decision (`optimizer.py`, `live_engine.py`).**

```
U = P(ahead at flag) − 0.25·deficit − 0.10·tail risk     (seconds scaled by pit loss, capped at 1)
```

Calls have to stay usable, so a stop must beat staying out by 0.02 utility or 1 s, because a stop
cannot be undone and waiting can. A held call changes only for a gain of 0.03 utility, or when the
race changes.

**Displayed win chance (`priors.py`).** The raw simulation is recalibrated against real results:

```
logit(win) = 0.72·logit(p_sim) + 0.234·clip(z, ±6),     z = gap ahead / (0.75·√laps left)
```

The gap term is the same for all options and 0.72 is positive, so this rescales the number shown and
cannot change which option ranks first.

**Whole field (`live_field.py`).** Rejoin position is projected across every running car, including
lapped traffic, assuming that they hold recent pace and do not stop.

**Explanation (`copilot.py`).** Muse Spark 1.3 runs at `minimal` reasoning, because `low` and `medium`
used up a 600-token limit on reasoning and returned no answer. Each question goes out with the current facts.
Every number in the reply is compared with the facts, allowing only for rounding, so 23 s may quote
22.9 s but 22.5 s may not.

### What is measured, fitted or assumed

| Value | Number | Source |
|---|---|---|
| Green-flag stop cost by circuit | 18.8 to 29.4 s, default 22.7 s | Measured on 2023-24: median against cars not stopping, across in-lap and out-lap |
| Stop cost in this race | Median of stops seen | Measured live, replaces the circuit value after 3 stops |
| Stop cost under safety car, VSC | 67%, 81% of green | Measured on 2023-24 |
| Chance a period lasts another lap | Safety car 74%, VSC 38% | Measured on 2023-24 |
| Normal stint length (90th percentile) | Soft 24, medium 30, hard 40 laps; varies by circuit | Measured on 2023-24 |
| Wear curve by circuit | 24 circuits | Fitted on 2023-24 |
| Wear prior strength | 10 | Tuned on training races |
| Pace gap | Race-long gap (shrinkage τ = 0.0001) | Tuned on training races |
| Win-chance blend | 0.72, 0.234, 0.75 | Fitted on 2023-24 |
| Fuel effect | 0.04 s/lap | **Assumed:** lap times cannot separate fuel from tyre age in a first stint |
| Compound wear rates | Soft 1.4, medium 1.0, hard 0.7 | **Assumed** |
| Overtaking threshold | 1.0 s/lap | **Assumed:** adjustable, and real circuits differ a lot |
| Wear past the normal stint | 0.25 s per extra lap | **Assumed:** too few stints run that long to measure it |
| Utility weights, call margins | See above | **Chosen by design** |

### How each question was evaluated

**1. Stop prediction.** Fitted on 41 dry races from 2023-24 and scored on 22 dry races from 2025,
against a tyre-age model and a constant base rate. Scores cover both per-lap predictions and the
three-lap warning at a 30% threshold.

**2. Live terminal.** Each test race was played through `LiveEngine`, one event at a time. Each race
has four pairs, taken from the top eight cars at lap 5. Decisions were simulated at 300 races per
option for speed. A forecast is scored only for the option the team really took, since the others
never happened.

| Measure | What it compares |
|---|---|
| Win chance | Brier score against the real result, compared with gap alone, current order and a coin flip |
| Calibration | How often each stated win-chance band came true |
| Pace | Next-lap forecast, and forecast 10 laps ahead in the same stint, against real lap times |
| Rejoin | Predicted position against the real one, taken one lap before a real green-flag stop |
| Rival warnings | Share of real stops flagged in the three laps before |
| Stability, speed | Call changes per race, and calculation time per decision |

**3. Strategy.** Five policies drove the same 300 randomised synthetic races with the same noise, so
the comparison is paired. Policies saw only past laps. Confidence intervals come from 2,000 bootstrap
resamples of the paired differences.

**4. Explanation.** Facts were taken from nine decisions in three 2026 races: Australia, Spain and
Italy, with one pair each, at three points in each race. Each decision got six questions through the
real API: four with a known numeric answer (gap, stop cost, win chance, rival stop chance), one asking
why, and one that the facts can't answer (fuel left, or what the driver said on the radio).

## Results

### 1. Stop prediction, on 22 unseen 2025 races

23,764 car-laps and 685 stops.

| Per-lap predictor | Brier | Log loss |
|---|---:|---:|
| **PITWALL hazard model** | **0.0274** | **0.1199** |
| Tyre age only | 0.0278 | 0.1274 |
| Constant base rate | 0.0280 | 0.1308 |

The three-lap warning at 30% came before 34% of 636 stops and fired on 6.7% of laps with no stop
coming. Predictions run high above about 30%: those of 40-50% came true 25% of the time.

**The edge over the baselines is real but small.** Most stops arrive without much warning.

### 2. Live terminal, on unseen races

21 dry races in 2025 and 18 in 2026, with 3,663 and 3,091 scored decisions.

| Measure | 2025 | 2026 |
|---|---:|---:|
| **Win chance, Brier (lower is better)** | **0.111** | **0.129** |
| Gap alone, a fair baseline | 0.129 | 0.170 |
| Current order holds | 0.155 | 0.219 |
| Coin flip | 0.250 | 0.250 |
| Next-lap pace, mean error | 0.47 s | 0.48 s |
| Pace 10 laps ahead, mean error (bias) | 0.69 s (+0.01) | 0.67 s (+0.01) |
| Rejoin within 2 places | 76% of 117 | 75% of 97 |
| Rejoin, mean error | 1.6 places | 1.8 places |
| Rival stops flagged in advance | 35% of 130 | 43% of 101 |
| Call changes per pair | 4.7 | 5.0 |
| Stop plan vs team's real stop, 5 laps before (median) | 4 laps | 4 laps |

How often each stated win chance came true:

| Screen says | 0-10% | 10-30% | 30-50% | 50-70% | 70-90% | 90-100% |
|---|---:|---:|---:|---:|---:|---:|
| 2025 | 3% | 16% | 38% | 71% | 86% | 98% |
| 2026 | 4% | 29% | 46% | 67% | 81% | 95% |

Win-chance Brier by laps remaining:

| Laps left | 2025 PITWALL | 2025 gap alone | 2026 PITWALL | 2026 gap alone |
|---|---:|---:|---:|---:|
| 26 or more | 0.149 | 0.173 | 0.169 | 0.225 |
| 11 to 25 | 0.077 | 0.104 | 0.113 | 0.156 |
| 1 to 10 | **0.057** | **0.042** | **0.039** | **0.026** |

**What this shows:**
- **Win chance is the strong part,** and it beats every baseline until the final laps.
- **In the last 10 laps it loses to gap alone.** Near the flag, trust the gap.
- **Absolute finish gaps are not reliable.** The mean error is 13.9 s in 2025 and 14.5 s in 2026.
  Assuming the gap stays as it is gives 11.3 s and 15.2 s, so the screen only shows time between options.
- **Most rival stops come without warning,** in line with the hazard results.

### 3. Strategy, in 300 simulated races

| Policy | P(ahead at flag) | Mean gap | Mean stops | PITWALL's gain, points (95% CI) |
|---|---:|---:|---:|---:|
| **PITWALL** | **70%** | **+17.2 s** | 2.24 | |
| PITWALL ignoring the opponent forecast | 70% | +16.9 s | 2.32 | 0 (−1.7 to +1.7) |
| Tyre-age rule | 50% | +4.4 s | 2.00 | +20 (16 to 25) |
| Reactive, cover their stop | 44% | +2.6 s | 1.85 | +26 (20 to 31) |
| Fixed one-stop | 35% | −9.8 s | 1.00 | +34 (28 to 40) |

**What this shows:**
- **The gain comes from modelling our own tyres and planning to the flag.** The opponent forecast adds
  no measurable benefit here, though these simulated opponents do not react to us, which limits what
  predicting them can add.
- **The test favours PITWALL.** These races share the simulator's hold-up rule, and PITWALL gets a
  synthetic private wear signal. They show good decisions inside this model's world, not proof that it
  would win real races.

### 4. Muse Spark explanations, on 2026 races

| Measure | Result |
|---|---:|
| Known-value questions answered correctly | 36 of 36 |
| Unanswerable questions refused with `NOT IN DATA` | 9 of 9 |
| Explanations returned | 9 of 9 |
| First attempts quoting a number not in the facts | 0 of 54 |
| Answers withheld, API errors | 0, 0 |
| Response time, median (worst) | 2.2 s (6.7 s) |
| Tokens for the whole run | 74,952 |

**What this shows:**
- **Answers stayed on the facts,** once PITWALL supplied the sums. The number check never had to step
  in during this run, so its retry and withholding are covered only by tests with fake replies.
- **Explanations mostly restate PITWALL's numbers.** They are accurate and short, but they compare win
  chance and time rather than add strategic reasoning.
- **The sample is small:** 54 questions on three races, with no error bars.
- **A correct answer means the right number appears in the reply.** The check doesn't read the sentence
  around that number.

### Speed

At 1,000 simulations per option, calculation took about 5 ms per decision, with a worst case of 58 ms
over 353 decisions, against a lap of about 80 s. In the larger scoring run at 300 simulations, the
median was 5 ms and the 99th percentile 24 ms, but the slowest decisions took 2.7 s (2025) and 1.3 s
(2026). The cause of those outliers has not been investigated.

## Threats to validity

**In the evaluation:**
- **Calls themselves are not scored.** Teams made other choices, so nothing here proves a call would
  have gained a place. The win-chance score shows that the forecast is sound, not that a call was the best.
- **The test years are not perfectly clean.** Nothing was fitted on them, but seeing their scores led to
  the win-chance blend and the race-long pace gap.
- **No confidence intervals on real-race scores.** Decisions within a race are correlated, so the
  decision counts overstate the evidence.
- **Selection:** win-chance scoring needs both cars to reach the flag, so retirements are excluded. Pairs
  come only from the top eight cars at lap 5. Decisions made on a projected rival lap are not scored.
- **False alarms are measured only for the hazard model** on 2025, not in the live terminal.
- **Hazard scores cover 2025 only.** 2026 enters only through the live terminal scores.
- **Scoring used 300 simulations per option,** and the live default is 1,000.
- **The number check can't judge reasoning.** A wrong statement built from correct numbers would pass,
  for example a sign error or the wrong driver. Muse Spark's answers aren't for reading out unchecked.

**In the models:**
- **The opponent does not react** to our stop, and the hazard model ignores the gap to us.
- **Future safety cars are not simulated.** A current one makes a stop cheaper, and that is all it changes.
  The way it bunches up the field is not modelled in the two-car race simulation.
- **Pace holds to the flag,** so large pace differences compound, which is one reason absolute gaps run wide.
- **2026 cars are outside the fitting data,** though the 2026 scores held up.

**In live use:**
- **Live polling has not run during a real session.** The parser has run on real rows from finished 2026
  races and on a local fake server with outages and logins. The first live session should be a rehearsal.
- **Polling adds delay:** up to 8 s, plus OpenF1's own.
- **A real team has better data.** Telemetry, remaining tyre sets and radio are not used.

## Reproduce the research

```bash
python -m pitwall.openf1             # refetch the bundled races, about 15 min, no login
python -m pitwall.priors             # relearn circuit values from 2023-24
python -m pitwall.train_hazard       # refit and rescore the hazard model
python -m pitwall.live_eval          # refit the win blend, rescore the live terminal
python -m pitwall.evaluate --races 300   # rerun the strategy comparison
python -m pitwall.copilot --eval     # rescore Muse Spark, needs MODEL_API_KEY and uses API credit
```

Stored results are in `pitwall/assets/`: `hazard_weights.json`, `live_eval.json`,
`strategy_eval.json`, `copilot_eval.json` and `race_priors.json`.

`python -m pitwall.live_eval --calibrate` retunes the wear prior strength and the pace settings by
coordinate search, on every second training race with three pairs, scoring every fourth lap. The
shipped τ = 0.0001 came from a follow-up sweep below the search grid in the code, which starts at 0.1, so
that command alone will not reproduce it.

## Code map

```
OpenF1 or bundled laps → event feed → whole-field state → tyre models → rival hazard
    → Monte Carlo to the flag → utility ranking → stable call → text screen
                                                        ↓ facts, one way
                                 Muse Spark → number check → labelled answer on screen
```

| File | Role |
|---|---|
| `copilot.py` | Muse Spark client, facts, number check, background worker, evaluation |
| `live.py`, `live_view.py` | Command line, clock, keyboard, text screen |
| `live_engine.py` | Runs the pipeline on each event, keeps the call stable, raises alerts |
| `live_feed.py` | Replay, rehearsal and live OpenF1 feeds |
| `live_field.py` | Race order, measured stop cost, rejoin position |
| `priors.py` | Values learned from 2023-24 |
| `degradation.py`, `owncar.py` | Tyre and pace models |
| `pit_model.py`, `train_hazard.py` | Hazard model and its training |
| `threshold.py` | Rival threat alerts |
| `simulator.py`, `optimizer.py` | Simulation and ranking |
| `live_eval.py`, `evaluate.py` | Scoring on real races, and strategy comparison |
| `synthetic.py`, `data.py` | Fabricated races and bundled data |
| `app.py`, `demo.py` | Dashboard and command-line demo |

If the feed drops, the screen keeps the last state and shows the outage. If a calculation fails, the
last call stays up marked `STALE`. If the rival retires, the target moves to the next car and says so.

## Assumptions and limits

- **Tyre choice:** the terminal checks which compounds normally reach the flag. It does not know which
  sets a team has left, and it does not race one compound against another.
- **Tyre wear:** drivers manage pace, so lap times often show little wear until a tyre is finished.
- **Own-car telemetry:** none exists on historical races, so both cars use public lap times there. The
  private wear signal exists only in synthetic races.
- **Dashboard and demo:** they keep a full pit loss under a safety car. Only the live terminal adjusts it.
  Their replays of 2023-24 races are in-sample for the hazard model, so use a 2025 race for a fair view.
- **FastF1:** that option in the dashboard needs an F1 account (`F1_USERNAME`, `F1_PASSWORD`). If it
  fails, the dashboard says so and labels the synthetic race it falls back to.
- **Data:** OpenF1 is unofficial. Check its terms before any use beyond this project.
- **Muse Spark data:** each question sends the current race state to Meta's API. The default model runs in
  standard mode, not the cheaper contributor mode, where content may be used for training.
