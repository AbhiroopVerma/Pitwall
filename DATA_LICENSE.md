# Data licence

The race data in this repository comes from [OpenF1](https://openf1.org), which is published under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). OpenF1 describes itself as
intended for education, personal learning, research and non-commercial fan engagement.

These files are shared under the same licence, so they may not be used commercially:

| Files | Contents | Changes from OpenF1 |
|---|---|---|
| `pitwall/assets/openf1_laps.csv.gz` | Lap rows for 89 races | Reduced to selected fields, with stops, tyre age and neutral laps added |
| `pitwall/assets/openf1_neutral.json` | Safety car, VSC and red flag periods | Derived from race-control messages |
| `fixtures/openf1_mexico_2026.json.gz` | OpenF1 rows from one race, for tests | Reduced to one session |
| `pitwall/assets/hazard_weights.json`, `race_priors.json`, `live_eval.json`, `strategy_eval.json`, `copilot_eval.json` | Fitted values and scores | Calculated from the data above |

The source code is under the MIT licence in [`LICENSE`](LICENSE). That licence doesn't extend to these
files, or to data fetched from OpenF1 when the code runs.

OpenF1 is unofficial, and neither it nor PITWALL is associated with Formula 1, the FIA or Formula One
Management. F1, FORMULA ONE, FORMULA 1 and GRAND PRIX are trademarks of Formula One Licensing B.V.

A team using this commercially would need its own licensed data feed, and OpenF1 asks anyone with other
uses in mind to [contact them](https://openf1.org).
