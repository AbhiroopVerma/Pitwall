# Three-minute demo script

Uses the real 2026 Mexico race, LEC against VER. Every number below came from a real run, but
Muse Spark's wording changes between runs, so read what is on screen.

## Set up first

```bash
cd Pitwall && source .venv/bin/activate
export MODEL_API_KEY="your-key"     # or keep it in .env
```

- Make the terminal at least 100 columns wide and 50 lines tall, with a large font.
- Don't show your key or `.env` on screen.
- Practise once. The replay runs at 10 race seconds per real second, so a lap takes about 8 s. Press
  `space` as soon as the header turns to SAFETY CAR.

## Script

| Time | Show | Say |
|---|---|---|
| 0:00 | This README's first screen | "In a race, a strategist has seconds to decide whether to stop, with the rival's plan hidden. We built PITWALL during this hackathon: a live strategy terminal that uses Meta's Muse Spark to explain its calls." |
| 0:20 | Run `python -m pitwall.live --start-lap 40 --speed 10` | "This is the real 2026 Mexico race, replayed one event at a time. The engine can't see the future. It predicts when the rival will stop, then simulates each option to the flag, 1,000 races per option, in about 5 milliseconds." |
| 0:45 | The call box and option table | "The call and the words for the driver come first. Below that, every option has its win chance and where we'd rejoin." |
| 1:00 | SAFETY CAR appears on lap 43. Press `space` | "A real safety car. The call stays BOX, but the stop cost drops from 22.9 seconds to 17.4, and the radio message changes to match." |
| 1:20 | Press `a`, ask: `What does a stop cost under this safety car, compared with green?` | "Now Muse Spark. The engineer can ask in plain words, and the replay waits for the answer here. In a live session the race carries on." |
| 1:40 | Press `a`, ask: `Why this call, and why not the next best option?` | "It only explains. The label says the call is PITWALL's, and a test confirms that its answers can't change the call." |
| 2:00 | Press `a`, ask: `How much fuel do we have left?` | "Fuel isn't in our data, so it says NOT IN DATA instead of guessing." |
| 2:10 | Run `pytest test_copilot.py -k wrong_sum -v` in a second window | "In testing, the model got a sum wrong: 19.8 seconds behind instead of 26.0. So PITWALL now supplies the sums and checks every number in an answer. If a number isn't in the facts, the answer gets one retry and is then withheld." |
| 2:25 | README results table | "We tested on real races it never trained on. Win chance beat the gap-only baseline, 0.111 to 0.129 in 2025 and 0.129 to 0.170 in 2026, and lower is better. Muse Spark got 45 of 45 known values right and refused all 9 questions it couldn't know, in a small test of 63 questions." |
| 2:45 | README's threats to validity | "We publish the weak spots too. In the last 10 laps the gap alone predicts better, most rival stops come without warning, and we can't prove a call would have gained a place." |
| 2:55 | The terminal | "PITWALL: real-time decisions under uncertainty, with Muse Spark explaining them." |

## Rules from the participant guide

- Show only what you built during the hackathon. All of the above qualifies, and the README lists it by
  commit and date.
- The race replays real events, so the cars do what really happened, not what the screen calls for.
- Make sure the video link opens without a login, and add it to the requirements table in the README.
