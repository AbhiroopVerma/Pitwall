"""Module 6 — 1v1 Monte Carlo counter-strategy simulation.

For each candidate action a in {pit now, +1, ..., +N}:
  repeat n_sims: sample opp pit lap ~ pit_dist, roll both cars forward over
  horizon H laps with pace forecasts + Gaussian noise, apply pit_loss on pit
  laps, record final relative gap. Undercut/overcut emerge from the dynamics:
  no special-case code, just pace deltas + pit losses.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ActionResult:
    action: str
    our_pit_offset: int
    p_ahead: float
    exp_gap: float      # + = we finish ahead (s), mean over sims
    risk: float         # P5 worst gap (downside): lower is worse
    gaps: list[float]


def simulate_action(own_stay: list[float], own_fresh: list[float],
                    opp_stay: list[float], opp_fresh_pace: float,
                    pit_dist: list[float], gap0: float, pit_loss: float,
                    our_offset: int, opp_base_age_pace: list[float],
                    horizon: int, noise: float, n_sims: int,
                    rng: np.random.Generator) -> ActionResult:
    """own_stay/fresh: our forecast pace; opp handled via sampled pit lap."""
    gaps = []
    k = len(pit_dist)  # = forecast k + 1 (last = no pit within window)
    for _ in range(n_sims):
        opp_pit_in = int(rng.choice(len(pit_dist), p=np.asarray(pit_dist) / sum(pit_dist)))
        gap = gap0  # + = we trail
        for h in range(horizon):
            our_pitted = (h == our_offset)
            opp_pitted = (opp_pit_in < k - 1 and h == opp_pit_in)
            # our lap
            if h < our_offset:
                ours = own_stay[h] if h < len(own_stay) else own_stay[-1]
            elif h == our_offset:
                ours = (own_fresh[0] if own_fresh else own_stay[h]) + pit_loss
            else:
                idx = h - our_offset - 1
                ours = own_fresh[idx] if idx < len(own_fresh) else own_fresh[-1]
            # opp lap: old-pace until their stop, fresh after
            if opp_pit_in >= k - 1:  # never pits in window
                opps = opp_base_age_pace[h] if h < len(opp_base_age_pace) else opp_base_age_pace[-1]
            elif h < opp_pit_in:
                opps = opp_stay[h] if h < len(opp_stay) else opp_stay[-1]
            elif h == opp_pit_in:
                opps = opp_fresh_pace + pit_loss
            else:
                opps = opp_fresh_pace + 0.06 * (h - opp_pit_in)  # fresh tyre degrades slowly
            gap += (ours - opps) + rng.normal(0, noise)
            _ = our_pitted, opp_pitted
        gaps.append(-gap)  # + = we are ahead
    gaps = np.asarray(gaps)
    return ActionResult(f"+{our_offset}" if our_offset else "now", our_offset,
                        float(np.mean(gaps > 0)), float(np.mean(gaps)),
                        float(np.percentile(gaps, 5)), list(map(float, gaps)))
