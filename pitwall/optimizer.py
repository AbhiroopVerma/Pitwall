"""Module 7 — decision optimisation.

U(a) = P(ahead at the flag) - w2 * deficit - w3 * tail risk

Deficit and tail risk are in seconds, so both are divided by the pit loss and
capped at 1 to put all three terms on the same scale (spec §10).
"""
from __future__ import annotations

from dataclasses import dataclass

from .simulator import ActionResult, action_label


@dataclass
class Recommendation:
    best_offset: int | None
    ranked: list[tuple[int | None, float]]
    explanation: str


def utility(r: ActionResult, pit_loss: float, w1: float = 1.0, w2: float = 0.25,
            w3: float = 0.10) -> float:
    deficit = min(1.0, max(0.0, -r.exp_gap) / pit_loss)
    tail = min(1.0, max(0.0, -r.risk) / pit_loss)
    return w1 * r.p_ahead - w2 * deficit - w3 * tail


def _situation(best: ActionResult, gap0: float, opp_stopped: bool, opp_p3: float) -> str:
    leading = gap0 < 0
    pitting = best.our_pit_offset is not None
    if best.p_ahead < 0.02:
        return ("This fight looks out of reach, so the ranking only shows how to lose the "
                "least time to this car.")
    if best.p_ahead > 0.98 and leading:
        return "We are clear of this car under every option, so this is the fastest way home."
    if opp_stopped:
        if pitting and leading:
            return "The opponent has just stopped, and responding protects our lead from the undercut."
        if pitting:
            return "The opponent has just stopped, and we should take our own stop before their fresh tyres pull away."
        return "The opponent has just stopped, and staying out on track position is modelled as better than responding yet."
    if pitting and leading:
        return f"With a {opp_p3:.0%} chance they stop within 3 laps, stopping first protects our lead."
    if pitting:
        return f"With a {opp_p3:.0%} chance they stop within 3 laps, stopping first is our best chance to undercut."
    if leading:
        return f"With a {opp_p3:.0%} chance they stop within 3 laps, stopping now would give up track position for too little tyre gain."
    return f"With a {opp_p3:.0%} chance they stop within 3 laps, extending keeps our stop for later; a stop now does not gain enough."


def optimise(results: list[ActionResult], pit_loss: float, gap0: float = 0.0,
             opp_p3: float = 0.0, opp_stopped: bool = False) -> Recommendation:
    # Utility differences under 1% are Monte Carlo noise, and when every option is
    # certain to win they vanish altogether. Expected gap then decides, so a car
    # already clear does not take a pointless stop because it was listed first.
    scored = sorted(((utility(r, pit_loss), i, r) for i, r in enumerate(results)),
                    key=lambda t: (-round(t[0], 2), -t[2].exp_gap, t[1]))
    ranked = [(r.our_pit_offset, u) for u, _, r in scored]
    best = scored[0][2]
    head = (f"Recommend: {action_label(best.our_pit_offset)}. "
            f"P(ahead at flag)={best.p_ahead:.0%}, expected gap at flag={best.exp_gap:+.1f}s, "
            f"worst 5%={best.risk:+.1f}s. ")
    if len(scored) > 1:
        other = scored[1][2]
        head += (f"Next best is {action_label(other.our_pit_offset)} "
                 f"({other.p_ahead:.0%}, {other.exp_gap:+.1f}s). ")
    return Recommendation(best.our_pit_offset, ranked,
                          head + _situation(best, gap0, opp_stopped, opp_p3))
