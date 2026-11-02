"""Module 7 — decision optimisation: U(a) = w1*P(ahead) - w2*deficit - w3*risk."""
from __future__ import annotations

from dataclasses import dataclass

from .simulator import ActionResult


@dataclass
class Recommendation:
    best_offset: int
    ranked: list[tuple[int, float]]
    explanation: str


def optimise(results: list[ActionResult], w1: float = 1.0, w2: float = 0.15,
             w3: float = 0.10, opp_p3: float = 0.0) -> Recommendation:
    scored = []
    for r in results:
        deficit = max(0.0, -r.exp_gap)
        tail = max(0.0, -r.risk)
        u = w1 * r.p_ahead - w2 * deficit - w3 * tail
        scored.append((u, r))
    scored.sort(key=lambda t: -t[0])
    ranked = [(r.our_pit_offset, u) for u, r in scored]
    best = scored[0][1]
    verb = "pit now (cover the undercut)" if best.our_pit_offset == 0 else f"pit in +{best.our_pit_offset} laps"
    explanation = (
        f"Opp 3-lap pit probability is {opp_p3:.0%}. Best response: {verb}: "
        f"P(stay ahead)={best.p_ahead:.0%}, expected gap={best.exp_gap:+.1f}s, "
        f"worst-5%={best.risk:+.1f}s. Staying out longer risks the undercut on "
        f"their fresh tyres; pitting earlier pays pit loss but buys pace."
    )
    return Recommendation(best.our_pit_offset, ranked, explanation)
