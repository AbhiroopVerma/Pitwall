"""Module 5 — probability threshold + threat override (rule-based, no ML)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Trigger:
    active: bool
    status: str          # MONITOR | TRIGGERED
    reason: str
    p_within_k: float
    horizon: int
    theta: float


class ThreatDetector:
    """Activate if p(pit within k) >= theta OR expected loss >= lam."""

    def __init__(self, theta: float = 0.65, horizon: int = 3, lam: float = 3.0):
        self.theta, self.horizon, self.lam = theta, horizon, lam

    def check(self, p_within_k: float, expected_loss_s: float = 0.0) -> Trigger:
        if p_within_k >= self.theta:
            return Trigger(True, "TRIGGERED",
                           f"p(pit within {self.horizon})={p_within_k:.0%} >= theta={self.theta:.0%}",
                           p_within_k, self.horizon, self.theta)
        if expected_loss_s >= self.lam:
            return Trigger(True, "TRIGGERED",
                           f"expected strategic loss {expected_loss_s:.1f}s >= {self.lam:.1f}s",
                           p_within_k, self.horizon, self.theta)
        return Trigger(False, "MONITOR",
                       f"p={p_within_k:.0%} < theta={self.theta:.0%}, loss={expected_loss_s:.1f}s",
                       p_within_k, self.horizon, self.theta)
