"""Sequential evidence over LRU hypotheses.

Same test as the ISR work, different sensors: Wald's sequential probability
ratio test, with the boundaries derived from two numbers a maintainer
understands rather than from a threshold somebody tuned.

    S  <-  leak^dt * S  +  llr(observation)

    S >= log((1-beta)/alpha)     name the part
    S <= log(beta/(1-alpha))     this line is ordinary; stop watching it closely
    otherwise                    keep watching

Why this and not a classifier. A classifier answers "which of these four is it?"
and must answer something, so on a healthy machine it confidently names the
least-quiet line. This asks a different question — "is there yet enough evidence
to justify a maintenance action?" — and its default answer is *no*. On a machine
that is fine, that is the only correct output, and it is the output that a
per-window classifier structurally cannot produce.

The false-alarm tolerance is a setting here, and it is the number to ask Matthew
for on Monday. One nuisance alert per engine per month and one per year are
completely different systems, and everything downstream — how long we watch
before speaking, how much link we spend — falls out of it.

Three evidence channels, combined per observation. Each is independent of the
others in the sense that matters: they fail in different ways.

  level      how far this line sits above its own baseline, in robust sigmas
  sidebands  present or absent, against what the physics says to expect
  agreement  how closely the measured peak lands on the predicted order

The sideband channel is what stops an inner-race call being made on outer-race
energy, and the agreement channel is what stops a coincidental peak from a
gear mesh or a line-frequency harmonic being read as a bearing fault.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Observation:
    symptom_id: str
    t: float
    z_level: float            # robust sigmas above this line's own baseline
    order_predicted: float
    order_measured: float
    sideband_ratio: float     # sideband energy / carrier energy
    expect_sidebands: bool
    baseline_ready: bool


@dataclass
class Verdict:
    symptom_id: str
    S: float
    upper: float
    lower: float
    decision: str             # name_it | ordinary | watching | no_baseline
    channels: Dict[str, float] = field(default_factory=dict)
    n_obs: int = 0

    @property
    def confidence(self) -> float:
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, self.S))))


class SPRT:
    # Level is the primary channel; the other two are corroboration. Weighting
    # them equally would let a good order match on a quiet line accumulate
    # evidence on its own, which is how a system starts reporting healthy
    # machines.
    W_LEVEL = 1.0
    W_SIDEBAND = 0.45
    W_AGREEMENT = 0.35

    # A line this many robust sigmas above its own baseline is the operating
    # point: below it an observation is evidence *against* a fault. Three sigma
    # on a robust estimator is a deliberate, statable choice, not a fitted one.
    Z_OPERATING = 3.0
    K = 0.55                  # nats per unit of combined evidence

    def __init__(self, false_alarm_rate: float = 0.02, miss_rate: float = 0.10,
                 leak_per_s: float = 0.999):
        self.alpha = false_alarm_rate
        self.beta = miss_rate
        self.leak = leak_per_s
        self.upper = math.log((1 - self.beta) / self.alpha)
        self.lower = math.log(self.beta / (1 - self.alpha))
        self._S: Dict[str, float] = {}
        self._t: Dict[str, float] = {}
        self._n: Dict[str, int] = {}
        self._ch: Dict[str, Dict[str, float]] = {}

    # -- channels ------------------------------------------------------------
    def _level(self, o: Observation) -> float:
        """Signed, saturating. Saturation matters: an enormous z from a loose
        sensor or a cable knock must not dominate the accumulator and force a
        call on one bad window."""
        return max(-1.0, min(1.5, (o.z_level - self.Z_OPERATING) / self.Z_OPERATING))

    def _sideband(self, o: Observation) -> float:
        """Physics says an inner-race defect modulates and an outer-race defect
        does not. Agreement is evidence; disagreement is evidence against."""
        strong = o.sideband_ratio > 0.25
        if o.expect_sidebands:
            return 0.8 if strong else -0.5
        return -0.7 if strong else 0.5

    def _agreement(self, o: Observation) -> float:
        """How close the measured peak landed to the predicted order. Bearings
        slip by a percent or two, so exact agreement is not expected — but a
        peak 5% away is a different line belonging to something else."""
        if o.order_predicted <= 0:
            return 0.0
        err = abs(o.order_measured - o.order_predicted) / o.order_predicted
        return max(-1.0, 1.0 - err / 0.02)

    # -- the test ------------------------------------------------------------
    def observe(self, o: Observation) -> Verdict:
        k = o.symptom_id
        if not o.baseline_ready:
            # Refusing to score is the correct behaviour, and saying so is
            # better than emitting a confident number from four samples.
            return Verdict(k, self._S.get(k, 0.0), self.upper, self.lower,
                           "no_baseline", {"reason": 0.0}, self._n.get(k, 0))

        S = self._S.get(k, 0.0)
        dt = max(0.0, o.t - self._t.get(k, o.t))
        S *= self.leak ** dt

        lv, sb, ag = self._level(o), self._sideband(o), self._agreement(o)
        combined = (self.W_LEVEL * lv + self.W_SIDEBAND * sb + self.W_AGREEMENT * ag)
        llr = self.K * combined
        S = max(self.lower - 2.0, min(self.upper + 2.0, S + llr))

        self._S[k] = S
        self._t[k] = o.t
        self._n[k] = self._n.get(k, 0) + 1
        ch = {"level": round(lv, 3), "sideband": round(sb, 3),
              "agreement": round(ag, 3), "llr": round(llr, 3),
              "z": round(o.z_level, 2)}
        self._ch[k] = ch

        dec = "name_it" if S >= self.upper else ("ordinary" if S <= self.lower else "watching")
        return Verdict(k, S, self.upper, self.lower, dec, ch, self._n[k])

    def state(self, symptom_id: str) -> float:
        return self._S.get(symptom_id, 0.0)

    def expected_observations_to_decide(self, mean_llr: float) -> Optional[float]:
        """Wald's approximation. Printed beside the measured count so the system
        can be checked against its own prediction rather than merely admired."""
        return (self.upper / mean_llr) if mean_llr > 0 else None

    def reset(self, symptom_id: str) -> None:
        for d in (self._S, self._t, self._n, self._ch):
            d.pop(symptom_id, None)
