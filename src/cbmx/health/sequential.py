"""Wald's sequential test, with the bearing-specific channels taken out.

`health/evidence.py` implements the same test for bearings, and its three
channels — level, sidebands, order agreement — are all statements about
rotating-element physics. That is right for a bearing and useless for a
hydraulic pump.

This module is the part that is not about bearings: the accumulator, the
boundaries derived from a stated false-alarm and miss rate, the saturation that
stops one bad window forcing a call, the leak that lets an old opinion decay,
and the refusal to score a line whose baseline is not yet ready. The channels
are supplied by the caller.

The reason to separate them is a claim in the proposal, and the claim should be
falsifiable in the code rather than only in the prose: what cbmx contributes is
not a bearing algorithm, it is a governance layer that can sit on top of any
symptom set that comes with a declared physical direction. If that layer cannot
be lifted off the bearing code and dropped onto a hydraulic rig with no bearings
in it, the claim is false. This file is where it gets lifted off.

Three channels, and what each of them is stopping:

  level          how far the strongest symptom sits above its own baseline, in
                 robust sigmas. Alone, this is an anomaly detector, and an
                 anomaly detector on a machine with four independent faults and
                 a temperature that wanders will report something every day.

  direction      whether the deviation went the way the physics said it would.
                 A cooler that is failing removes LESS heat. If the temperature
                 drop across it has grown, that is not weaker evidence of a
                 cooler fault — it is evidence against one, and it has to be
                 able to push the accumulator back down. Without this channel
                 the test cannot distinguish 'the machine changed' from 'this
                 component is failing'.

  corroboration  whether the component's other independent symptoms agree.
                 Pump leakage should show up as less flow per kilowatt AND as
                 hotter oil. One without the other is more likely a sensor.

The output is one of four words, and the fourth is the one that matters:

  name_it        enough evidence to justify a maintenance action
  ordinary       enough evidence that this component is fine; stop watching hard
  watching       not yet
  no_baseline    this operating point has never been observed healthy, so no
                 statement is available. Not a failure and not a default answer —
                 a distinct output, counted separately, and the only honest
                 thing to say about a regime the machine has not shown us.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Channels:
    """Evidence for one entity at one observation, already reduced to [-1, 1]-ish.

    Each is signed: positive is evidence toward the fault, negative is evidence
    against it. A channel that can only be zero or positive cannot exonerate,
    and a test built only from such channels convicts eventually.
    """

    level: float
    direction: float
    corroboration: float
    detail: Dict[str, float] = field(default_factory=dict)


@dataclass
class Judgement:
    entity: str
    S: float
    upper: float
    lower: float
    decision: str                 # name_it | ordinary | watching | no_baseline
    channels: Dict[str, float] = field(default_factory=dict)
    n_obs: int = 0

    @property
    def confidence(self) -> float:
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, self.S))))


class WaldAccumulator:
    """One sequential test per entity, boundaries from alpha and beta.

    alpha and beta are the two numbers a maintainer can actually argue about —
    how often may this thing cry wolf, and how often may it stay quiet through a
    real fault. Everything else about when to speak follows from them. That is
    the opposite of a tuned threshold, and it is the property to point at when
    somebody asks how the sensitivity was chosen.
    """

    W_LEVEL = 1.0
    W_DIRECTION = 0.45
    W_CORROBORATION = 0.35

    K = 0.55                      # nats per unit of combined evidence

    def __init__(self, false_alarm_rate: float = 0.02, miss_rate: float = 0.10,
                 leak_per_obs: float = 0.97):
        self.alpha = false_alarm_rate
        self.beta = miss_rate
        self.leak = leak_per_obs
        self.upper = math.log((1 - self.beta) / self.alpha)
        self.lower = math.log(self.beta / (1 - self.alpha))
        self._S: Dict[str, float] = {}
        self._n: Dict[str, int] = {}
        self._last: Dict[str, Judgement] = {}

    def observe(self, entity: str, ch: Optional[Channels]) -> Judgement:
        if ch is None:
            j = Judgement(entity, self._S.get(entity, 0.0), self.upper,
                          self.lower, "no_baseline", {}, self._n.get(entity, 0))
            self._last[entity] = j
            return j

        S = self._S.get(entity, 0.0) * self.leak
        combined = (self.W_LEVEL * ch.level
                    + self.W_DIRECTION * ch.direction
                    + self.W_CORROBORATION * ch.corroboration)
        llr = self.K * combined

        # Clamp just outside the boundaries. Without this a long healthy run
        # drives S so far negative that the first real fault takes weeks to
        # climb back, which is how a monitor misses the second failure of a
        # component it once cleared.
        S = max(self.lower - 2.0, min(self.upper + 2.0, S + llr))

        self._S[entity] = S
        self._n[entity] = self._n.get(entity, 0) + 1

        dec = ("name_it" if S >= self.upper else
               "ordinary" if S <= self.lower else "watching")
        j = Judgement(entity, S, self.upper, self.lower, dec,
                      {"level": round(ch.level, 3),
                       "direction": round(ch.direction, 3),
                       "corroboration": round(ch.corroboration, 3),
                       "llr": round(llr, 3), **ch.detail},
                      self._n[entity])
        self._last[entity] = j
        return j

    def state(self, entity: str) -> float:
        return self._S.get(entity, 0.0)

    def last(self, entity: str) -> Optional[Judgement]:
        return self._last.get(entity)

    def reset(self, entity: str) -> None:
        self._S.pop(entity, None)
        self._n.pop(entity, None)
        self._last.pop(entity, None)

    def expected_obs_to_decide(self, mean_llr: float) -> Optional[float]:
        """Wald's approximation, printed beside the measured count.

        A sequential test that decides much faster than its own theory says it
        should is usually a test whose observations are correlated, not a fast
        test. Printing both is how that gets noticed."""
        return (self.upper / mean_llr) if mean_llr > 0 else None


# ── channel construction from a set of directed symptoms ────────────────────

Z_OPERATING = 3.0


def build_channels(z_by_symptom: Dict[str, float],
                   direction_by_symptom: Dict[str, int],
                   primary: Dict[str, bool]) -> Optional[Channels]:
    """Turn per-symptom robust z-scores into the three channels.

    `z_by_symptom` holds only symptoms whose baseline is ready. If no primary
    symptom for this component has a ready baseline, the answer is None, which
    the accumulator turns into `no_baseline` rather than into a guess.
    """
    prim = {k: v for k, v in z_by_symptom.items() if primary.get(k, False)}
    if not prim:
        return None

    # Signed toward the fault: multiply by the declared direction.
    signed = {k: v * direction_by_symptom[k] for k, v in prim.items()}
    lead = max(signed, key=lambda k: abs(signed[k]))
    e = signed[lead]

    level = max(-1.0, min(1.5, (abs(e) - Z_OPERATING) / Z_OPERATING))
    if e < 0:
        # The strongest movement went against the physics. Level is evidence of
        # *something*, so it must not be allowed to count toward this component.
        level = min(level, 0.0)

    direction = 0.8 if e >= 0 else -0.7

    others = [v for k, v in signed.items() if k != lead]
    others += [v * direction_by_symptom[k]
               for k, v in z_by_symptom.items() if not primary.get(k, False)]
    if not others:
        corroboration = 0.0
    elif any(o > Z_OPERATING / 2 for o in others):
        corroboration = 0.6
    elif any(o < -Z_OPERATING / 2 for o in others):
        corroboration = -0.4
    else:
        corroboration = -0.1     # silence is mild evidence against

    return Channels(level, direction, corroboration,
                    {"lead_z": round(e, 2), "lead": 0.0})
