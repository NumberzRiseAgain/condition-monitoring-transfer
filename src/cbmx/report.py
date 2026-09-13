"""The thing that actually gets sent to shore.

Two numbers in the DV087 topic are thresholds rather than prose, and both are
about this object rather than about the detector:

    "<1 second anomaly-detection latency"
    "<10 MB/hour data-transfer footprints to shore when intermittent links
     reconnect"

A monitor that decides quickly and then emits a megabyte of diagnostics has not
met the second one. So the report is designed against the bandwidth budget from
the start, and the budget is checked by measurement rather than asserted.

What a report has to carry, and why each field is not optional:

    the named part          a maintainer orders a part, not a subsystem. The
                            asset path resolves to a Lowest Replaceable Unit.
    predicted vs measured   the frequency the geometry said, and the frequency
                            actually found. A Navy engineer can check this on
                            paper against the bearing drawing. It is the single
                            field that makes the call falsifiable.
    the baseline it broke   how far above normal, in robust sigmas, and WHICH
                            normal — which operating regime, learned over how
                            many observations. A deviation without its regime
                            is not a measurement.
    persistence             how many observations of evidence, and the Wald
                            boundary they crossed.
    the false-alarm rate    the alpha that produced this confidence. A
                            confidence with no error rate attached is
                            decoration.
    what was NOT concluded  the competing parts that were considered and
                            rejected, with their scores. This is what separates
                            a diagnosis from an assertion, and it costs almost
                            nothing to send.
    the chain hash          each report hashes the previous one, so a
                            maintenance record cannot be quietly altered after
                            the fact and a gap in the sequence is visible.

Deliberately NOT carried: raw waveform, spectra, or feature vectors. Those are
what the bandwidth budget exists to avoid sending. If a shore analyst needs the
underlying capture, the report names it and it can be pulled on request — a
decision that is itself part of meeting the 10 MB/hour figure.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class Alternative:
    """A part that was considered and not named, and how far short it fell."""

    part: str
    z: float
    reason: str


@dataclass
class Report:
    # what
    asset: str                    # e.g. "AAG/HPU/pump_motor"
    lru: str                      # the replaceable unit
    part: str                     # outer_race | inner_race | cage | ...
    stock_number: Optional[str] = None

    # why, in a form that can be checked on paper
    order_predicted: float = 0.0  # multiples of shaft speed, from geometry
    freq_predicted_hz: float = 0.0
    freq_measured_hz: float = 0.0
    shaft_hz: float = 0.0

    # against what normal
    z: float = 0.0
    regime: str = ""
    baseline_n: int = 0
    baseline_frozen: bool = False

    # how sure, and at what stated error rate
    decision: str = "watching"
    evidence_nats: float = 0.0
    boundary_nats: float = 0.0
    observations: int = 0
    false_alarm_rate: float = 0.02
    miss_rate: float = 0.10

    # what was rejected
    alternatives: List[Alternative] = field(default_factory=list)

    # provenance
    sensor: str = ""
    band_hz: Optional[List[float]] = None
    capture_id: str = ""
    method: str = ""
    prev_hash: str = ""

    def sentence(self) -> str:
        """The one line a maintainer reads. Everything else is the audit trail."""
        agree = (abs(self.freq_measured_hz - self.freq_predicted_hz)
                 / max(self.freq_predicted_hz, 1e-9) * 100.0)
        return (
            f"{self.lru}, {self.part.replace('_', ' ')}: energy at "
            f"{self.freq_measured_hz:.1f} Hz, where the geometry predicts "
            f"{self.freq_predicted_hz:.1f} Hz for this part at "
            f"{self.shaft_hz*60:.0f} rpm ({agree:.1f}% off). "
            f"{self.z:.1f} sigma above its own baseline for regime "
            f"{self.regime}, learned over {self.baseline_n} observations. "
            f"Sustained {self.observations} observations to cross the "
            f"{self.boundary_nats:.2f}-nat threshold set by a "
            f"{self.false_alarm_rate:.0%} false-alarm rate."
        )

    def to_record(self) -> Dict:
        d = asdict(self)
        d["sentence"] = self.sentence()
        d["hash"] = self.digest()
        return d

    def digest(self) -> str:
        """Chain hash. Truncated to 16 hex characters — collision risk is
        irrelevant for a tamper-evidence chain of this length, and the 48 bytes
        saved per report are 48 bytes not spent on the link."""
        body = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256((self.prev_hash + body).encode()).hexdigest()[:16]

    def serialise(self, compact: bool = True) -> bytes:
        """What actually crosses the link.

        Compact form drops the human sentence — it is reconstructible ashore
        from the fields, so sending it is paying bandwidth twice for the same
        information. Both sizes are reported by the benchmark so the trade is
        visible rather than assumed.
        """
        d = asdict(self)
        d["hash"] = self.digest()
        if not compact:
            d["sentence"] = self.sentence()
        return json.dumps(d, separators=(",", ":")).encode()
