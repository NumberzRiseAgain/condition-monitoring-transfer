#!/usr/bin/env python3
"""Run the same evaluation against any of the public datasets.

Nothing in cbmx changes between them. The asset card gets a different bearing
geometry, the loader normalises the labels, and everything downstream is
identical. That is the claim being tested here: if the method is really
geometry-first, moving to a rig with a different bearing and a different rig
should cost a config line and nothing else.

    python tools/eval_dataset.py --dataset mfpt      --data data/mfpt
    python tools/eval_dataset.py --dataset ottawa    --data data/ottawa
    python tools/eval_dataset.py --dataset paderborn --data data/paderborn

Held-out protocol, same shape as the CWRU run: the band is commissioned on a
small, named subset which is then excluded; the baseline is learned from healthy
records only; everything else is test, and healthy records are where false
alarms are counted.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbmx.graph.asset import Asset, LRU, SensorPoint          # noqa: E402
from cbmx.health.baseline import Baseline                     # noqa: E402
from cbmx.io.base import Record                               # noqa: E402
from cbmx.monitor import Monitor                              # noqa: E402
from cbmx.physics.bearing import get as get_bearing           # noqa: E402
from cbmx.physics.envelope import commission_band             # noqa: E402


_CODE_RE = re.compile(r"_([A-Z]{1,2}\d{2,3})_\d+$", re.I)


def code_of(rec) -> str:
    """The bearing this record came from.

    Leave-one-bearing-out needs to know which physical bearing produced a
    record, and that is not the same question as which fault class it carries.
    Paderborn puts it in the filename; everything else gets a single synthetic
    unit, which makes bearing-holdout a no-op there rather than a wrong answer.
    """
    m = _CODE_RE.search(Path(rec.path).stem)
    if m:
        return m.group(1).upper()
    n = (rec.note or "").split()
    return n[0].upper() if n else ("healthy" if rec.is_healthy else rec.fault)


PARTS = ["outer_race", "inner_race", "rolling_element", "cage"]


def load(dataset: str, path: str, channel: str, ppr: int):
    if dataset == "cwru":
        from cbmx.io.cwru import load_dir
        return load_dir(path)
    if dataset == "mfpt":
        from cbmx.io.mfpt import load_dir
        return load_dir(path)
    if dataset == "ottawa":
        from cbmx.io.ottawa import load_dir
        return load_dir(path, ppr)
    if dataset == "paderborn":
        from cbmx.io.paderborn import load_dir
        return load_dir(path, channel)
    raise SystemExit(f"unknown dataset {dataset}")


def build_asset(dataset: str, bearing_key: str, fs: float) -> Asset:
    """An asset card generated from the record's own bearing, so no dataset
    needs a hand-written YAML."""
    b = get_bearing(bearing_key)
    lrus = {"machine": LRU("machine", f"{dataset} test rig"),
            "brg": LRU("brg", f"{b.designation}", part_number=bearing_key,
                       parent="machine")}
    sensors = {"acc": SensorPoint("acc", "accelerometer", "housing", fs, ["brg"])}
    a = Asset(id=f"{dataset}_rig", name=f"{dataset} rig", lrus=lrus,
              sensors=sensors, bearings={"brg": b}, shaft_ratio={"brg": 1.0})
    a.build_symptoms()
    return a


def windows(rec: Record, window_s: float, hop_s=None):
    n = int(window_s * rec.fs)
    hop = int((hop_s or window_s) * rec.fs)
    if n <= 0 or len(rec.signal) < n:
        return
    for i in range(0, len(rec.signal) - n + 1, hop):
        yield i / rec.fs, rec.signal[i:i + n]


def window_shaft_hz(rec: Record, i0: int, n: int) -> float:
    """Shaft speed for THIS window, not for the whole record.

    On a constant-speed rig the two are the same and this returns the record's
    nominal speed. On a variable-speed rig they are emphatically not: a uOttawa
    record sweeps the shaft across its whole range inside ten seconds, so the
    record's mean speed describes no window in it.

    That distinction decides whether the run says anything at all. The regime is
    a speed bin, and a baseline is only ever compared against its own regime. Bin
    every window of every record on the record's mean and the healthy records
    land in one set of bins, the damaged records land in another, and the system
    correctly answers `no_baseline` to all of them — which is honest, and
    useless. It is also exactly what happened on the first uOttawa run: 33 of 33
    fault records declined, 0 of 12 healthy records alarmed, nothing judged.
    """
    rs = rec.rpm_series
    if rs is None or len(rs) < 2:
        return rec.shaft_hz
    a = int(i0 / max(len(rec.signal), 1) * len(rs))
    b = int((i0 + n) / max(len(rec.signal), 1) * len(rs))
    seg = np.asarray(rs[a:max(b, a + 1)], dtype=float)
    seg = seg[np.isfinite(seg) & (seg > 0)]
    return float(np.mean(seg)) / 60.0 if seg.size else rec.shaft_hz


def slice_rec(rec: Record, lo: float, hi: float) -> Record:
    n = len(rec.signal)
    a, b = int(lo * n), int(hi * n)
    rs = None
    if rec.rpm_series is not None:
        rs = rec.rpm_series[a:b]
    r = copy.copy(rec)
    r.signal = rec.signal[a:b]
    r.rpm_series = rs
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                    choices=["cwru", "mfpt", "ottawa", "paderborn"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--window", type=float, default=0.75)
    ap.add_argument("--alpha", type=float, default=0.02)
    ap.add_argument("--beta", type=float, default=0.10)
    ap.add_argument("--min-samples", type=int, default=30)
    ap.add_argument("--attribution", default="family", choices=["line", "family"])
    ap.add_argument("--channel", default="vibration", help="paderborn only")
    ap.add_argument("--ppr", type=int, default=1, help="ottawa tacho pulses/rev")
    ap.add_argument("--max-fs", type=float, default=48000.0,
                    help="decimate above this; 200 kHz records are 16x more "
                         "compute than the analysis needs")
    ap.add_argument("--limit", type=int, default=0,
                    help="use only the first N records — for a smoke test")
    ap.add_argument("--out", default=None)
    ap.add_argument("--holdout", default="record", choices=["record", "bearing"],
                    help="record: healthy baseline from the first 60%% of each "
                         "healthy record, tested on its own tail. bearing: "
                         "leave-one-bearing-out — the baseline never sees the "
                         "healthy bearing it is scored against, which is the "
                         "false-alarm question a reviewer actually asks.")
    ap.add_argument("--commission", default="fault",
                    choices=["fault", "healthy"],
                    help="how the demodulation band is chosen. 'fault' is the "
                         "original: score candidate bands by how well they "
                         "resolve a fault order on one damaged record per class. "
                         "'healthy' is spectral kurtosis over HEALTHY windows "
                         "only, which is what eval_kaist uses. The fault "
                         "criterion has now produced a silent zero-detection run "
                         "on two independent datasets, so both are selectable "
                         "and the log says which was used.")
    ap.add_argument("--bearing-key", default=None,
                    help="override the catalogue geometry for every record")
    ap.add_argument("--band", nargs=2, type=float, default=None,
                    metavar=("LO", "HI"),
                    help="Force the demodulation band instead of commissioning "
                         "one. The method's own claim is that the band is chosen "
                         "ONCE at installation and frozen; re-deriving it from "
                         "whichever fault record happens to sort first is not "
                         "that, and at fleet scale it does not even give the "
                         "same answer twice.")
    ap.add_argument("--condition", default=None,
                    help="Paderborn: keep only one operating condition, e.g. "
                         "N15_M07_F10. The whole set is 1,120 records at 64 kHz "
                         "and will not fit in a small machine's memory; running "
                         "one condition at a time and reporting all four is "
                         "both tractable and a stronger claim than one blend.")
    ap.add_argument("--real-damage-only", action="store_true",
                    help="Paderborn: keep only accelerated-lifetime damage "
                         "codes, excluding machined defects.")
    a = ap.parse_args()

    print("=" * 78)
    print(f"DATASET {a.dataset.upper()}   attribution={a.attribution}   "
          f"window={a.window}s   alpha={a.alpha}")
    print("=" * 78)

    # Distinguish "you have not downloaded this yet" from "the files are here
    # and unreadable". They are completely different problems and the old
    # message — no usable records, go run inspect_mat.py — sent you to debug a
    # parser when the directory did not exist.
    root = Path(a.data)
    if not root.exists():
        print(f"\n{a.data} does not exist.\n")
        print("Nothing has been downloaded to that path yet. See DOWNLOADS.md")
        print(f"for where to get the {a.dataset} set and how to lay it out.")
        return 1
    n_files = sum(1 for _ in root.rglob("*.mat")) + sum(1 for _ in root.rglob("*.csv"))
    if n_files == 0:
        print(f"\n{a.data} exists but holds no .mat or .csv files.\n")
        print("Extraction probably did not finish, or the files landed one")
        print("directory deeper. Check with:  find", a.data, "-name '*.mat' | head")
        return 1

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        recs = load(a.dataset, a.data, a.channel, a.ppr)
        skipped = [str(x.message)[:110] for x in w]
    if a.limit:
        recs = recs[:a.limit]
    if not recs:
        print(f"\n{n_files} file(s) found in {a.data}, but none could be read as "
              f"{a.dataset} records.")
        print("This is a naming or format problem, not a missing-data problem.")
        for s in skipped[:10]:
            print("   ", s)
        print("\nRun tools/inspect_mat.py on one file and send me the output.")
        return 1
    if skipped:
        print(f"\n{len(skipped)} file(s) skipped:")
        for s in skipped[:6]:
            print("   ", s)

    # Put every record on ONE rate, and only ever by an integer factor.
    #
    # The obvious version of this — decimate everything toward a fixed ceiling —
    # is wrong and silently so. MFPT records its healthy baselines at 97656 Hz
    # and its faults at 48828 Hz; clamping both toward 48000 leaves neither at
    # an integer ratio, so neither gets decimated and the run proceeds with two
    # different sample rates in the same baseline. Every order then lands in the
    # wrong bin for half the records, with no error anywhere.
    #
    # So: the common rate is the LOWEST rate present (which every higher rate
    # must divide into exactly), and any further reduction for speed is a whole
    # extra factor on top of that.
    from cbmx.io.cwru import align_rate
    base = min(r.fs for r in recs)
    bad = [r for r in recs if abs(r.fs / base - round(r.fs / base)) > 1e-6]
    if bad:
        print(f"\n!! {len(bad)} record(s) are not an integer multiple of the "
              f"lowest rate ({base:.0f} Hz). Mixing rates would put every order "
              f"in the wrong bin, so these are dropped:")
        for r in bad[:5]:
            print(f"     {Path(r.path).name}  {r.fs:.0f} Hz")
        recs = [r for r in recs if not any(r is x for x in bad)]
        if not recs:
            return 1
        base = min(r.fs for r in recs)

    extra = 1
    while base / extra > a.max_fs and (base / (extra + 1)) == int(base / (extra + 1)):
        extra += 1
    tgt = base / extra
    for i, r in enumerate(recs):
        if abs(r.fs - tgt) > 1e-6:
            recs[i] = align_rate(r, tgt)
    print(f"\nloaded {len(recs)} records, all decimated to {tgt/1000:.1f} kHz "
          f"(lowest rate present {base/1000:.1f} kHz)")

    var = [r for r in recs if r.variable_speed]
    if var:
        print(f"   {len(var)} records carry a MOVING speed trace — this is the "
              f"set that\n   actually tests order tracking, which is what an "
              f"arrestment needs.")

    if a.condition:
        want = a.condition.upper()
        before = len(recs)
        recs = [r for r in recs if Path(r.path).stem.upper().startswith(want)]
        print(f"   --condition {want}: {before} -> {len(recs)} records")
        if not recs:
            print("   no records match; check the prefix against a filename")
            return 1

    if a.real_damage_only:
        from cbmx.io.paderborn import REAL_DAMAGE
        before = len(recs)
        recs = [r for r in recs if r.is_healthy or code_of(r) in REAL_DAMAGE]
        print(f"   --real-damage-only: {before} -> {len(recs)} records; machined "
              f"defects excluded, accelerated-lifetime damage only")

    healthy = [r for r in recs if r.is_healthy]
    faults = [r for r in recs if not r.is_healthy and r.fault in PARTS]
    other = [r for r in recs if not r.is_healthy and r.fault not in PARTS]
    print(f"   healthy {len(healthy)}   faults {len(faults)}"
          + (f"   excluded (combined/unknown) {len(other)}" if other else ""))
    if not healthy or not faults:
        print("\nNeed at least one healthy and one fault record.")
        return 1

    # Commission on one fault record per class, then exclude them.
    seen, commission = set(), []
    for r in faults:
        if r.fault not in seen:
            seen.add(r.fault)
            commission.append(r)
    # Identity, not equality. `Record` is a dataclass, so `in` calls __eq__,
    # which compares the signal arrays elementwise and raises the moment two
    # records differ in length by a sample — which decimation guarantees at
    # fleet scale and never produced with three bearings on one rate.
    test = [r for r in faults if not any(r is c for c in commission)]
    if not test:
        print("\nToo few fault records to hold any out.")
        return 1

    asset = build_asset(a.dataset, a.bearing_key or recs[0].bearing_key, tgt)
    b = asset.bearings["brg"]
    print(f"\nBEARING  {b.designation}")
    for p, o in b.orders().items():
        print(f"   {p:<18}{o:8.4f} x shaft")
    print(f"   source: {b.source}")

    if a.band:
        from cbmx.physics.envelope import Band as _Band
        band = _Band(float(a.band[0]), float(a.band[1]), 0.0, 0)
        how = "GIVEN on the command line — the installation band"
    elif a.commission == "healthy":
        # Spectral kurtosis over healthy windows, majority vote. No fault label
        # and no fault order enters this, so it cannot select the band that best
        # displays the damage — and unlike the fault criterion it does not swing
        # to a near-Nyquist band when the record set changes.
        from collections import Counter as _C
        from cbmx.physics.envelope import choose_band as _cb, Band as _Band
        votes = _C()
        for r in healthy[:12]:
            for wt, w in windows(r, a.window):
                bb = _cb(w, tgt, levels=(3, 4), min_hz=500.0)
                votes[(round(bb.lo, 1), round(bb.hi, 1))] += 1
                break
        (lo_, hi_), _ = votes.most_common(1)[0]
        band = _Band(lo_, hi_, 0.0, 0)
        how = (f"spectral kurtosis over {sum(votes.values())} HEALTHY windows; "
               f"runners-up " + ", ".join(f"{a_:.0f}-{b_:.0f}Hz x{n}"
                                          for (a_, b_), n in votes.most_common(4)[1:]))
    else:
        band = commission_band([r.signal for r in commission], tgt, b.orders(),
                               float(np.mean([r.shaft_hz for r in commission])))
        how = f"commissioned on {len(commission)} fault record(s), excluded from test"
    asset.sensors["acc"].band_lo_hz, asset.sensors["acc"].band_hi_hz = band.lo, band.hi
    print(f"\nBAND     {band.lo:.0f}-{band.hi:.0f} Hz, frozen  ({how})")

    heal_s = min(0.6 * len(r.signal) / r.fs for r in healthy)
    hop = max(a.window / 8.0, min(a.window, heal_s / (a.min_samples + 12)))
    bl = Baseline(asset.id, min_samples=a.min_samples)
    learner = Monitor(asset, "acc", bl, window_s=a.window, attribution=a.attribution)
    t = 0.0
    for r in healthy:
        head = slice_rec(r, 0.0, 0.6)
        for wt, w in windows(head, a.window, hop):
            learner.step(t + wt, w,
                         window_shaft_hz(head, int(wt * head.fs), len(w)),
                         load=r.load_hp / 10.0, learning=True)
        t += len(head.signal) / head.fs
    cov = bl.coverage()
    print(f"BASELINE {cov['lines_ready']}/{cov['lines_tracked']} lines ready, "
          f"{heal_s:.1f}s healthy per record, ~{heal_s/a.window:.0f} independent windows")
    if cov["lines_ready"] == 0:
        print("\nNo line reached the minimum. Try --window 0.5 or --min-samples 20.")
        return 1

    def run(rec, tail=False):
        r2 = slice_rec(rec, 0.6, 1.0) if tail else rec
        m = Monitor(asset, "acc", copy.deepcopy(bl), a.alpha, a.beta, a.window,
                    attribution=a.attribution)
        first = None
        for wt, w in windows(r2, a.window):
            f = m.step(wt, w, window_shaft_hz(r2, int(wt * r2.fs), len(w)),
                       load=r2.load_hp / 10.0)
            if f and first is None:
                first = (wt, f[0])
        return m, first

    # ---- leave-one-bearing-out -------------------------------------------
    if a.holdout == "bearing":
        by_code = defaultdict(list)
        for r in healthy:
            by_code[code_of(r)].append(r)
        codes = sorted(by_code)
        if len(codes) < 2:
            print(f"\n--holdout bearing needs at least two healthy bearings; "
                  f"found {codes}. Falling back to record holdout.")
        else:
            print(f"\nLEAVE-ONE-BEARING-OUT over {len(codes)} healthy bearings: "
                  f"{', '.join(codes)}")
            print("The baseline never sees the healthy bearing it is scored")
            print("against, so a false alarm here is a false alarm on a machine")
            print("the monitor has genuinely not met before.\n")

            def fit(recs_in):
                bb = Baseline(asset.id, min_samples=a.min_samples)
                lr = Monitor(asset, "acc", bb, window_s=a.window,
                             attribution=a.attribution)
                tt = 0.0
                for rr in recs_in:
                    for wt, w in windows(rr, a.window, hop):
                        lr.step(tt + wt, w,
                                window_shaft_hz(rr, int(wt * rr.fs), len(w)),
                                load=rr.load_hp / 10.0, learning=True)
                    tt += len(rr.signal) / rr.fs
                for st in bb.stats.values():
                    st.frozen = True
                return bb

            folds, fa_records, fa_total = [], 0, 0
            det = defaultdict(list)
            print(f"   {'held-out bearing':<20}{'its records':>12}"
                  f"{'false alarms':>14}{'fault records named':>22}")
            for h in codes:
                fit_recs = [r for c in codes if c != h for r in by_code[c]]
                bb = fit(fit_recs)
                if bb.coverage()["lines_ready"] == 0:
                    print(f"   {h:<20}{'-':>12}{'no baseline ready':>14}")
                    continue
                nfa = 0
                for r in by_code[h]:
                    m = Monitor(asset, "acc", copy.deepcopy(bb), a.alpha, a.beta,
                                a.window, attribution=a.attribution)
                    hit = None
                    for wt, w in windows(r, a.window):
                        f = m.step(wt, w,
                                   window_shaft_hz(r, int(wt * r.fs), len(w)),
                                   load=r.load_hp / 10.0)
                        if f and hit is None:
                            hit = f[0]
                    if hit is not None:
                        nfa += 1
                nok = 0
                for r in test:
                    m = Monitor(asset, "acc", copy.deepcopy(bb), a.alpha, a.beta,
                                a.window, attribution=a.attribution)
                    hit = None
                    for wt, w in windows(r, a.window):
                        f = m.step(wt, w,
                                   window_shaft_hz(r, int(wt * r.fs), len(w)),
                                   load=r.load_hp / 10.0)
                        if f and hit is None:
                            hit = f[0]
                    got = hit.symptom.part if hit else "nothing"
                    det[code_of(r)].append(got == r.fault)
                    nok += int(got == r.fault)
                fa_records += nfa
                fa_total += len(by_code[h])
                folds.append({"held_out": h, "healthy_records": len(by_code[h]),
                              "false_alarms": nfa, "fault_correct": nok,
                              "fault_total": len(test)})
                print(f"   {h:<20}{len(by_code[h]):>12}{nfa:>14}"
                      f"{nok:>13}/{len(test):<8}")

            print(f"\n   FALSE ALARMS {fa_records} of {fa_total} healthy records "
                  f"across {len(folds)} unseen bearings")
            print(f"   {'bearing':<12}{'fault':<16}{'named correctly':>18}")
            for c in sorted(det):
                v = det[c]
                cls = next((r.fault for r in test if code_of(r) == c), "?")
                print(f"   {c:<12}{cls:<16}{sum(v):>10}/{len(v):<7}")
            allv = [x for v in det.values() for x in v]
            print(f"\n   DETECTION {sum(allv)}/{len(allv)} fault records across "
                  f"{len(det)} damaged bearings, every fold")
            print("   Detection and false alarm, together. Neither alone is a result.")

            out = a.out or f"runs/{a.dataset}_bearing_holdout.json"
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text(json.dumps({
                "dataset": a.dataset, "holdout": "bearing",
                "real_damage_only": a.real_damage_only,
                "bearing": b.designation, "band_hz": [band.lo, band.hi],
                "healthy_bearings": codes, "folds": folds,
                "false_alarm_records": fa_records, "healthy_records": fa_total,
                "detection": {c: [bool(x) for x in v] for c, v in det.items()},
            }, indent=2))
            print(f"   written to {out}")
            return 0

    print("\nHEALTHY RECORDS (unseen segments) — the test we most want to fail\n")
    fa = 0
    for r in healthy:
        m, first = run(r, tail=True)
        if first is None:
            print(f"   {r.label:<34} silent   PASS")
        else:
            fa += 1
            print(f"   {r.label:<34} reported {first[1].symptom.part}   FALSE ALARM")

    print("\nFAULT RECORDS\n")
    conf = defaultdict(lambda: defaultdict(int))
    rows = []
    for r in sorted(test, key=lambda x: (x.fault, x.load_hp, x.file_no)):
        m, first = run(r)
        cvg = m.coverage()
        got = (first[1].symptom.part if first
               else ("no baseline" if cvg["fully_blind"] else "nothing"))
        ok = got == r.fault
        conf[r.fault][got] += 1
        lat = f"{first[0]:.1f}s" if first else "—"
        mark = "" if ok else ("   <-- wrong" if first else "")
        note = f"  [{r.note}]" if r.note else ""
        print(f"   {r.label:<30}{got:<16}{lat:>8}{mark}{note}")
        rows.append({"file": r.path, "true": r.fault, "reported": got,
                     "correct": ok, "latency_s": first[0] if first else None,
                     "note": r.note})

    print("\nCONFUSION MATRIX\n")
    cols = PARTS + ["nothing", "no baseline"]
    print("   true \\ reported   " + "".join(f"{c[:11]:>13}" for c in cols))
    for tp in PARTS:
        if conf[tp]:
            print(f"   {tp:<18}" + "".join(f"{conf[tp][c]:>13}" for c in cols))

    n_ok = sum(1 for r in rows if r["correct"])
    n_dec = sum(1 for r in rows if r["reported"] == "no baseline")
    n_wrong = sum(1 for r in rows if not r["correct"] and
                  r["reported"] not in ("nothing", "no baseline"))
    print("\nTOTALS")
    print(f"   correct           {n_ok}/{len(rows)}")
    if n_dec:
        print(f"   judged            {n_ok}/{len(rows)-n_dec}")
    print(f"   WRONG-PART calls  {n_wrong}")
    print(f"   false alarms      {fa}/{len(healthy)}")

    out = a.out or f"runs/{a.dataset}_{a.attribution}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({
        "dataset": a.dataset, "attribution": a.attribution,
        "bearing": b.designation, "band_hz": [band.lo, band.hi],
        "correct": n_ok, "total": len(rows), "wrong_part": n_wrong,
        "declined": n_dec, "false_alarms": fa, "healthy": len(healthy),
        "variable_speed_records": len(var), "rows": rows,
    }, indent=2))
    print(f"   written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
