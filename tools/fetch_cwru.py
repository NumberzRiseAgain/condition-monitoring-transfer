#!/usr/bin/env python3
"""Download the Case Western seeded-fault records.

Run this on a machine with network access. It fetches only the files the
evaluation needs — 40 records, about 60 MB — from the official Case Western
Bearing Data Center, and records a SHA-256 for each so a reviewer can confirm
they were given the same bytes we measured.

    python tools/fetch_cwru.py --out data/cwru

The data is published by Case Western Reserve University for research use.
Cite them, and check their conditions of use before anything derived from these
files goes into a submission:

    https://engineering.case.edu/bearingdatacenter

If a URL 404s, the site has been reorganised — the file numbers in
src/cbmx/io/cwru.py are stable, the paths are not. Download by hand from the
page above into the same directory and re-run; nothing else depends on how the
files arrived.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbmx.io.cwru import catalogue        # noqa: E402

# The Bearing Data Center has moved hosts more than once. Try each in turn.
BASES = [
    "https://engineering.case.edu/sites/default/files/{n}.mat",
    "https://csegroups.case.edu/sites/default/files/bearingdatacenter/files/Datafiles/{n}.mat",
]


def fetch_one(n: int, out: Path, timeout: float = 60.0) -> tuple:
    dest = out / f"{n}.mat"
    if dest.exists() and dest.stat().st_size > 1000:
        return dest, "cached", dest.stat().st_size
    last = None
    for base in BASES:
        url = base.format(n=n)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "cbmx/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                blob = r.read()
            if len(blob) < 1000:
                last = f"suspiciously small ({len(blob)} bytes)"
                continue
            dest.write_bytes(blob)
            return dest, "downloaded", len(blob)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            last = str(e)
    return None, f"failed: {last}", 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/cwru")
    ap.add_argument("--only-normal", action="store_true",
                    help="just the four healthy baseline records")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = catalogue()
    if args.only_normal:
        rows = [r for r in rows if r["fault"] == "normal"]

    manifest, ok, failed = {}, 0, []
    for r in rows:
        n = r["file_no"]
        path, status, size = fetch_one(n, out)
        if path is None:
            failed.append((n, status))
            print(f"  {n:>4}  {r['fault']:<16} {status}")
            continue
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest[str(n)] = {**r, "sha256": h, "bytes": size, "status": status}
        ok += 1
        print(f"  {n:>4}  {r['fault']:<16} {r['defect_in']:.3f}in "
              f"{r['load_hp']}hp  {size/1e6:5.2f} MB  {status}  {h[:12]}")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{ok}/{len(rows)} records in {out}")
    print(f"manifest with checksums: {out/'manifest.json'}")
    if failed:
        print(f"\n{len(failed)} failed. Download these by hand from")
        print("  https://engineering.case.edu/bearingdatacenter")
        print("  into", out, "— file numbers:", ", ".join(str(n) for n, _ in failed))
        return 1
    print("\nnext:  python tools/eval_cwru.py --data", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
