#!/usr/bin/env python3
"""Describe a .mat file without assuming anything about it.

Run this first when a loader fails. Every one of these datasets stores its
signals under a different name, in a different nesting, at a different rate, and
the published descriptions are not always current. Rather than guessing twice,
print what is actually in the file and fix the loader against that.

    python tools/inspect_mat.py data/ottawa/H-A-1.mat
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def describe(v, depth=0, name=""):
    pad = "   " * (depth + 1)
    if hasattr(v, "_fieldnames"):
        print(f"{pad}{name}: struct with fields {list(v._fieldnames)}")
        for f in v._fieldnames:
            if depth < 2:
                describe(getattr(v, f), depth + 1, f)
        return
    a = np.asarray(v)
    if a.dtype == object:
        print(f"{pad}{name}: object array shape {a.shape}")
        for x in np.atleast_1d(a).ravel()[:3]:
            if depth < 2:
                describe(x, depth + 1, "item")
        return
    if a.size > 8:
        print(f"{pad}{name}: {a.dtype} shape {a.shape}  "
              f"min {np.nanmin(a):.4g} max {np.nanmax(a):.4g} "
              f"mean {np.nanmean(a):.4g}")
    else:
        print(f"{pad}{name}: {a.dtype} shape {a.shape}  value {a.ravel()[:8]}")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    from scipy.io import loadmat
    for p in sys.argv[1:]:
        path = Path(p)
        print("=" * 78)
        print(path)
        print("=" * 78)
        m = loadmat(str(path), squeeze_me=True, struct_as_record=False)
        for k, v in m.items():
            if k.startswith("__"):
                continue
            describe(v, 0, k)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
