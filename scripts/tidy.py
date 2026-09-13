#!/usr/bin/env python3
"""Remove unused imports and pointless f-string prefixes, in place.

    python3 scripts/tidy.py --check     # report only, exit 1 if anything found
    python3 scripts/tidy.py             # rewrite the files

Both classes of finding are mechanical and safe to fix automatically, which is
why they are fixed automatically rather than argued about in review:

* An unused import is dead weight that also lies about a module's dependencies.
  The name is removed from its import statement; if that empties the statement,
  the whole line goes.

* An `f` prefix on a string with no placeholder does nothing at runtime and
  invites the next reader to hunt for the interpolation that is not there.

Deliberate column alignment inside dict and table literals is NOT touched.
pycodestyle calls that E241/E272; it is intentional here and is silenced in
setup.cfg with a note saying so.
"""
from __future__ import annotations

import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["src/cbmx", "tools", "tests"]

UNUSED = re.compile(r"^(?P<file>[^:]+):(?P<line>\d+):\d+: '(?P<name>.+?)' imported but unused$")
FSTRING = re.compile(
    r"^(?P<file>[^:]+):(?P<line>\d+):(?P<col>\d+): f-string is missing placeholders$")


def pyflakes() -> list[str]:
    out = subprocess.run(
        [sys.executable, "-m", "pyflakes", *TARGETS],
        cwd=ROOT, capture_output=True, text=True,
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin:/usr/local/bin"},
    )
    return [ln for ln in out.stdout.splitlines() if ln.strip()]


def drop_name(line: str, dotted: str) -> str | None:
    """The import line with `dotted` removed, or None if the line should go.

    pyflakes reports the dotted path ('typing.Optional'); the source binds only
    the last component.
    """
    name = dotted.rsplit(".", 1)[-1]
    if re.match(rf"^\s*import\s+{re.escape(dotted)}\s*$", line):
        return None
    if re.match(rf"^\s*import\s+\S+\s+as\s+{re.escape(name)}\s*$", line):
        return None
    m = re.match(r"^(?P<head>\s*from\s+\S+\s+import\s+)(?P<names>.+?)(?P<tail>\s*(#.*)?)$", line)
    if not m:
        return line
    kept = [n.strip() for n in m.group("names").split(",")
            if n.strip() and n.strip().split()[-1] != name]
    if not kept:
        return None
    return f"{m.group('head')}{', '.join(kept)}{m.group('tail')}"


def main(argv: list[str]) -> int:
    check = "--check" in argv
    findings = pyflakes()
    if not findings:
        print("pyflakes: clean")
        return 0

    by_file: dict[Path, list[tuple[int, str]]] = defaultdict(list)
    other: list[str] = []
    for f in findings:
        m = UNUSED.match(f) or FSTRING.match(f)
        if m is None:
            other.append(f)
            continue
        # pyflakes has no notion of `# noqa`; flake8 does, and this project's
        # lint gate is flake8. Honour the suppression here too, or `--check`
        # reports findings the gate has already accepted — a probe import in
        # preflight.py is deliberate and marked as such.
        path = ROOT / m["file"]
        try:
            line = path.read_text(encoding="utf-8").split("\n")[int(m["line"]) - 1]
        except (OSError, IndexError):
            line = ""
        if "# noqa" in line:
            continue
        kind = m["name"] if "name" in m.groupdict() and m.groupdict().get("name") else "@fstring"
        by_file[path].append((int(m["line"]), kind))

    if not by_file and not other:
        print("pyflakes: clean (after honouring # noqa)")
        return 0

    n_imp = sum(1 for v in by_file.values() for _, n in v if n != "@fstring")
    n_fs = sum(1 for v in by_file.values() for _, n in v if n == "@fstring")
    print(f"unused imports: {n_imp}   pointless f-prefixes: {n_fs}   other: {len(other)}")
    for o in other:
        print(f"   NOT AUTO-FIXED  {o}")
    if check:
        return 1 if (by_file or other) else 0

    for path, items in by_file.items():
        lines = path.read_text(encoding="utf-8").split("\n")
        # Work bottom-up so earlier line numbers stay valid.
        for lineno, name in sorted(items, key=lambda t: -t[0]):
            i = lineno - 1
            if name == "@fstring":
                lines[i] = re.sub(r'(?<![A-Za-z0-9_])f(["\'])', r"\1", lines[i])
            else:
                new = drop_name(lines[i], name)
                if new is None:
                    del lines[i]
                else:
                    lines[i] = new
        path.write_text("\n".join(lines), encoding="utf-8")
    print(f"rewrote {len(by_file)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
