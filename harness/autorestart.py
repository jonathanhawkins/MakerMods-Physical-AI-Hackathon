"""Runs `harness run` again from the stage it stopped in, when a run ends before every stage passed.

    python -m harness.autorestart --color orange ... (the same options as `harness run`)

Each attempt gets the time still left on the overall clock (--time-limit, default 300 s) and --start-stage set to
the stage the last attempt was in (the next one, if that stage had passed). Stops when every stage has passed, the
time is used up, after MAX_ATTEMPTS, or on Ctrl-C (press it during the pause before a restart to cancel).
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time

STAGES = ("unplug", "stack", "balls")
MAX_ATTEMPTS = 6
PAUSE_S = 5.0
MIN_LEFT_S = 20.0


def _pop(args: list[str], name: str, default: str | None) -> str | None:
    """Removes `name value` (or `name=value`) from args, returning the value."""
    out = default
    i = 0
    while i < len(args):
        if args[i] == name and i + 1 < len(args):
            out = args[i + 1]
            del args[i:i + 2]
        elif args[i].startswith(name + "="):
            out = args[i].split("=", 1)[1]
            del args[i]
        else:
            i += 1
    return out


def next_stage(report: dict, current: str) -> str | None:
    """The stage to restart in: the one the run was in, or the next if it passed. None: every stage passed."""
    passed = set(report.get("passed") or {})
    events = report.get("events") or []
    stage = next((e.get("stage") for e in reversed(events) if e.get("stage") in STAGES), current)
    for st in STAGES[STAGES.index(stage):]:
        if st not in passed:
            return st
    return None


def main(argv: list[str]) -> int:
    args = list(argv)
    total = float(_pop(args, "--time-limit", "300"))
    stage = _pop(args, "--start-stage", "unplug")
    _pop(args, "--out", None)
    started = time.monotonic()
    root = pathlib.Path("~/hackathon_runs").expanduser()
    scores: dict[str, float] = {}
    for attempt in range(1, MAX_ATTEMPTS + 1):
        left = total - (time.monotonic() - started)
        if left < MIN_LEFT_S:
            print(f"[autorestart] {left:.0f} s left: not restarting")
            break
        out = root / (time.strftime("%H%M%S") + f"_a{attempt}")
        cmd = [sys.executable, "-m", "harness", "run", *args, "--time-limit", f"{left:.0f}",
               "--start-stage", stage, "--out", str(out)]
        print(f"\n[autorestart] attempt {attempt}: from {stage} with {left:.0f} s left -> {out}", flush=True)
        try:
            code = subprocess.call(cmd)
        except KeyboardInterrupt:
            print("[autorestart] stopped by Ctrl-C")
            return 130
        if code in (130, -2):
            print("[autorestart] the run was stopped by Ctrl-C: not restarting")
            return 130
        try:
            report = json.loads((out / "report.json").read_text())
        except (OSError, ValueError) as exc:
            print(f"[autorestart] no report from attempt {attempt} ({exc}); restarting from {stage}")
            report = {}
        if code != 0 and not (out / "go.jpg").exists():
            # Never reached go: the arms or cameras would not open (a gripper left powered, a camera unplugged).
            # Retrying can't fix that; a person has to.
            print(f"[autorestart] attempt {attempt} failed before go (exit {code}): the arms or cameras did not open. "
                  "Fix that (see the error above), then run this again. Not restarting.")
            return code
        for st, t in (report.get("passed") or {}).items():
            scores.setdefault(st, t)
        nxt = next_stage(report, stage) if report else stage
        if nxt is None or all(st in scores for st in STAGES[STAGES.index(stage):]):
            print(f"[autorestart] every stage passed: {scores}")
            return 0
        print(f"[autorestart] attempt {attempt} ended (exit {code}) in {nxt}; passed so far {sorted(scores)}. "
              f"Restarting from {nxt} in {PAUSE_S:.0f} s (Ctrl-C to cancel)", flush=True)
        try:
            time.sleep(PAUSE_S)
        except KeyboardInterrupt:
            print("[autorestart] cancelled")
            return 130
        stage = nxt
    print(f"[autorestart] done; passed {sorted(scores)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
