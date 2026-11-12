"""Runs `harness run` again from the stage it stopped in, when a run ends before every stage passed.

    python -m harness.autorestart --color orange ... (the same options as `harness run`)

Each attempt gets the time still left on the overall clock (--time-limit, default 300 s) and --start-stage set to
the stage the last attempt was in (the next one, if that stage had passed). Stops when every stage has passed, the
time is used up, or after MAX_ATTEMPTS.

Stopping is never a restart, and never skips the arms' home and park:
- Ctrl-C (or SIGTERM / SIGHUP) here is passed to the running attempt once, as an end request; this waits for it to
  home and park the arms, then exits without restarting. The attempt runs in its own session, so a Ctrl-C in the
  terminal reaches it only through here (one end request, not two), and it is never killed.
- An attempt ended from outside (the voice agent's SIGINT to its pid, `kill -INT`, a freeze then an end) writes
  `end_requested: true` in its state.json: that also stops here, with no restart.
"""

from __future__ import annotations

import json
import pathlib
import signal
import subprocess
import sys
import time

STAGES = ("unplug", "stack", "balls")
MAX_ATTEMPTS = 6
PAUSE_S = 5.0
MIN_LEFT_S = 20.0
STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
HARNESS = [sys.executable, "-m", "harness", "run"]  # the command each attempt runs (tests swap it)
ROOT = pathlib.Path("~/hackathon_runs").expanduser()  # where each attempt's run folder goes


class _Stopper:
    """A stop asked of this wrapper: passed once to the running attempt; no restart after it."""

    def __init__(self) -> None:
        self.requested = False
        self.child: subprocess.Popen | None = None

    def handler(self, signum, frame) -> None:
        first = not self.requested
        self.requested = True
        child = self.child
        if child is not None and child.poll() is None:
            if first:
                print(f"\n[autorestart] {signal.Signals(signum).name}: ending the run (the arms home and park); "
                      "no restart", flush=True)
                try:
                    child.send_signal(signal.SIGINT)  # the harness's end request; it homes and parks
                except ProcessLookupError:
                    pass
            else:
                print("[autorestart] already ending; waiting for the arms to home and park", flush=True)
        elif first:
            print(f"\n[autorestart] {signal.Signals(signum).name}: not restarting", flush=True)


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


def _read_json(path: pathlib.Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _run_attempt(cmd: list[str], stopper: _Stopper) -> int:
    """Runs one attempt to its end and returns its exit code. Never kills it: a stop is passed on as an end request
    (see _Stopper), and this keeps waiting while the arms home and park."""
    child = subprocess.Popen(cmd, start_new_session=True)  # its own session: the terminal's Ctrl-C comes via us
    stopper.child = child
    try:
        if stopper.requested:  # a stop landed between attempts' checks and the start: end this one at once
            child.send_signal(signal.SIGINT)
        while True:
            try:
                return child.wait()
            except KeyboardInterrupt:  # only if a handler was not installed; keep waiting, never kill
                stopper.handler(signal.SIGINT, None)
    finally:
        stopper.child = None


def _pause(seconds: float, stopper: _Stopper) -> None:
    end = time.monotonic() + seconds
    while not stopper.requested and time.monotonic() < end:
        time.sleep(0.05)


def main(argv: list[str]) -> int:
    args = list(argv)
    total = float(_pop(args, "--time-limit", "300"))
    stage = _pop(args, "--start-stage", "unplug")
    _pop(args, "--out", None)
    started = time.monotonic()
    scores: dict[str, float] = {}
    stopper = _Stopper()
    previous = {s: signal.signal(s, stopper.handler) for s in STOP_SIGNALS}
    try:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            if stopper.requested:
                return 130
            left = total - (time.monotonic() - started)
            if left < MIN_LEFT_S:
                print(f"[autorestart] {left:.0f} s left: not restarting")
                break
            out = ROOT / (time.strftime("%H%M%S") + f"_a{attempt}")
            cmd = [*HARNESS, *args, "--time-limit", f"{left:.0f}", "--start-stage", stage, "--out", str(out)]
            print(f"\n[autorestart] attempt {attempt}: from {stage} with {left:.0f} s left -> {out}", flush=True)
            code = _run_attempt(cmd, stopper)
            if stopper.requested or code in (130, -2):
                print(f"[autorestart] the run was stopped (exit {code}): not restarting")
                return 130
            state = _read_json(out / "state.json")
            if state.get("end_requested"):
                print(f"[autorestart] attempt {attempt} was ended on request (voice, kill or Ctrl-C; exit {code}): "
                      "not restarting")
                return 130
            report = _read_json(out / "report.json")
            if not report:
                print(f"[autorestart] no report from attempt {attempt}; restarting from {stage}")
            if code != 0 and not (out / "go.jpg").exists():
                # Never reached go: the arms or cameras would not open (a gripper left powered, a camera unplugged).
                # Retrying can't fix that; a person has to.
                print(f"[autorestart] attempt {attempt} failed before go (exit {code}): the arms or cameras did not "
                      "open. Fix that (see the error above), then run this again. Not restarting.")
                return code
            for st, t in (report.get("passed") or {}).items():
                scores.setdefault(st, t)
            nxt = next_stage(report, stage) if report else stage
            if nxt is None or all(st in scores for st in STAGES[STAGES.index(stage):]):
                print(f"[autorestart] every stage passed: {scores}")
                return 0
            print(f"[autorestart] attempt {attempt} ended (exit {code}) in {nxt}; passed so far {sorted(scores)}. "
                  f"Restarting from {nxt} in {PAUSE_S:.0f} s (Ctrl-C to cancel)", flush=True)
            _pause(PAUSE_S, stopper)
            if stopper.requested:
                print("[autorestart] cancelled")
                return 130
            stage = nxt
        print(f"[autorestart] done; passed {sorted(scores)}")
        return 0
    finally:
        for s, h in previous.items():
            signal.signal(s, h)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
