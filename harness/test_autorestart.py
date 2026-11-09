"""harness/autorestart.py against a fake `harness run` child (no arms, no Dream): when it restarts, and that a stop is
passed on once and never kills the run before its arms have homed and parked."""

from __future__ import annotations

import json
import os
import pathlib
import signal
import subprocess
import sys
import textwrap
import time

from harness import autorestart

REPO = pathlib.Path(__file__).resolve().parent.parent

# A stand-in for `harness run`. MODE (from the FAKE_MODE env var):
#   end_requested - ends at once, its state.json saying an end was asked for (a voice "end the run").
#   incomplete    - ends by itself with unplug passed, in the stack stage.
#   trap          - runs until SIGINT, then "homes and parks" for a moment before it exits 0, like the harness.
FAKE = textwrap.dedent('''
    import json, os, pathlib, signal, sys, time
    args = sys.argv[1:]
    out = pathlib.Path(args[args.index("--out") + 1])
    out.mkdir(parents=True)
    (out / "go.jpg").write_bytes(b"")
    log = pathlib.Path(os.environ["FAKE_LOG"])
    with log.open("a") as f:
        f.write(json.dumps({"pid": os.getpid(), "start_stage": args[args.index("--start-stage") + 1]}) + "\\n")
    mode = os.environ["FAKE_MODE"]
    ended = False
    if mode == "end_requested":
        ended = True
    elif mode == "trap":
        sigints = []
        signal.signal(signal.SIGINT, lambda s, f: sigints.append(s))
        (out / "started").write_text("")
        while not sigints:
            time.sleep(0.01)
        time.sleep(0.5)  # homing and parking
        (out / "parked").write_text(str(len(sigints)))
        ended = True
    passed = {"unplug": 3.0} if mode == "incomplete" else {}
    (out / "report.json").write_text(json.dumps({"passed": passed, "events": [{"stage": "stack"}]}))
    (out / "state.json").write_text(json.dumps({"state": "done", "end_requested": ended}))
''')


def _setup(tmp_path, monkeypatch, mode):
    fake = tmp_path / "fake_harness.py"
    fake.write_text(FAKE)
    log = tmp_path / "attempts.jsonl"
    monkeypatch.setenv("FAKE_MODE", mode)
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setattr(autorestart, "HARNESS", [sys.executable, str(fake)])
    monkeypatch.setattr(autorestart, "ROOT", tmp_path / "runs")
    monkeypatch.setattr(autorestart, "PAUSE_S", 0.0)
    return fake, log


def _attempts(log: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def test_a_run_ended_on_request_is_never_restarted(tmp_path, monkeypatch):
    _, log = _setup(tmp_path, monkeypatch, "end_requested")
    assert autorestart.main(["--color", "orange"]) == 130
    assert len(_attempts(log)) == 1


def test_a_run_that_stops_short_restarts_from_its_stage(tmp_path, monkeypatch):
    _, log = _setup(tmp_path, monkeypatch, "incomplete")
    monkeypatch.setattr(autorestart, "MAX_ATTEMPTS", 2)
    autorestart.main(["--color", "orange"])
    assert [a["start_stage"] for a in _attempts(log)] == ["unplug", "stack"]


def test_ctrl_c_ends_the_run_once_waits_for_the_park_and_does_not_restart(tmp_path, monkeypatch):
    fake, log = _setup(tmp_path, monkeypatch, "trap")
    code = (f"import pathlib, sys\nfrom harness import autorestart as a\n"
            f"a.HARNESS = [sys.executable, {str(fake)!r}]\na.ROOT = pathlib.Path({str(tmp_path / 'runs')!r})\n"
            f"a.PAUSE_S = 0.0\nsys.exit(a.main(['--color', 'orange']))\n")
    parent = subprocess.Popen([sys.executable, "-c", code], cwd=REPO, env=os.environ.copy())
    try:
        until = time.monotonic() + 20
        while not list((tmp_path / "runs").glob("*/started")):
            assert time.monotonic() < until and parent.poll() is None, "the attempt never started"
            time.sleep(0.02)
        parent.send_signal(signal.SIGINT)  # Ctrl-C in the wrapper's terminal
        time.sleep(0.1)
        parent.send_signal(signal.SIGINT)  # pressed again while the arms are homing
        assert parent.wait(timeout=20) == 130
    finally:
        if parent.poll() is None:
            parent.kill()
    parked = list((tmp_path / "runs").glob("*/parked"))
    assert len(parked) == 1, "the run was killed before its arms parked"
    assert parked[0].read_text() == "1"  # one end request reached it, however often Ctrl-C was pressed
    assert len(_attempts(log)) == 1  # and no restart


def test_a_stop_during_the_pause_cancels_the_restart(tmp_path, monkeypatch):
    _, log = _setup(tmp_path, monkeypatch, "incomplete")
    monkeypatch.setattr(autorestart, "PAUSE_S", 5.0)
    real_pause = autorestart._pause

    def pause(seconds, stopper):
        stopper.handler(signal.SIGINT, None)  # Ctrl-C during "Restarting from stack in 5 s"
        real_pause(seconds, stopper)

    monkeypatch.setattr(autorestart, "_pause", pause)
    t = time.monotonic()
    assert autorestart.main(["--color", "orange"]) == 130
    assert time.monotonic() - t < 4.0 and len(_attempts(log)) == 1
