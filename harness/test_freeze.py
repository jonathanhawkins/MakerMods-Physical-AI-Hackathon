"""Freeze control, state.json and the early SIGUSR1 handler (stdlib only; no robot needed)."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness.freeze import FreezeControl, FreezeRequested, StateFile, first_alarm, hold_measured

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def ctl():
    old = signal.getsignal(signal.SIGUSR1)
    c = FreezeControl()
    c.install_freeze_signal()
    yield c
    signal.signal(signal.SIGUSR1, old)


def test_sigusr1_sets_freeze_and_clears_end(ctl):
    calls = []
    ctl.on_freeze = lambda: calls.append(1)
    ctl.end_requested.set()
    os.kill(os.getpid(), signal.SIGUSR1)
    assert ctl.frozen
    assert not ctl.end_requested.is_set()
    assert calls == [1]


def test_interruptible_sleep_raises_after_freeze(ctl):
    threading.Timer(0.05, ctl.freeze.set).start()
    t0 = time.monotonic()
    with pytest.raises(FreezeRequested):
        ctl.interruptible_sleep(5)
    assert time.monotonic() - t0 < 0.5


def test_begin_end_allows_refreeze(ctl):
    os.kill(os.getpid(), signal.SIGUSR1)
    assert ctl.begin_end() is False
    ctl.interruptible_sleep(0.01)
    os.kill(os.getpid(), signal.SIGUSR1)
    with pytest.raises(FreezeRequested):
        ctl.interruptible_sleep(0.01)


def test_begin_end_reports_freeze_after_end_request(ctl):
    ctl.request_end()
    os.kill(os.getpid(), signal.SIGUSR1)
    assert ctl.begin_end() is False
    ctl.request_end()
    assert ctl.begin_end() is True


def test_on_freeze_exception_does_not_escape(ctl):
    def boom():
        raise RuntimeError("loop closed")
    ctl.on_freeze = boom
    os.kill(os.getpid(), signal.SIGUSR1)
    assert ctl.frozen


@pytest.mark.parametrize("which", ["freeze", "end_requested"])
def test_handler_reentrant_while_event_lock_held(ctl, which):
    done = threading.Event()

    def run():
        with getattr(ctl, which)._cond:
            os.kill(os.getpid(), signal.SIGUSR1)
        done.set()

    # the handler runs on the main thread, so the deadlock would hang it: drive from main with a watchdog
    dog = threading.Timer(3, lambda: os._exit(70))
    dog.start()
    try:
        run()
    finally:
        dog.cancel()
    assert done.is_set() and ctl.frozen


def test_state_file_cleans_tmp_on_failure(tmp_path):
    sf = StateFile(tmp_path, link=tmp_path / "link")
    with pytest.raises(TypeError):
        sf.update(bad=object())
    assert not list(tmp_path.glob("*.tmp"))


def test_state_file_atomic_and_schema(tmp_path):
    sf = StateFile(tmp_path, link=tmp_path / "link")
    sf.update(state="starting", pid=os.getpid(), started_at=1.0, dry_run=True, freeze_supported=True,
              stage=None, score=None, alarm=None, error=None, version=1)
    data = json.loads((tmp_path / "state.json").read_text())
    assert data["version"] == 1
    for key in ("state", "pid", "started_at", "stage", "dry_run", "freeze_supported", "score", "alarm",
                "error", "updated_at"):
        assert key in data
    assert not list(tmp_path.glob("*.tmp"))


def test_link_current_points_at_run_dir(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    link = tmp_path / "current"
    sf = StateFile(out, link=link)
    sf.link_current()
    sf.link_current()
    assert link.resolve() == out.resolve()


def test_hold_measured_writes_measured_once():
    written = []
    arm = SimpleNamespace(read=lambda: SimpleNamespace(positions_rad=[1, 2], gripper=0.3), write=written.append)
    assert hold_measured({"left": arm}, lambda pos, grip: ("cmd", list(pos), grip)) == {}
    assert written == [("cmd", [1, 2], 0.3)]


def test_hold_measured_clamps_gripper():
    written = []
    def mk(g):
        return SimpleNamespace(read=lambda: SimpleNamespace(positions_rad=[0], gripper=g), write=written.append)
    hold_measured({"a": mk(1.02), "b": mk(-0.003)}, lambda pos, grip: grip)
    assert written == [1.0, 0.0]


def test_hold_measured_isolates_arms():
    written = []
    def bad_read():
        raise OSError("bus")
    bad = SimpleNamespace(read=bad_read, write=written.append)
    good = SimpleNamespace(read=lambda: SimpleNamespace(positions_rad=[0], gripper=0.5), write=written.append)
    failed = hold_measured({"left": bad, "right": good}, lambda pos, grip: grip)
    assert written == [0.5]
    assert list(failed) == ["left"]


def test_first_alarm():
    ok = SimpleNamespace(alarm=lambda: None, is_alive=lambda: True)
    hot = SimpleNamespace(alarm=lambda: "over temp", is_alive=lambda: True)
    dead = SimpleNamespace(alarm=lambda: None, is_alive=lambda: False)
    assert first_alarm({"left": hot, "right": ok}) == "left: over temp"
    assert first_alarm({"left": ok, "right": dead}) == "right: not answering"
    named = SimpleNamespace(alarm=lambda: "arm 'left': over temp", is_alive=lambda: True)
    assert first_alarm({"left": named}) == "arm 'left': over temp"
    assert first_alarm({"left": ok}) is None


def test_first_alarm_arm_without_alarm_is_no_alarm_and_quiet(caplog):
    class NoAlarm:
        def is_alive(self):
            return True

        def alarm(self):
            raise AttributeError("no alarm in a dry run")

    hot = SimpleNamespace(alarm=lambda: "over temp", is_alive=lambda: True)
    with caplog.at_level("DEBUG"):
        assert first_alarm({"left": NoAlarm()}) is None
        assert first_alarm({"left": NoAlarm(), "right": hot}) == "right: over temp"
    assert not caplog.records
    boom = SimpleNamespace(alarm=lambda: (_ for _ in ()).throw(ValueError("x")), is_alive=lambda: True)
    with pytest.raises(ValueError):
        first_alarm({"left": boom})


def test_main_installs_freeze_before_heavy_imports(tmp_path):
    code = (
        "import json, signal, sys\n"
        "seen = {}\n"
        "class Spy:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'cv2':\n"
        "            seen['h'] = signal.getsignal(signal.SIGUSR1)\n"
        "            raise ImportError('cv2 blocked')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Spy())\n"
        "import harness.__main__ as m\n"
        f"out = {str(tmp_path / 'run')!r}\n"
        "try:\n"
        "    m.main(['run', '--color', 'x', '--out', out])\n"
        "except ImportError:\n"
        "    pass\n"
        "else:\n"
        "    raise SystemExit('no ImportError')\n"
        "h = seen['h']  # what was installed while the heavy imports ran\n"
        "assert getattr(h, '__func__', None) is not None and h.__func__.__name__ == '_on_sigusr1', h\n"
        "assert signal.getsignal(signal.SIGUSR1) == signal.SIG_DFL  # the original is back at exit\n"
        "st = json.load(open(out + '/state.json'))\n"
        "assert st['freeze_supported'] is True\n"
        "assert st['state'] == 'error' and 'cv2' in st['error'], st\n"
    )
    env = {**os.environ, "HOME": str(tmp_path)}
    res = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=60, env=env)
    assert res.returncode == 0, res.stderr


def test_main_help_text():
    code = (
        "import sys; sys.modules['cv2'] = None\n"
        "import harness.__main__ as m\n"
        "try:\n"
        "    m.main(['run', '--help'])\n"
        "except SystemExit as e:\n"
        "    assert e.code == 0\n"
        "assert sys.modules.get('cv2') is None\n"
    )
    res = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    helptext = " ".join(res.stdout.split())
    assert "SIGUSR1 freezes the arms" in helptext
    assert "omit to bin every ball except" in helptext


def test_state_update_survives_unserialisable_value(tmp_path):
    state = StateFile(tmp_path / "run", link=tmp_path / "link")
    state.update(state="running")
    with pytest.raises(TypeError):
        state.update(score=object())
    state.update(state="error", error="boom")  # a failed write must not poison later ones
    on_disk = json.loads((tmp_path / "run" / "state.json").read_text())
    assert on_disk["state"] == "error" and "score" not in on_disk


@pytest.mark.parametrize("sig", ["SIGTERM", "SIGINT", "SIGHUP"])
def test_end_signal_during_heavy_imports_is_a_clean_done(tmp_path, sig):
    code = (
        "import json, os, signal, sys, types\n"
        "class Empty:\n"
        "    def create_module(self, spec): return types.ModuleType('cv2')\n"
        "    def exec_module(self, module): pass\n"
        "class Blocker:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        f"        if name == 'cv2':\n"
        f"            os.kill(os.getpid(), signal.{sig})  # an end signal while the heavy imports run\n"
        "            import importlib.util\n"
        "            return importlib.util.spec_from_loader('cv2', Empty())\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocker())\n"
        "import harness.__main__ as m\n"
        "stub = types.ModuleType('harness.dream_executor')\n"
        "async def run_match(match, ckpt, dry_run, out_dir, ctl, state, **kw):\n"
        "    assert ctl.end_requested.is_set()\n"
        "    state.update(state='done', error='ended before the arms opened')\n"
        "    raise SystemExit(0)\n"
        "stub.run_match = run_match\n"
        "sys.modules['harness.dream_executor'] = stub\n"
        f"out = {str(tmp_path / 'run')!r}\n"
        "try:\n"
        "    m.main(['run', '--out', out])\n"
        "except SystemExit:\n"
        "    pass\n"
        "st = json.load(open(out + '/state.json'))\n"
        "assert st['state'] == 'done', st\n"
    )
    env = {**os.environ, "HOME": str(tmp_path)}
    res = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=60, env=env)
    assert res.returncode == 0, (res.returncode, res.stderr)


# ---------------------------------------------------------------------------------------------------------------
# Round 3: gripper hold, alarm names, run dir, link restore, early handlers, exit codes, state fields.

def test_hold_measured_keeps_the_last_commanded_gripper():
    written = []
    arm = SimpleNamespace(read=lambda: SimpleNamespace(positions_rad=[1], gripper=0.7), write=written.append,
                          _last_command=SimpleNamespace(gripper=0.2))  # squeezing an object: measured is wider
    assert hold_measured({"left": arm}, lambda pos, grip: grip) == {}
    assert written == [0.2]


def test_hold_measured_clamps_the_last_command_and_falls_back_without_one():
    written = []
    mk = lambda **kw: SimpleNamespace(read=lambda: SimpleNamespace(positions_rad=[0], gripper=0.4),  # noqa: E731
                                      write=written.append, **kw)
    hold_measured({"a": mk(_last_command=SimpleNamespace(gripper=1.2)), "b": mk(_last_command=None), "c": mk()},
                  lambda pos, grip: grip)
    assert written == [1.0, 0.4, 0.4]


def test_hold_measured_write_failure_falls_back_to_hold_and_records_both():
    holds = []

    def bad_write(cmd):
        raise OSError("bus write")
    arm = SimpleNamespace(read=lambda: SimpleNamespace(positions_rad=[0], gripper=0.5), write=bad_write,
                          hold=lambda: holds.append(1))
    failed = hold_measured({"left": arm}, lambda pos, grip: grip)
    assert holds == [1]  # tried once
    assert "bus write" in str(failed["left"]) and "hold()" in str(failed["left"])

    def bad_hold():
        raise OSError("hold bus")
    arm2 = SimpleNamespace(read=lambda: SimpleNamespace(positions_rad=[0], gripper=0.5), write=bad_write,
                           hold=bad_hold)
    msg = str(hold_measured({"left": arm2}, lambda pos, grip: grip)["left"])
    assert "bus write" in msg and "hold bus" in msg


def test_first_alarm_prefix_test_not_substring():
    hot = SimpleNamespace(alarm=lambda: "temp of left gripper high", is_alive=lambda: True)  # mentions "left"
    assert first_alarm({"left": hot}) == "left: temp of left gripper high"
    hot2 = SimpleNamespace(alarm=lambda: "arm 'leftover': hot", is_alive=lambda: True)
    assert first_alarm({"left": hot2}) == "left: arm 'leftover': hot"


def test_restore_link_repoints_to_the_previous_run(tmp_path):
    link = tmp_path / "current"
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    StateFile(old, link=link).link_current()
    sf = StateFile(new, link=link)
    sf.link_current()
    assert link.resolve() == new.resolve()
    sf.restore_link()
    assert link.resolve() == old.resolve()


def test_restore_link_removes_a_link_that_had_no_predecessor(tmp_path):
    link = tmp_path / "current"
    sf = StateFile(tmp_path / "run", link=link)
    sf.link_current()
    sf.restore_link()
    assert not link.is_symlink()


def test_restore_link_leaves_a_link_someone_else_moved(tmp_path):
    link = tmp_path / "current"
    other = tmp_path / "other"
    other.mkdir()
    sf = StateFile(tmp_path / "run", link=link)
    sf.link_current()
    StateFile(other, link=link).link_current()
    sf.restore_link()
    assert link.resolve() == other.resolve()


def test_run_dir_default_never_reuses_an_existing_dir(tmp_path, monkeypatch):
    from harness.__main__ import _run_dir
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(time, "strftime", lambda fmt: "20261108-100000")
    first = _run_dir(None)
    second = _run_dir(None)
    assert first.name == "20261108-100000" and second.name == f"20261108-100000-{os.getpid()}"
    assert first != second and first.is_dir() and second.is_dir()


def test_run_dir_refuses_an_out_dir_with_a_state_file(tmp_path):
    from harness.__main__ import _run_dir
    (tmp_path / "state.json").write_text("{}")
    with pytest.raises(SystemExit) as exc:
        _run_dir(tmp_path)
    assert "state.json" in str(exc.value.code)
    assert _run_dir(tmp_path / "fresh") == tmp_path / "fresh"


def _main_subprocess(tmp_path, body: str, extra_args: str = "") -> subprocess.CompletedProcess:
    """Runs harness.__main__.main(['run', ...]) with run_match replaced by `body` (a function body)."""
    code = (
        "import json, signal, sys, types\n"
        "import harness.__main__ as m\n"
        "stub = types.ModuleType('harness.dream_executor')\n"
        "async def run_match(match, ckpt, dry_run, out_dir, ctl, state, **kw):\n"
        + "".join(f"    {line}\n" for line in body.splitlines())
        + "stub.run_match = run_match\n"
        "sys.modules['harness.dream_executor'] = stub\n"
        f"out = {str(tmp_path / 'run')!r}\n"
        f"m.main(['run', '--out', out{extra_args}])\n"
    )
    env = {**os.environ, "HOME": str(tmp_path)}
    return subprocess.run([sys.executable, "-I", "-c", "import sys; sys.path.insert(0, %r)\n" % str(REPO) + code],
                          cwd=tmp_path, capture_output=True, text=True, timeout=60, env=env)


def test_main_color_omitted_end_to_end(tmp_path):
    res = _main_subprocess(tmp_path,
                           "assert match.own_color is None, match.own_color\n"
                           "state.update(state='done', score=0)\n"
                           "return types.SimpleNamespace(score=lambda: 0, passed={})")
    assert res.returncode == 0, res.stderr
    assert "score 0" in res.stdout


def test_main_exit_code_is_nonzero_for_a_pre_open_abort(tmp_path):
    res = _main_subprocess(tmp_path,
                           "state.update(state='done', error='frozen before the arms opened', "
                           "reason='frozen_before_open')\n"
                           "return types.SimpleNamespace(score=lambda: 0, passed={})")
    assert res.returncode == 3, (res.returncode, res.stderr)
    st = json.loads((tmp_path / "run" / "state.json").read_text())
    assert st["state"] == "done" and st["reason"] == "frozen_before_open"  # still a clean done, not an error


def test_main_exit_code_is_zero_for_a_normal_run_and_a_park_freeze(tmp_path):
    res = _main_subprocess(tmp_path,
                           "state.update(state='done', error='freeze arrived during the park', "
                           "reason='freeze_during_park')\n"
                           "return types.SimpleNamespace(score=lambda: 0, passed={})")
    assert res.returncode == 0, res.stderr


def test_main_initial_state_has_the_round3_fields(tmp_path):
    res = _main_subprocess(tmp_path, "assert state.data['reason'] is None\n"
                                     "assert state.data['eight_ball_alert'] is False\n"
                                     "state.update(state='done')\n"
                                     "return types.SimpleNamespace(score=lambda: 0, passed={})")
    assert res.returncode == 0, res.stderr


def test_main_systemexit_with_a_status_gets_a_real_message(tmp_path):
    res = _main_subprocess(tmp_path, "raise SystemExit(2)")
    assert res.returncode == 2
    st = json.loads((tmp_path / "run" / "state.json").read_text())
    assert st["state"] == "error" and "status 2" in st["error"]


def test_main_restores_the_original_handlers_after_the_run(tmp_path):
    code = (
        "import signal, sys, types\n"
        "import harness.__main__ as m\n"
        "names = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGUSR1)\n"
        "before = {n: signal.getsignal(n) for n in names}\n"
        "stub = types.ModuleType('harness.dream_executor')\n"
        "async def run_match(match, ckpt, dry_run, out_dir, ctl, state, **kw):\n"
        "    assert signal.getsignal(signal.SIGUSR1) != before[signal.SIGUSR1]\n"
        "    state.update(state='done')\n"
        "    return types.SimpleNamespace(score=lambda: 0, passed={})\n"
        "stub.run_match = run_match\n"
        "sys.modules['harness.dream_executor'] = stub\n"
        f"m.main(['run', '--out', {str(tmp_path / 'run')!r}])\n"
        "assert {n: signal.getsignal(n) for n in names} == before, 'handlers not restored'\n"
    )
    env = {**os.environ, "HOME": str(tmp_path)}
    res = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=60, env=env)
    assert res.returncode == 0, res.stderr


def test_main_second_early_end_signal_forces_the_default(tmp_path):
    code = (
        "import os, signal, sys, types\n"
        "class Empty:\n"
        "    def create_module(self, spec): return types.ModuleType('cv2')\n"
        "    def exec_module(self, module): pass\n"
        "class Blocker:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'cv2':\n"
        "            os.kill(os.getpid(), signal.SIGTERM)  # the first end: asked politely\n"
        "            os.kill(os.getpid(), signal.SIGINT)   # the second: force\n"
        "            import time; time.sleep(5)  # never reached: the second signal raises here\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocker())\n"
        "import harness.__main__ as m\n"
        f"m.main(['run', '--out', {str(tmp_path / 'run')!r}])\n"
    )
    env = {**os.environ, "HOME": str(tmp_path)}
    res = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=30, env=env)
    assert res.returncode != 0 and "KeyboardInterrupt" in res.stderr, (res.returncode, res.stderr)
    st = json.loads((tmp_path / "run" / "state.json").read_text())
    assert st["state"] == "error" and "end signal" in st["error"]


def test_sdk_contract_the_apis_the_harness_uses_exist():
    """Against the REAL Dream SDK (read-only; skipped where it is not installed)."""
    import dataclasses
    import inspect

    pytest.importorskip("robots.arms_lock")
    from robots import arms_lock, bodies, followers, home, types as rtypes
    dream = pytest.importorskip("dream")

    assert list(inspect.signature(arms_lock.holding).parameters)[:2] == ["path", "holder"]
    params = inspect.signature(dream.DebugView).parameters
    assert "port" in params and list(params)[0] == "client"
    assert callable(dream.DebugView.watch) and len(inspect.signature(dream.DebugView.watch).parameters) == 2
    assert "sleep" in inspect.signature(home.return_to_home).parameters
    assert "duration_s" in inspect.signature(home.return_to_home).parameters
    assert {"positions_rad", "gripper"} <= {f.name for f in dataclasses.fields(rtypes.ArmCommand)}
    for name in ("alarm", "hold", "read", "write", "is_alive", "torque_off"):
        assert callable(getattr(followers.FollowerArm, name)), name
    assert "_last_command" in inspect.getsource(followers.FollowerArm.__init__)
    assert "open_arm_read_only" in {f.name for f in dataclasses.fields(bodies.Body)}
