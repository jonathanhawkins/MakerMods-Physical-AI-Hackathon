"""run_match with Dream's `robots`/`dream` packages stubbed (never imports the real ones, opens no hardware).

`stub_dream(monkeypatch)` returns a StubWorld that later tests reuse. It records the call order in `world.calls`
(open_arm, home, torque_off, write, lock_acquired, lock_released, view_opened, view_closed), lets a test choose how
the control loop ends (`world.loop_error`, or `world.loop_gate` to hold it until the test sets it) and replaces the
vision, the relay and the supervisor with trivial stand-ins that finish the relay at once (`world.finish = False`
keeps the supervisor waiting for a stop).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
import os
import signal
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness.config import MatchConfig
from harness.freeze import FreezeControl, StateFile

# harness.claude_check imports anthropic at module level and the main env lacks it: import dream_executor with a
# stand-in, removed again afterwards so it does not leak into other tests.
_saved = sys.modules.get("anthropic")
if _saved is None:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        sys.modules["anthropic"] = types.ModuleType("anthropic")
from harness import dream_executor  # noqa: E402
if _saved is None and isinstance(sys.modules.get("anthropic"), types.ModuleType) \
        and not hasattr(sys.modules["anthropic"], "Anthropic"):
    del sys.modules["anthropic"]


class _Tee(list):
    """`world.calls`; every entry also lands in `world.order`, which interleaves calls with state.json writes."""

    def __init__(self, order: list) -> None:
        super().__init__()
        self.order = order

    def append(self, item) -> None:
        super().append(item)
        self.order.append(item)


class StubWorld:
    def __init__(self) -> None:
        self.order: list[str] = []  # calls and "state:<name>" writes, in order
        self.calls: list[str] = _Tee(self.order)
        self.on_open_client = None  # called inside open_client (a worker thread)
        self.on_open_arm = None  # called inside open_arm, on the event loop thread
        self.on_home_done = None  # called as (home_call_no) after a home move's last sleep
        self.ik_calls = 0
        self.on_home_sleep = None  # called as (home_call_no, sleep_no) inside each home move's sleeps
        self.home_calls = 0
        self.loop_exits = 0
        self.alarm: str | None = None
        self.states: list[dict] = []  # every state.json write, in order
        self.lock_error: Exception | None = None
        self.view_error: OSError | None = None
        self.lock_args: list = []
        self.view_ports: list[int] = []
        self.watched: list = []
        self.view_clients: list = []
        self.loop_error: Exception | None = None
        self.loop_gate: asyncio.Event | None = None  # ControlLoop.run() waits on it when set
        self.finish = True  # the stub supervisor ends the relay at once
        self.arms: dict = {}
        self.home_sleeps = 0
        self.home_result = True  # what return_to_home returns
        self.exit_error: Exception | None = None  # raised by the control loop's __aexit__
        self.on_write = None  # called inside every arm write (a worker thread)
        self.no_frame = False  # the overhead camera never sends a frame
        self.on_reader_start = None  # called as the overhead reader starts
        self.camera_opens = 0
        self.probe = lambda: None  # ClaudeVerifier.probe, runs in a worker thread
        self.client = SimpleNamespace(name="client", wait_for_ik=self._wait_for_ik)


StubWorld._wait_for_ik = lambda self, t: (setattr(self, "ik_calls", self.ik_calls + 1), True)[1]


class StubArm:
    home_rad = [0.0]

    def __init__(self, world: StubWorld, name: str) -> None:
        self.world, self.name = world, name

    def write(self, cmd) -> None:
        self.world.calls.append("write")
        if self.world.on_write is not None:
            self.world.on_write()

    def read(self):
        return SimpleNamespace(positions_rad=[0.1], gripper=1.5)  # gripper out of range: hold clamps it

    def is_alive(self) -> bool:
        return True

    def alarm(self):
        return self.world.alarm

    def torque_off(self) -> None:
        self.world.calls.append("torque_off")


class StubRelay:
    m_stages = ("unplug", "stack", "balls")
    stage = "unplug"
    step = SimpleNamespace(instruction="unplug cables", kind="unplug", stage="unplug")
    passed = {"unplug": 3.0}
    own_in_bin = other_in_bin = 0
    placed = None
    events: list = []
    cal = None

    def __init__(self, match) -> None:
        pass

    def start(self, world, t) -> None:
        pass

    def next_step(self, world, t):
        return self.step

    def score(self):
        return 10

    def time_left(self, now):
        return 100.0

    def log(self, t, what) -> None:
        pass


def _module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    return mod


@pytest.fixture
def stub_dream(monkeypatch) -> StubWorld:
    world = StubWorld()

    @dataclasses.dataclass(frozen=True)
    class ArmCommand:
        positions_rad: list
        gripper: float

    class ArmError(Exception): ...
    class SafetyStop(Exception): ...
    class CameraError(Exception): ...

    errors = _module("robots.errors", ArmError=ArmError, SafetyStop=SafetyStop, CameraError=CameraError,
                     StationError=type("StationError", (Exception,), {}))

    @contextlib.contextmanager
    def holding(path, holder):
        world.lock_args.append((path, holder))
        if world.lock_error is not None:
            raise world.lock_error
        world.calls.append("lock_acquired")
        try:
            yield
        finally:
            world.calls.append("lock_released")

    arms_lock = _module("robots.arms_lock", holding=holding, LOCK_NAME="arms.lock")

    class ControlLoop:
        task = ""
        video: dict = {}
        state = None

        def __init__(self, **kw) -> None:
            self.task = kw["task"]

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            world.calls.append("loop_exit")
            if world.exit_error is not None:
                raise world.exit_error
            return False

        async def run(self) -> None:
            if world.loop_error is not None:
                raise world.loop_error
            if world.loop_gate is not None:
                await world.loop_gate.wait()
            else:
                await asyncio.Event().wait()  # until cancelled

        def set_task(self, task) -> None:
            self.task = task

    control = _module("robots.control", ControlLoop=ControlLoop,
                      run_teardown=lambda what, fn: fn())

    class FollowerArm:
        def __init__(self, name, arm, home_rad=None) -> None:
            self.arm = arm

        def torque_off(self) -> None:
            self.arm.torque_off()

        def read(self):
            return self.arm.read()

        def write(self, cmd) -> None:
            self.arm.write(cmd)

        def is_alive(self) -> bool:
            return self.arm.is_alive()

        def alarm(self):
            return self.arm.alarm()

    class DryArm:  # Dream's DryRun: reads through, swallows writes
        def __init__(self, arm) -> None:
            self.arm = arm

        def read(self):
            return self.arm.read()

        def write(self, cmd) -> None:
            world.calls.append("dry_write")

        def is_alive(self) -> bool:
            return self.arm.is_alive()

        def alarm(self):
            return self.arm.alarm()

        def torque_off(self) -> None:
            self.arm.torque_off()

    followers = _module("robots.followers", FollowerArm=FollowerArm)

    def return_to_home(arms, *, duration_s=0.0, sleep=None, **kw) -> bool:
        world.calls.append("home")
        world.home_calls += 1
        for i in range(3):
            if world.on_home_sleep is not None:
                world.on_home_sleep(world.home_calls, i + 1)
            (sleep or (lambda dt: None))(0.0)
            world.home_sleeps += 1
        if world.on_home_done is not None:
            world.on_home_done(world.home_calls)
        return world.home_result

    home = _module("robots.home", return_to_home=return_to_home, HOME_DURATION_S=1.0)

    def open_client(cfg):
        world.calls.append("open_client")
        if world.on_open_client is not None:
            world.on_open_client()
        return world.client

    def make_cameras(app):
        world.camera_opens += 1
        frame = None if world.no_frame else SimpleNamespace(data=None, timestamp_s=0.0)
        return [SimpleNamespace(name="scene", frame=frame,
                                start=lambda: world.on_reader_start() if world.on_reader_start else None,
                                stop=lambda: None, set_exposure_hold=lambda on: None)]

    rrun = _module(
        "robots.run",
        make_cameras=make_cameras,
        make_kinematics=lambda app: None, open_client=open_client, DryRun=DryArm,
        loop_settings=lambda app: {}, policy_config=lambda app, checkpoint: {"checkpoint": checkpoint},
        with_task=lambda cfg, task: cfg, TRAJ_WARMUP_S=1.0)

    def open_arm(cfg, name):
        world.calls.append("open_arm")
        if world.on_open_arm is not None:
            world.on_open_arm()
        arm = StubArm(world, name)
        world.arms[name] = arm
        return arm

    bodies = _module("robots.bodies", for_config=lambda app: SimpleNamespace(
        open_arm=open_arm, open_arm_read_only=None))
    rconfig = _module("robots.config", load=lambda path: SimpleNamespace(
        follower_l=None, follower_r=None, inference_log_path=None))
    rtypes = _module("robots.types", ArmCommand=ArmCommand)
    robots = _module("robots", config=rconfig, control=control, followers=followers, home=home, run=rrun,
                     bodies=bodies, errors=errors, arms_lock=arms_lock, types=rtypes)
    robots.__path__ = []

    class DebugView:
        def __init__(self, client, port=0) -> None:
            if world.view_error is not None:
                raise world.view_error
            world.calls.append("view_opened")
            world.view_ports.append(port)
            world.view_clients.append(client)
            self.url = f"http://127.0.0.1:{port}/"

        def watch(self, client) -> None:
            world.watched.append(client)

        def close(self) -> None:
            world.calls.append("view_closed")

    dream = _module("dream", DebugView=DebugView)

    for mod in (robots, rconfig, control, followers, home, rrun, bodies, errors, arms_lock, rtypes, dream):
        monkeypatch.setitem(sys.modules, mod.__name__, mod)

    # Stand-ins for the vision, the relay, the supervisor and the report: this file tests run_match's wiring.
    fake_vision = SimpleNamespace(locate_layout=lambda img, cal: SimpleNamespace(note=lambda: "stub"),
                                  with_layout=lambda cal, layout: SimpleNamespace(bin_roi=None, hub_roi=None),
                                  detect=lambda *a: None, draw=lambda *a: None)
    monkeypatch.setattr(dream_executor, "vision", fake_vision)
    monkeypatch.setattr(dream_executor.cv2, "imwrite", lambda *a: True)
    monkeypatch.setattr(dream_executor, "Relay", StubRelay)
    monkeypatch.setattr(dream_executor, "ClaudeVerifier",
                        lambda: SimpleNamespace(probe=lambda: world.probe(), available=False))
    monkeypatch.setattr(dream_executor, "_write_report", lambda relay, out_dir: None)

    async def supervise(loop, relay, match, stop, out_dir, claude, go_rgb=None, **kw):
        if world.finish:
            await asyncio.sleep(0)
            stop.set()
        await stop.wait()

    monkeypatch.setattr(dream_executor, "_supervise", supervise)
    return world


class RecordingState(StateFile):
    def __init__(self, out_dir: Path, world: StubWorld) -> None:
        super().__init__(out_dir, link=out_dir / "link")
        self.world = world

    def update(self, **fields):
        out = super().update(**fields)
        self.world.states.append(out)
        self.world.order.append(f"state:{out['state']}")
        return out


_running = threading.Event()  # set while run_match runs: driver threads send SIGINT only then


def run(world: StubWorld, tmp_path: Path, ctl: FreezeControl | None = None, dry_run: bool = False, **kw):
    state = RecordingState(tmp_path, world)
    match = MatchConfig(own_color="white")
    coro = dream_executor.run_match(match, "ckpt", dry_run=dry_run, out_dir=tmp_path, ctl=ctl or FreezeControl(),
                                    state=state, **kw)
    _running.set()
    try:
        return asyncio.run(coro)
    finally:
        _running.clear()


def test_holds_arms_lock_for_whole_run(stub_dream, tmp_path):
    run(stub_dream, tmp_path)
    calls = stub_dream.calls
    assert calls[0] == "lock_acquired"
    assert calls[-1] == "lock_released"
    assert calls.index("lock_released") > max(i for i, c in enumerate(calls) if c == "torque_off")
    assert calls.index("open_arm") > calls.index("lock_acquired")
    path, holder = stub_dream.lock_args[0]
    assert path == Path("~/.dreamlabs/arms.lock").expanduser() and holder == "harness run"


def test_lock_held_elsewhere_fails_cleanly(stub_dream, tmp_path):
    stub_dream.lock_error = RuntimeError("the arms are held by 1234 dreamcli robot run")
    with pytest.raises(RuntimeError, match="dreamcli robot run"):
        run(stub_dream, tmp_path)
    assert "open_arm" not in stub_dream.calls
    final = json.loads((tmp_path / "state.json").read_text())
    assert final["state"] == "error" and "dreamcli robot run" in final["error"]


def test_debugview_opened_on_8093_and_closed(stub_dream, tmp_path):
    run(stub_dream, tmp_path)
    assert stub_dream.view_ports == [8093]
    assert stub_dream.view_clients == [stub_dream.client]
    calls = stub_dream.calls
    assert calls.index("open_client") < calls.index("view_opened") < calls.index("view_closed")


def test_debugview_oserror_is_a_warning(stub_dream, tmp_path, caplog):
    stub_dream.view_error = OSError("port in use")
    with caplog.at_level(logging.WARNING):
        run(stub_dream, tmp_path)
    assert "view_closed" not in stub_dream.calls and "open_arm" in stub_dream.calls
    assert any("8093" in r.message for r in caplog.records)


def test_state_transitions_normal_end(stub_dream, tmp_path):
    run(stub_dream, tmp_path)
    names = [s["state"] for s in stub_dream.states]
    dedup = [n for i, n in enumerate(names) if i == 0 or n != names[i - 1]]
    assert dedup == ["homing", "running", "homing", "parking", "done"]
    assert stub_dream.states[-1]["score"] == 10
    assert type(stub_dream.states[-1]["score"]) is int


def test_exception_writes_error_state(stub_dream, tmp_path):
    stub_dream.finish = False
    stub_dream.loop_error = RuntimeError("loop blew up")
    with pytest.raises(RuntimeError, match="loop blew up"):
        run(stub_dream, tmp_path)
    final = json.loads((tmp_path / "state.json").read_text())
    assert final["state"] == "error" and "loop blew up" in final["error"]
    calls = stub_dream.calls
    assert calls.count("torque_off") == 2
    assert calls.index("lock_released") > max(i for i, c in enumerate(calls) if c == "torque_off")


def test_report_failure_still_releases_lock_and_writes_error(stub_dream, tmp_path, monkeypatch):
    def boom(relay, out_dir):
        raise OSError("disk full")
    monkeypatch.setattr(dream_executor, "_write_report", boom)
    with pytest.raises(OSError):
        run(stub_dream, tmp_path)
    assert stub_dream.calls[-1] == "lock_released"
    assert stub_dream.states[-1]["state"] == "error"


def test_view_follows_restarted_client(stub_dream, tmp_path):
    stub_dream.finish = False
    stub_dream.loop_error = RuntimeError("stalled")
    with pytest.raises(RuntimeError):
        run(stub_dream, tmp_path)
    assert stub_dream.watched == [stub_dream.client] * 3  # one per restart


def test_real_supervise_writes_stage(tmp_path, monkeypatch):
    # The real _supervise (the fixture replaces it): a stage change must reach state.json, although the loop's
    # own `state` is a robot state.
    monkeypatch.setattr(dream_executor, "POLL_S", 0.001)
    monkeypatch.setattr(dream_executor.vision, "detect", lambda *a: None)
    frame = SimpleNamespace(data=None, timestamp_s=0.0)
    arm = SimpleNamespace(gripper=0.0)
    loop = SimpleNamespace(video={"scene": frame}, state=SimpleNamespace(left=arm, right=arm), task="t")
    relay = SimpleNamespace(stack_start_rgb=None, stage="unplug", confirm_kinds=set(), pending_confirm=None, step=None,
                            observe=lambda *a: dream_executor.Outcome.RUNNING)
    statefile = StateFile(tmp_path, link=tmp_path / "link")
    stop = asyncio.Event()

    async def go():
        task = asyncio.create_task(dream_executor._supervise(
            loop, relay, SimpleNamespace(calibration=None, all_colors=False, stack_pass="6in"), stop, tmp_path, SimpleNamespace(available=False), statefile=statefile))
        await asyncio.sleep(0.05)
        relay.stage = "stack"
        await asyncio.sleep(0.05)
        stop.set()
        await task

    asyncio.run(go())
    assert json.loads((tmp_path / "state.json").read_text())["stage"] == "stack"


# ---------------------------------------------------------------------------------------------------------------
# Freeze and end. Real signals to this process (the handlers are the real ones); the arms are stubs.

@pytest.fixture
def ctl():
    """A FreezeControl with the real SIGUSR1 handler; the previous handler comes back afterwards."""
    old = signal.getsignal(signal.SIGUSR1)
    control = FreezeControl()
    control.install_freeze_signal()
    yield control
    signal.signal(signal.SIGUSR1, old)


@pytest.fixture
def fast_poll(monkeypatch):
    monkeypatch.setattr(dream_executor, "FROZEN_POLL_S", 0.02)
    monkeypatch.setattr(dream_executor, "ALARM_POLL_S", 0.04)


def freeze(ctl: FreezeControl) -> None:
    os.kill(os.getpid(), signal.SIGUSR1)
    assert ctl.freeze.wait(5), "the freeze handler never ran"


def end() -> None:
    os.kill(os.getpid(), signal.SIGINT)


def wait_for(cond, timeout=5.0):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if cond():
            return True
        time.sleep(0.005)
    raise AssertionError("timed out")


def has_state(world, name) -> bool:
    return any(s["state"] == name for s in world.states)


def in_thread(fn):
    errors: list[BaseException] = []

    def body():
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
            if _running.is_set():
                end()  # never leave run_match waiting (but never SIGINT a pytest that is no longer in run_match)
    t = threading.Thread(target=body, daemon=True)
    t.start()
    return t, errors


def finish_driver(t, errors):
    t.join(10)
    assert not errors, errors


def after_frozen(order: list[str]) -> list[str]:
    return order[order.index("state:frozen") + 1:]


def test_freeze_before_open_arm_aborts_without_arms(stub_dream, tmp_path, ctl):
    stub_dream.on_open_client = lambda: freeze(ctl)
    run(stub_dream, tmp_path, ctl)
    calls = stub_dream.calls
    assert "open_arm" not in calls and "home" not in calls and "torque_off" not in calls
    assert calls[-1] == "lock_released"
    assert "state:parking" not in stub_dream.order  # no arms: nothing to park
    final = json.loads((tmp_path / "state.json").read_text())
    assert final["state"] == "done" and final["error"] == "frozen before the arms opened"


def test_freeze_after_open_arm_skips_startup_home(stub_dream, tmp_path, ctl, fast_poll):
    class Opener(dict):
        def __setitem__(self, k, v):
            super().__setitem__(k, v)
            if len(self) == 2:
                freeze(ctl)  # the freeze lands as the second arm opens
    stub_dream.arms = Opener()

    def driver():
        wait_for(lambda: has_state(stub_dream, "frozen"))
        end()
    t, errors = in_thread(driver)
    run(stub_dream, tmp_path, ctl)
    finish_driver(t, errors)
    assert stub_dream.order.index("state:frozen") < stub_dream.order.index("home")  # the only home is the end one
    assert stub_dream.calls.count("home") == 1


def test_freeze_during_startup_home_holds_no_torque_off(stub_dream, tmp_path, ctl, fast_poll):
    stub_dream.on_home_sleep = lambda n, i: freeze(ctl) if (n, i) == (1, 2) else None
    snap: dict = {}

    def driver():
        wait_for(lambda: has_state(stub_dream, "frozen"))
        time.sleep(0.2)
        snap["order"] = list(stub_dream.order)
        end()
    t, errors = in_thread(driver)
    run(stub_dream, tmp_path, ctl)
    finish_driver(t, errors)
    before = snap["order"]
    assert before.count("write") == 2 and "torque_off" not in before and "state:running" not in before
    assert before.index("home") < before.index("state:frozen")
    tail = [e for e in after_frozen(stub_dream.order) if e in ("state:homing", "home", "state:parking",
                                                                "torque_off", "state:done")]
    # the held arms were commanded once each before the end (2 writes), then the end path ran
    assert tail == ["state:homing", "home", "state:parking", "torque_off", "torque_off", "state:done"]
    assert stub_dream.calls.count("write") == 2


def test_freeze_during_loop_holds_until_end(stub_dream, tmp_path, ctl, monkeypatch):
    stub_dream.finish = False
    logs: list[str] = []
    monkeypatch.setattr(StubRelay, "log", lambda self, t, what: logs.append(what))
    snap: dict = {}

    def driver():
        wait_for(lambda: has_state(stub_dream, "running"))
        time.sleep(0.1)
        freeze(ctl)
        wait_for(lambda: has_state(stub_dream, "frozen"))
        mark = len(stub_dream.order)
        time.sleep(1.0)
        snap["quiet"] = stub_dream.order[mark:]
        snap["order"] = list(stub_dream.order)
        end()
    t, errors = in_thread(driver)
    run(stub_dream, tmp_path, ctl)
    finish_driver(t, errors)
    order = snap["order"]
    assert order.index("loop_exit") < order.index("state:frozen")  # frozen only once the loop let go
    assert order.index("loop_exit") < order.index("write")  # ... and the first measured write comes after it
    assert "frozen at stage unplug" in logs  # the report records the freeze
    assert snap["quiet"] == []  # nothing touched the arms or the state for 1 s
    assert order.count("write") == 2 and "torque_off" not in order and order.count("home") == 1  # start-up home only
    assert stub_dream.states[-1]["state"] == "done"


def test_end_then_freeze_during_end_home_does_not_fall_through(stub_dream, tmp_path, ctl, fast_poll):
    stub_dream.finish = False
    stub_dream.on_home_sleep = lambda n, i: freeze(ctl) if (n, i) == (2, 2) else None
    snap: dict = {}

    def driver():
        wait_for(lambda: has_state(stub_dream, "running"))
        end()
        wait_for(lambda: has_state(stub_dream, "frozen"))
        time.sleep(0.3)  # the SIGINT's end request was cancelled by the freeze: it must not end the run
        snap["order"] = list(stub_dream.order)
        end()
    t, errors = in_thread(driver)
    run(stub_dream, tmp_path, ctl)
    finish_driver(t, errors)
    order = snap["order"]
    assert order.index("state:homing") < order.index("state:frozen")
    assert "torque_off" not in order and "state:parking" not in order
    tail = [e for e in after_frozen(stub_dream.order) if e in ("state:homing", "home", "state:parking",
                                                                "torque_off", "state:done")]
    assert tail == ["state:homing", "home", "state:parking", "torque_off", "torque_off", "state:done"]
    assert stub_dream.calls.count("write") == 2  # one measured hold per arm, for the one freeze


def test_alarm_while_frozen_is_recorded_and_still_holds(stub_dream, tmp_path, ctl, fast_poll):
    stub_dream.finish = False
    snap: dict = {}

    def driver():
        wait_for(lambda: has_state(stub_dream, "running"))
        freeze(ctl)
        wait_for(lambda: has_state(stub_dream, "frozen"))
        stub_dream.alarm = "joint 3 at 70 C"
        state_path = tmp_path / "state.json"
        wait_for(lambda: json.loads(state_path.read_text()).get("alarm"))
        time.sleep(0.2)
        snap["state"] = json.loads(state_path.read_text())
        snap["order"] = list(stub_dream.order)
        end()
    t, errors = in_thread(driver)
    run(stub_dream, tmp_path, ctl)
    finish_driver(t, errors)
    assert snap["state"]["state"] == "frozen" and "70 C" in snap["state"]["alarm"]
    assert "torque_off" not in snap["order"] and snap["order"].count("write") == 2


def test_hold_failure_is_written_as_alarm(stub_dream, tmp_path, ctl, fast_poll, monkeypatch):
    stub_dream.finish = False
    snap: dict = {}

    def boom(self, cmd):
        raise OSError("bus off")
    monkeypatch.setattr(StubArm, "write", boom)

    def driver():
        wait_for(lambda: has_state(stub_dream, "running"))
        freeze(ctl)
        wait_for(lambda: has_state(stub_dream, "frozen"))
        snap["state"] = json.loads((tmp_path / "state.json").read_text())
        end()
    t, errors = in_thread(driver)
    run(stub_dream, tmp_path, ctl)
    finish_driver(t, errors)
    assert snap["state"]["state"] == "frozen" and "bus off" in snap["state"]["alarm"]


def test_end_request_cancelled_by_freeze_in_frozen_wait_keeps_holding(stub_dream, tmp_path, ctl, fast_poll,
                                                                      monkeypatch):
    # begin_end() False (a freeze landed between the end request and the poll): back to waiting, no home move.
    stub_dream.finish = False
    snap: dict = {}
    real = ctl.begin_end
    seen: list[bool] = []

    def begin_end():
        ctl.end_requested.clear()  # as the SIGUSR1 handler does
        seen.append(real())
        return seen[-1]
    monkeypatch.setattr(ctl, "begin_end", begin_end)

    def driver():
        wait_for(lambda: has_state(stub_dream, "running"))
        freeze(ctl)
        wait_for(lambda: has_state(stub_dream, "frozen"))
        end()  # begin_end answers False for this one
        wait_for(lambda: bool(seen))
        time.sleep(0.2)
        assert "state:homing" not in after_frozen(stub_dream.order) and "torque_off" not in stub_dream.order
        snap["writes"] = stub_dream.calls.count("write")
        monkeypatch.setattr(ctl, "begin_end", real)
        end()
    t, errors = in_thread(driver)
    run(stub_dream, tmp_path, ctl)
    finish_driver(t, errors)
    assert seen == [False]
    assert snap["writes"] == 4  # the first hold (2 arms) plus one re-hold when the end request was cancelled
    assert "state:homing" in after_frozen(stub_dream.order)


def test_dry_run_freeze_and_end(stub_dream, tmp_path, ctl, fast_poll):
    stub_dream.finish = False
    snap: dict = {}

    def driver():
        wait_for(lambda: has_state(stub_dream, "running"))
        freeze(ctl)
        wait_for(lambda: has_state(stub_dream, "frozen"))
        time.sleep(0.1)
        snap["order"] = list(stub_dream.order)
        end()
    t, errors = in_thread(driver)
    run(stub_dream, tmp_path, ctl, dry_run=True)
    finish_driver(t, errors)
    assert "torque_off" not in snap["order"]
    assert "write" not in stub_dream.calls and "home" not in stub_dream.calls  # nothing moved, nothing homed
    assert stub_dream.calls.count("dry_write") == 2
    assert stub_dream.calls.count("torque_off") == 2  # the DryRun close, on the end path only
    assert stub_dream.states[-1]["state"] == "done"


def test_end_then_freeze_in_the_same_busy_moment_stays_frozen(stub_dream, tmp_path, ctl, fast_poll):
    # SIGINT then SIGUSR1 while the loop thread is blocked in open_arm: both handlers run at once, in order, so the
    # freeze wins. (A deferred end handler would set end_requested again after the freeze cleared it.)
    started = threading.Event()

    def slow_open():
        if not started.is_set():
            started.set()
            time.sleep(0.4)
    stub_dream.on_open_arm = slow_open
    snap: dict = {}

    def driver():
        assert started.wait(5)
        end()
        time.sleep(0.05)
        freeze(ctl)
        wait_for(lambda: has_state(stub_dream, "frozen"))
        time.sleep(0.3)
        snap["order"] = list(stub_dream.order)
        end()
    t, errors = in_thread(driver)
    run(stub_dream, tmp_path, ctl)
    finish_driver(t, errors)
    order = snap["order"]
    assert "state:frozen" in order and "torque_off" not in order and "state:parking" not in order
    assert "home" not in order  # never homed before the second end
    assert "state:homing" not in order[order.index("state:frozen"):]  # the start-up homing write is earlier


def test_freeze_again_inside_frozen_wait_end_home(stub_dream, tmp_path, ctl, fast_poll):
    stub_dream.finish = False
    # home call 1 is the start-up one; call 2 is the end move started from the frozen wait.
    stub_dream.on_home_sleep = lambda n, i: freeze(ctl) if (n, i) == (2, 2) else None
    snap: dict = {}

    def frozen_count():
        return sum(1 for s in stub_dream.states if s["state"] == "frozen")

    def driver():
        wait_for(lambda: has_state(stub_dream, "running"))
        freeze(ctl)
        wait_for(lambda: frozen_count() == 1)
        end()
        wait_for(lambda: frozen_count() == 2)
        time.sleep(0.2)
        snap["order"] = list(stub_dream.order)
        end()
    t, errors = in_thread(driver)
    run(stub_dream, tmp_path, ctl)
    finish_driver(t, errors)
    order = snap["order"]
    assert "state:parking" not in order and "torque_off" not in order
    assert order.count("write") == 4 and order.count("home") == 2  # two holds; start-up and the interrupted end move
    tail = [e for e in after_frozen(after_frozen(stub_dream.order))
            if e in ("state:homing", "home", "state:parking", "torque_off", "state:done")]
    assert tail == ["state:homing", "home", "state:parking", "torque_off", "torque_off", "state:done"]


def test_freeze_during_person_in_scene_pause_is_prompt(stub_dream, tmp_path, ctl, monkeypatch):
    stub_dream.finish = False
    stub_dream.loop_error = RuntimeError("unsafe instruction in this scene")
    logs: list[str] = []
    monkeypatch.setattr(StubRelay, "log", lambda self, t, what: logs.append(what))
    snap: dict = {}

    def driver():
        wait_for(lambda: any("person is in the scene" in m for m in logs))
        t0 = time.monotonic()
        freeze(ctl)
        wait_for(lambda: has_state(stub_dream, "frozen"), timeout=2.0)
        snap["delay"] = time.monotonic() - t0
        end()
    t, errors = in_thread(driver)
    run(stub_dream, tmp_path, ctl)
    finish_driver(t, errors)
    assert snap["delay"] < 1.0  # not the 3 s pause


def test_freeze_before_restart_does_no_restart_work(stub_dream, tmp_path, ctl, monkeypatch):
    stub_dream.finish = False
    stub_dream.loop_error = RuntimeError("stalled")

    def log(self, t, what):
        if "Dream loop stopped" in what:
            freeze(ctl)  # the freeze lands just before the restart work
    monkeypatch.setattr(StubRelay, "log", log)

    def driver():
        wait_for(lambda: has_state(stub_dream, "frozen"))
        end()
    t, errors = in_thread(driver)
    run(stub_dream, tmp_path, ctl)
    finish_driver(t, errors)
    assert stub_dream.calls.count("open_client") == 1 and stub_dream.ik_calls == 1  # the initial ones only
    assert has_state(stub_dream, "frozen") and stub_dream.states[-1]["state"] == "done"


def test_loop_fault_survives_a_freeze(stub_dream, tmp_path, ctl, monkeypatch):
    stub_dream.finish = False
    stub_dream.loop_error = RuntimeError("stalled")

    def log(self, t, what):
        if "restart 4" in what:
            freeze(ctl)  # restart 4 is the one that gives up and records the fault
    monkeypatch.setattr(StubRelay, "log", log)

    def driver():
        wait_for(lambda: has_state(stub_dream, "frozen"))
        end()
    t, errors = in_thread(driver)
    with pytest.raises(RuntimeError, match="stalled"):
        run(stub_dream, tmp_path, ctl)
    finish_driver(t, errors)
    final = json.loads((tmp_path / "state.json").read_text())
    assert final["state"] == "error" and "stalled" in final["error"]
    assert has_state(stub_dream, "frozen")


def test_freeze_as_end_home_finishes_is_not_lost(stub_dream, tmp_path, ctl, fast_poll):
    stub_dream.finish = False
    stub_dream.on_home_done = lambda n: freeze(ctl) if n == 2 else None  # call 2: the post-loop end move
    snap: dict = {}

    def driver():
        wait_for(lambda: has_state(stub_dream, "running"))
        end()
        wait_for(lambda: has_state(stub_dream, "frozen"))
        time.sleep(0.2)
        snap["order"] = list(stub_dream.order)
        end()
    t, errors = in_thread(driver)
    run(stub_dream, tmp_path, ctl)
    finish_driver(t, errors)
    assert "state:parking" not in snap["order"] and "torque_off" not in snap["order"]


# ---------------------------------------------------------------------------------------------------------------
# Final-review fixes.

def dedup_states(world) -> list[str]:
    names = [s["state"] for s in world.states]
    return [n for i, n in enumerate(names) if i == 0 or n != names[i - 1]]


def test_state_sequence_live_run(stub_dream, tmp_path):
    """`starting` means no arm touched: homing is written just before open_arm, running after the start-up home."""
    run(stub_dream, tmp_path)
    order = stub_dream.order
    assert order.index("state:homing") < order.index("open_arm")
    first_running = order.index("state:running")
    assert order.index("home") < first_running  # the start-up home came first
    running = next(s for s in stub_dream.states if s["state"] == "running")
    assert running["stage"] is None


def test_state_sequence_dry_run(stub_dream, tmp_path):
    run(stub_dream, tmp_path, dry_run=True)
    order = stub_dream.order
    assert order.index("state:running") < order.index("open_arm")
    assert "home" not in order
    assert next(s for s in stub_dream.states if s["state"] == "running")["stage"] is None
    assert dedup_states(stub_dream)[0] == "running"


def test_freeze_during_slow_probe_reaches_frozen_quickly(stub_dream, tmp_path, ctl, fast_poll):
    release = threading.Event()
    stub_dream.probe = lambda: (release.wait(8), None)[1]
    snap: dict = {}

    def driver():
        wait_for(lambda: has_state(stub_dream, "running"))
        time.sleep(0.1)
        t0 = time.monotonic()
        freeze(ctl)
        wait_for(lambda: has_state(stub_dream, "frozen"), timeout=2.0)
        snap["dt"] = time.monotonic() - t0
        release.set()  # let the abandoned probe thread go
        end()
    t, errors = in_thread(driver)
    run(stub_dream, tmp_path, ctl)
    release.set()
    finish_driver(t, errors)
    assert snap["dt"] < 2.0
    assert "loop_exit" not in stub_dream.calls  # the control loop never started


def test_exception_while_frozen_holds_until_end(stub_dream, tmp_path, ctl, fast_poll):
    stub_dream.finish = False
    stub_dream.exit_error = sys.modules["robots.errors"].ArmError("motor 3 overheated")
    snap: dict = {}

    def driver():
        wait_for(lambda: has_state(stub_dream, "running"))
        time.sleep(0.1)
        freeze(ctl)
        wait_for(lambda: has_state(stub_dream, "frozen"), timeout=2.0)
        time.sleep(0.3)
        snap["order"] = list(stub_dream.order)
        snap["frozen"] = next(s for s in stub_dream.states if s["state"] == "frozen")
        end()
    t, errors = in_thread(driver)
    with pytest.raises(sys.modules["robots.errors"].ArmError, match="overheated"):
        run(stub_dream, tmp_path, ctl)
    finish_driver(t, errors)
    before = snap["order"]
    assert "state:parking" not in before and "torque_off" not in before and before.count("home") == 1
    assert "overheated" in snap["frozen"]["alarm"]
    final = json.loads((tmp_path / "state.json").read_text())
    assert final["state"] == "error" and "overheated" in final["error"]
    assert stub_dream.calls.count("torque_off") == 2  # parked only after the end request


def test_freeze_during_first_frame(stub_dream, tmp_path, ctl, fast_poll):
    stub_dream.no_frame = True
    stub_dream.on_reader_start = lambda: freeze(ctl)

    def driver():
        wait_for(lambda: has_state(stub_dream, "frozen"), timeout=3.0)
        end()
    t, errors = in_thread(driver)
    run(stub_dream, tmp_path, ctl)
    finish_driver(t, errors)
    assert stub_dream.states[-1]["state"] == "done"  # a freeze is never an error ("no frame" after 10 s would be)


def test_freeze_with_failed_home_result_is_still_a_freeze(stub_dream, tmp_path, ctl, fast_poll):
    stub_dream.home_result = False  # the bool says "did not reach home" because the freeze interrupted it
    stub_dream.on_home_done = lambda n: freeze(ctl) if n == 1 else None

    def driver():
        wait_for(lambda: has_state(stub_dream, "frozen"))
        end()
    t, errors = in_thread(driver)
    run(stub_dream, tmp_path, ctl)
    finish_driver(t, errors)
    assert not has_state(stub_dream, "error")


def test_end_during_hold_is_dropped_until_the_hold_is_in_place(stub_dream, tmp_path, ctl, fast_poll):
    stub_dream.finish = False
    sent: list[int] = []

    def on_write():
        if not sent:  # the first hold write: an end signal lands before `frozen` is written
            sent.append(1)
            end()
            ctl.end_requested.wait(5)
    stub_dream.on_write = on_write
    snap: dict = {}

    def driver():
        wait_for(lambda: has_state(stub_dream, "running"))
        freeze(ctl)
        wait_for(lambda: has_state(stub_dream, "frozen"))
        time.sleep(0.4)
        snap["order"] = list(stub_dream.order)
        end()
    t, errors = in_thread(driver)
    run(stub_dream, tmp_path, ctl)
    finish_driver(t, errors)
    assert sent
    assert "state:parking" not in snap["order"] and snap["order"].count("home") == 1  # still frozen at +0.4 s


@pytest.mark.parametrize("how", ["sigint", "stop"])
def test_end_before_arms_open_aborts_cleanly(stub_dream, tmp_path, ctl, how):
    def on_open_client():
        if how == "sigint":
            end()
            ctl.end_requested.wait(5)
        else:
            ctl.request_end()
    stub_dream.on_open_client = on_open_client
    run(stub_dream, tmp_path, ctl)
    calls = stub_dream.calls
    assert "open_arm" not in calls and "home" not in calls and "torque_off" not in calls
    assert calls[-1] == "lock_released"
    final = json.loads((tmp_path / "state.json").read_text())
    assert final["state"] == "done"


def test_freeze_with_camera_error_does_no_camera_work(stub_dream, tmp_path, ctl, fast_poll, monkeypatch):
    stub_dream.finish = False
    stub_dream.loop_error = sys.modules["robots.errors"].CameraError("usb dropped")

    def log(self, t, what):
        if "camera dropped" in what:
            freeze(ctl)
    monkeypatch.setattr(StubRelay, "log", log)

    def driver():
        wait_for(lambda: has_state(stub_dream, "frozen"))
        end()
    t, errors = in_thread(driver)
    run(stub_dream, tmp_path, ctl)
    finish_driver(t, errors)
    assert stub_dream.camera_opens == 1  # the initial open only: no reopen after the freeze
    assert stub_dream.states[-1]["state"] == "done"


def test_config_load_failure_releases_lock_and_handlers(stub_dream, tmp_path, monkeypatch):
    def boom(path):
        raise RuntimeError("bad station.yaml")
    monkeypatch.setattr(sys.modules["robots.config"], "load", boom)
    before = {n: signal.getsignal(n) for n in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    with pytest.raises(RuntimeError, match="bad station.yaml"):
        run(stub_dream, tmp_path)
    assert "lock_released" in stub_dream.calls
    assert {n: signal.getsignal(n) for n in before} == before
    final = json.loads((tmp_path / "state.json").read_text())
    assert final["state"] == "error" and "bad station.yaml" in final["error"]
    assert stub_dream.order.index("state:error") < stub_dream.order.index("lock_released")


def test_terminal_state_is_written_before_the_lock_is_released(stub_dream, tmp_path):
    run(stub_dream, tmp_path)
    assert stub_dream.order.index("state:done") < stub_dream.order.index("lock_released")


def test_freeze_during_the_final_park_is_recorded_in_error_not_alarm(stub_dream, tmp_path, ctl, monkeypatch):
    logs: list[str] = []
    monkeypatch.setattr(StubRelay, "log", lambda self, t, what: logs.append(what))
    sent: list[bool] = []

    def torque_off(self) -> None:
        self.world.calls.append("torque_off")
        if not sent:
            sent.append(True)
            freeze(ctl)  # a freeze arrives mid-park: the park cannot be stopped

    monkeypatch.setattr(StubArm, "torque_off", torque_off)
    run(stub_dream, tmp_path, ctl)
    assert stub_dream.calls.count("torque_off") == 2  # the park still ran to the end
    final = stub_dream.states[-1]
    assert final["state"] == "done" and "freeze arrived during the park" in (final["error"] or "")
    assert not final.get("alarm") and not any(st.get("alarm") for st in stub_dream.states)
    assert any("freeze arrived during the park" in m for m in logs)


def test_debugview_any_exception_is_a_warning(stub_dream, tmp_path, caplog):
    stub_dream.view_error = RuntimeError("bad init")
    with caplog.at_level(logging.WARNING):
        run(stub_dream, tmp_path)
    assert "open_arm" in stub_dream.calls and stub_dream.states[-1]["state"] == "done"
    assert any("debug view" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------------------------------------------
# Round 3: freezing, early aborts, probe thread, reasons, link restore, eight_ball_alert.

def test_freeze_writes_freezing_before_frozen(stub_dream, tmp_path, ctl, fast_poll):
    stub_dream.finish = False

    def driver():
        wait_for(lambda: has_state(stub_dream, "running"))
        time.sleep(0.1)
        freeze(ctl)
        wait_for(lambda: has_state(stub_dream, "frozen"))
        end()
    t, errors = in_thread(driver)
    run(stub_dream, tmp_path, ctl)
    finish_driver(t, errors)
    order = stub_dream.order
    assert "state:freezing" in order
    assert order.index("state:running") < order.index("state:freezing") < order.index("state:frozen")
    assert [s for s in stub_dream.states if s["state"] == "freezing"][0]["stage"] == "unplug"


def test_freeze_while_already_frozen_does_not_leave_the_state_at_freezing(stub_dream, tmp_path, ctl, fast_poll):
    stub_dream.finish = False
    snap: dict = {}

    def driver():
        wait_for(lambda: has_state(stub_dream, "running"))
        freeze(ctl)
        wait_for(lambda: has_state(stub_dream, "frozen"))
        n = len(stub_dream.states)
        freeze(ctl)  # a second freeze while the hold is in place
        time.sleep(0.3)
        snap["after"] = [s["state"] for s in stub_dream.states[n:]]
        snap["now"] = stub_dream.states[-1]["state"]
        end()
    t, errors = in_thread(driver)
    run(stub_dream, tmp_path, ctl, )
    finish_driver(t, errors)
    assert snap["now"] == "frozen" and "freezing" not in snap["after"]


def test_freeze_before_the_arms_open_has_a_reason(stub_dream, tmp_path, ctl):
    stub_dream.on_open_client = lambda: freeze(ctl)
    run(stub_dream, tmp_path, ctl)
    final = stub_dream.states[-1]
    assert final["state"] == "done" and final["reason"] == "frozen_before_open"


def test_end_before_the_arms_open_has_a_reason(stub_dream, tmp_path, ctl):
    stub_dream.on_open_client = ctl.request_end
    run(stub_dream, tmp_path, ctl)
    final = stub_dream.states[-1]
    assert final["state"] == "done" and final["reason"] == "ended_before_open"


def test_normal_run_has_no_reason(stub_dream, tmp_path):
    run(stub_dream, tmp_path)
    assert stub_dream.states[-1].get("reason") is None


@pytest.mark.parametrize("how", ["freeze", "end"])
def test_freeze_or_end_before_the_cameras_open_does_no_camera_or_client_work(stub_dream, tmp_path, ctl, monkeypatch,
                                                                             how):
    def load(path):
        freeze(ctl) if how == "freeze" else ctl.request_end()
        return SimpleNamespace(follower_l=None, follower_r=None, inference_log_path=None)
    monkeypatch.setattr(sys.modules["robots.config"], "load", load)
    run(stub_dream, tmp_path, ctl)
    assert stub_dream.camera_opens == 0 and "open_client" not in stub_dream.calls
    assert stub_dream.ik_calls == 0 and "open_arm" not in stub_dream.calls
    final = stub_dream.states[-1]
    assert final["state"] == "done" and final["reason"] == f"{'frozen' if how == 'freeze' else 'ended'}_before_open"


def test_freeze_before_the_ik_warmup_skips_it(stub_dream, tmp_path, ctl):
    stub_dream.on_open_client = lambda: freeze(ctl)  # lands inside open_client, i.e. before the warm-up
    run(stub_dream, tmp_path, ctl)
    assert stub_dream.ik_calls == 0


def test_lock_refusal_restores_the_previous_run_link(stub_dream, tmp_path):
    link = tmp_path / "link"
    prev = tmp_path / "prev-run"
    prev.mkdir()
    StateFile(prev, link=link).link_current()
    stub_dream.lock_error = RuntimeError("the arms are held by 1234 harness run")
    state = RecordingState(tmp_path / "second", stub_dream)
    state.link = link
    state.link_current()
    assert link.resolve() == (tmp_path / "second").resolve()
    coro = dream_executor.run_match(MatchConfig(own_color="white"), "ckpt", dry_run=False,
                                    out_dir=tmp_path / "second", ctl=FreezeControl(), state=state)
    with pytest.raises(RuntimeError, match="held by 1234"):
        asyncio.run(coro)
    assert link.resolve() == prev.resolve()  # the live run's link, not ours
    assert json.loads((tmp_path / "second" / "state.json").read_text())["state"] == "error"


def test_claude_probe_runs_in_a_daemon_thread_and_is_bounded(stub_dream, tmp_path, monkeypatch):
    release = threading.Event()
    seen: dict = {}

    def probe():
        t = threading.current_thread()
        seen["name"], seen["daemon"] = t.name, t.daemon
        release.wait(10)
    stub_dream.probe = probe
    claude = SimpleNamespace(probe=lambda: stub_dream.probe(), available=True)
    monkeypatch.setattr(dream_executor, "ClaudeVerifier", lambda: claude)
    monkeypatch.setattr(dream_executor, "PROBE_TIMEOUT_S", 0.3)
    t0 = time.monotonic()
    try:
        run(stub_dream, tmp_path)
    finally:
        release.set()
    assert time.monotonic() - t0 < 3.0  # did not wait out the 10 s probe
    assert seen["daemon"] is True and seen["name"] == "claude-probe"
    assert claude.available is False  # a probe that never answers: Claude checks are off
    assert stub_dream.states[-1]["state"] == "done"


def test_end_during_the_probe_does_not_wait_it_out(stub_dream, tmp_path, ctl, monkeypatch):
    release = threading.Event()
    stub_dream.probe = lambda: release.wait(10)
    snap: dict = {}

    def driver():
        wait_for(lambda: has_state(stub_dream, "running"))
        time.sleep(0.1)
        snap["t0"] = time.monotonic()
        end()
    t, errors = in_thread(driver)
    try:
        run(stub_dream, tmp_path, ctl)
    finally:
        release.set()
    finish_driver(t, errors)
    assert time.monotonic() - snap["t0"] < 3.0
    assert "loop_exit" not in stub_dream.calls  # the control loop never started
    assert stub_dream.states[-1]["state"] == "done"


def test_freeze_during_the_park_has_a_reason(stub_dream, tmp_path, ctl, monkeypatch):
    sent: list[bool] = []

    def torque_off(self) -> None:
        self.world.calls.append("torque_off")
        if not sent:
            sent.append(True)
            freeze(ctl)

    monkeypatch.setattr(StubArm, "torque_off", torque_off)
    run(stub_dream, tmp_path, ctl)
    assert stub_dream.states[-1]["reason"] == "freeze_during_park"
    assert "freezing" not in [s["state"] for s in stub_dream.states]


def test_real_supervise_writes_eight_ball_alert(tmp_path, monkeypatch):
    monkeypatch.setattr(dream_executor, "POLL_S", 0.001)
    monkeypatch.setattr(dream_executor.vision, "detect", lambda *a: None)
    monkeypatch.setattr(dream_executor.vision, "draw", lambda *a: None)
    monkeypatch.setattr(dream_executor.cv2, "imwrite", lambda *a: True)
    frame = SimpleNamespace(data=None, timestamp_s=0.0)
    arm = SimpleNamespace(gripper=0.0)
    loop = SimpleNamespace(video={"scene": frame}, state=SimpleNamespace(left=arm, right=arm), task="t",
                           set_task=lambda t: None)
    outcomes = iter([dream_executor.Outcome.PASSED])
    relay = SimpleNamespace(stack_start_rgb=None, stage="balls", confirm_kinds=set(), pending_confirm=None, step=None,
                            observe=lambda *a: next(outcomes, dream_executor.Outcome.RUNNING),
                            next_step=lambda world, now: SimpleNamespace(instruction="t"), eight_ball_alert=False,
                            log=lambda *a: None, score=lambda: 0, passed={})
    claude = SimpleNamespace(available=True, ask=lambda question, *a: (True, "a black ball in the basket"))
    statefile = StateFile(tmp_path, link=tmp_path / "link")
    stop = asyncio.Event()

    async def go():
        task = asyncio.create_task(dream_executor._supervise(
            loop, relay, SimpleNamespace(calibration=None, all_colors=False, stack_pass="6in"), stop, tmp_path, claude, statefile=statefile))
        await asyncio.sleep(0.2)
        stop.set()
        await task

    asyncio.run(go())
    assert relay.eight_ball_alert is True
    assert json.loads((tmp_path / "state.json").read_text())["eight_ball_alert"] is True


def test_freeze_while_starting_never_writes_freezing(stub_dream, tmp_path, ctl):
    """No arm is open yet: state stays `starting` until the pre-open abort writes `done` (frozen_before_open)."""
    def slow_open():
        freeze(ctl)
        time.sleep(0.2)  # the loop thread handles the wake while open_client is still running
    stub_dream.on_open_client = slow_open
    state = RecordingState(tmp_path, stub_dream)
    state.update(state="starting")  # what the CLI writes before run_match
    asyncio.run(dream_executor.run_match(MatchConfig(own_color="white"), "ckpt", dry_run=False, out_dir=tmp_path,
                                         ctl=ctl, state=state))
    seen = [s["state"] for s in stub_dream.states]
    assert "freezing" not in seen
    assert seen[-1] == "done" and stub_dream.states[-1]["reason"] == "frozen_before_open"
