"""Runs the relay on the real station with Dream-1: one control loop, the instruction switched between steps.

This mirrors `dreamcli robot run` (`robots.run.main`): open the cameras, the Dream client and the arms the same way,
home, then run Dream's own `ControlLoop`. Beside it runs our supervisor, which reads the loop's newest overhead frame
and arm state twice a second, feeds them to the relay, and calls `loop.set_task()` whenever the relay moves to the
next instruction, so the arms never go home between steps. When the relay is done, the clock runs out, or Ctrl-C is
pressed, the loop stops the way `dreamcli` stops it: the arms hold, move home and switch off.

A dry run (`dry_run=True`) opens the arms read-only and writes no target to them, exactly as `dreamcli ... --dry-run`.
Nothing moves, so steps time out; it checks the wiring, the camera calibration and the instruction sequence.

Only this module imports Dream's `robots` package, so the relay and vision run (and are tested) without it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import pathlib
import signal
import threading
import time

import cv2

from harness.claude_check import ClaudeVerifier
from harness.config import MatchConfig
from harness.freeze import FreezeControl, FreezeRequested, first_alarm, hold_measured
from harness.relay import Outcome, Relay
from harness import augment, vision

_log = logging.getLogger(__name__)

STATION_CONFIG = pathlib.Path("~/.dreamlabs/station.yaml").expanduser()
OVERHEAD = "scene"  # the overhead camera's name in the station config
POLL_S = 0.5
MAX_RESTARTS = 8  # camera drops and Dream stalls per run (a wrist camera dropped 3 times in 10 s mid-stack)
FROZEN_POLL_S = 0.5  # how often the frozen wait looks for an end request
ALARM_POLL_S = 1.0  # ... and for an arm alarm
PROBE_TIMEOUT_S = 5.0  # the most any path (start, freeze, end) waits for the Claude probe
VIEW_PORT = 8093  # Dream's DebugView: what the model sees (the page `dreamcli robot run --view` serves)


def _set_state(statefile, /, **fields) -> None:
    """Writes state.json when there is one; a failed write is logged, never allowed to break the run or its teardown."""
    if statefile is None:
        return
    try:
        statefile.update(**fields)
    except Exception:  # noqa: BLE001
        _log.exception("could not write state.json")


async def run_match(match: MatchConfig, checkpoint: str, *, dry_run: bool, out_dir: pathlib.Path,
                    station_config: pathlib.Path = STATION_CONFIG, augment_spec: str | None = None,
                    augment_detector: str = "both", ctl=None, state=None) -> Relay:
    from robots import config as rconfig
    from robots import control, followers, home, run as rrun
    from robots import bodies
    from robots import errors as rerrors
    from robots import arms_lock
    from robots import types as rtypes
    import dream

    if ctl is None:
        ctl = FreezeControl()

    out_dir.mkdir(parents=True, exist_ok=True)
    # One holder of the arms at a time, as `dreamcli robot run` does; released last, after the arms switch off.
    lock = contextlib.ExitStack()
    try:
        lock.enter_context(arms_lock.holding(pathlib.Path("~/.dreamlabs/arms.lock").expanduser(), "harness run"))
    except Exception as exc:  # noqa: BLE001 - held by dreamcli or the station agent: nothing is opened
        _set_state(state, state="error", error=str(exc))
        if state is not None:  # the live holder's run is the current one, not this refused start
            try:
                state.restore_link()
            except Exception:  # noqa: BLE001
                _log.exception("could not restore the current-run link")
        raise
    stop = asyncio.Event()
    aloop = asyncio.get_running_loop()

    def _woken() -> None:
        """Runs on the loop thread, never in the signal handler: wake the loop, and say `freezing` until `frozen`."""
        stop.set()
        # Not while starting (no arm is open: the run aborts to `done`, reason frozen_before_open, and the state
        # stays `starting` until then), nor while frozen/parking/done: a repeat freeze changes nothing.
        if state is not None and state.data.get("state") in ("running", "homing"):
            _set_state(state, state="freezing")

    ctl.on_freeze = lambda: aloop.call_soon_threadsafe(_woken)  # the handler itself does no arm or file work
    # The end signals use signal.signal like SIGUSR1 (not loop.add_signal_handler, whose callback is deferred until
    # the loop runs): both run on the main thread as soon as it is free. Pending handlers run in signal-number
    # order, not arrival order, so a SIGTERM racing a freeze while already frozen can still end the run (our agent
    # sends SIGINT). _frozen_wait drops any end that landed before its hold was in place.
    def _on_end_signal(signum, frame) -> None:
        ctl.request_end()  # first: the event is reentrant and needs no loop
        try:
            aloop.call_soon_threadsafe(stop.set)
        except RuntimeError:  # the loop is closed
            pass

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        lock.callback(signal.signal, signum, signal.signal(signum, _on_end_signal))  # restored as the lock closes

    relay = Relay(match)

    def model_for(stage: str) -> str:
        return match.stage_models.get(stage) or checkpoint

    models = list(dict.fromkeys(model_for(s) for s in relay.m_stages))  # each distinct model once, in stage order
    arms: dict = {}
    readers: list = []
    view = None
    error: str | None = None
    done_error: str | None = None  # a clean abort: written with state "done"
    done_reason: str | None = None  # ... and its machine-readable reason

    def abort_before_open() -> bool:
        """A freeze or an end that arrived before any arm opened: nothing to hold, home or park."""
        nonlocal done_error, done_reason
        if ctl.frozen:
            done_error, done_reason = "frozen before the arms opened", "frozen_before_open"
        elif ctl.end_requested.is_set() or stop.is_set():
            done_error, done_reason = "ended before the arms opened", "ended_before_open"
        else:
            return False
        relay.log(time.monotonic(), done_error)
        return True

    fault: Exception | None = None
    try:
        app = rrun.with_task(rconfig.load(station_config), "hold the arms still")
        body = bodies.for_config(app)
        if abort_before_open():  # checked before each slow step: no cameras, client or IK work for a dead run
            return relay
        readers = await asyncio.to_thread(rrun.make_cameras, app)
        kin = rrun.make_kinematics(app)
        if abort_before_open():
            return relay

        async def open_model(model: str):
            """A warmed Dream client for one model: opened before go for every stage's model, so a stage change
            only swaps the client (about a second) instead of opening and compiling one mid-match."""
            c = await asyncio.to_thread(rrun.open_client, rrun.policy_config(app, model))
            c = _augmented(c, match, augment_spec, augment_detector, out_dir)
            if ctl.frozen or ctl.end_requested.is_set() or stop.is_set():
                return c  # a freeze or an end landed while it opened: no IK work; the caller stops
            warmup = getattr(rrun, "TRAJ_WARMUP_S", None)  # only some dreamlabs versions compile the IK up front
            if warmup is not None and hasattr(c, "wait_for_ik"):
                if not await asyncio.to_thread(c.wait_for_ik, warmup):
                    _log.warning("%s: the IK's traj solve did not compile in time; the arms open anyway", model)
            return c

        # Later stages' models go cold while earlier stages run (a cold one took up to 12 min to start): warm each on
        # Dream's servers in the background now, by replaying the saved observation (harness/cloud_warm.py: no
        # arms, its answers thrown away). Read before this run's loop overwrites the saved inputs.
        if not dry_run:
            _warm_later_stages(relay, model_for, match)
        clients = {}
        for m in models:
            clients[m] = await open_model(m)
            if abort_before_open():  # checked after each slow open: a dead run opens no more models
                return relay
        _log.info("models: %s", {s: model_for(s) for s in relay.m_stages})
        view = _debug_view(dream, clients[model_for(match.start_stage)])
        if abort_before_open():  # a freeze or end while the harness was starting: the arms are never opened
            return relay
        open_arm = body.open_arm_read_only if dry_run and body.open_arm_read_only is not None else body.open_arm
        # `starting` means no arm has been touched: say so before the first open.
        if dry_run:
            _set_state(state, state="running", stage=None)
        else:
            _set_state(state, state="homing")
        for name, arm_cfg in (("left", app.follower_l), ("right", app.follower_r)):
            arm = open_arm(arm_cfg, name)
            arms[name] = followers.FollowerArm(name, rrun.DryRun(arm), home_rad=arm.home_rad) if dry_run else arm
        if ctl.frozen:
            raise FreezeRequested  # opened but not homed: hold where they are
        if not dry_run:
            homed = await asyncio.to_thread(home.return_to_home, arms, duration_s=home.HOME_DURATION_S,
                                            sleep=ctl.interruptible_sleep)
            if ctl.frozen:
                raise FreezeRequested  # a freeze also makes the move report "not home": that is not a fault
            if not homed:
                raise RuntimeError("an arm did not reach home before the run")
            _set_state(state, state="running", stage=None)
        for reader in readers:
            reader.start()
        for reader in readers:
            reader.set_exposure_hold(True)
        overhead = next(r for r in readers if r.name == OVERHEAD)
        first = await _first_frame(overhead, ctl=ctl)
        if ctl.frozen:
            raise FreezeRequested
        # Find the bin and the strip where they stand today (the judges place them), then plan from that.
        layout = vision.locate_layout(first.data, match.calibration)
        match.calibration = vision.with_layout(match.calibration, layout)
        relay.cal = match.calibration
        _log.info("layout at go: %s; bin %s, hub %s", layout.note(), match.calibration.bin_roi,
                  match.calibration.hub_roi)
        world = vision.detect(first.data, match.calibration, first.timestamp_s)
        cv2.imwrite(str(out_dir / "go.jpg"), vision.draw(first.data, world))
        t0 = time.monotonic()
        relay.start(world, t0)
        step = relay.next_step(world, t0)
        if step is None:
            raise RuntimeError("the relay has nothing to do at go")
        settings = rrun.loop_settings(app)
        # Dream-1 answered in up to ~4.3 s in practice: don't throw such a chunk away, and ride out a 20 s gap.
        # Slow answers (6+ s seen at peak) beat a frozen arm: with gravity compensation lowered, a frozen arm sags onto
        # the table, its joint reads as blocked and the arm's guard stops the run.
        settings["action_timeout_s"] = max(settings.get("action_timeout_s") or 0, 8.0)
        settings["max_stall_ticks"] = max(settings.get("max_stall_ticks") or 0, 400)
        restarts = 0
        _set_state(state, state="running", stage=relay.stage)
        claude = ClaudeVerifier()
        problem = await _probe_claude(claude, stop, PROBE_TIMEOUT_S)  # a freeze or end abandons it; at most ~5 s
        if ctl.frozen:
            raise FreezeRequested  # frozen during the probe: abandoned, the loop never starts
        if problem:
            _log.warning("Claude vision checks off: %s", problem)
        if ctl.frozen:
            raise FreezeRequested
        def halted() -> bool:  # a freeze flips the flag at once; stop follows when the loop next runs
            return stop.is_set() or ctl.frozen

        current = model_for(step.stage)
        while not halted():
            task = relay.step.instruction if relay.step else step.instruction
            if relay.step is not None and model_for(relay.step.stage) != current:
                relay.log(time.monotonic(), f"stage {relay.step.stage}: switching model {current} -> "
                                            f"{model_for(relay.step.stage)}")
                current = model_for(relay.step.stage)
                _watch(view, clients[current])  # the :8093 page follows the new stage's model
            stage_now = relay.step.stage if relay.step is not None else step.stage
            if match.stage_gravity.get(stage_now) is not None:
                from harness import gravity
                if gravity.scale() != match.stage_gravity[stage_now]:
                    relay.log(time.monotonic(), f"gravity compensation x{match.stage_gravity[stage_now]:g} for {stage_now}")
                gravity.set_scale(match.stage_gravity[stage_now])
            loop = control.ControlLoop(followers=arms, cameras=readers, client=clients[current], task=task,
                                       kinematics=kin, inference_log_path=app.inference_log_path, camera_owner=False,
                                       recorder=None, **settings)
            try:
                async with loop:
                    runner = asyncio.create_task(loop.run(), name="dream-loop")
                    supervisor = asyncio.create_task(
                        _supervise(loop, relay, match, stop, out_dir, claude, go_rgb=first.data,
                                   model_for=model_for, current=current, statefile=state), name="relay")
                    stopped = asyncio.create_task(stop.wait(), name="stop")
                    await asyncio.wait({runner, supervisor, stopped}, return_when=asyncio.FIRST_COMPLETED)
                    for t in (supervisor, stopped):
                        t.cancel()
                    if runner.done():
                        runner.result()
                    runner.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await runner
                    result = supervisor.result() if supervisor.done() and not supervisor.cancelled() \
                        and supervisor.exception() is None else None
                    switch = result == "switch"
                if result == "release":
                    # Claude sees the cubes stacked, with the gripper still on them. Stop the model, send the arms
                    # home (grippers open, away from the tower), then ask whether it stands on its own.
                    t = time.monotonic()
                    if not dry_run:
                        relay.log(t, "stacked: arms home to let go of the tower")
                        await asyncio.to_thread(home.return_to_home, arms, duration_s=home.HOME_DURATION_S,
                                                sleep=ctl.interruptible_sleep)
                    if ctl.frozen:
                        raise FreezeRequested
                    await asyncio.sleep(1.0)  # let the tower settle and the camera catch up
                    after = next(r for r in readers if r.name == OVERHEAD).frame  # readers may be reopened
                    t = time.monotonic()
                    world = vision.detect(after.data, match.calibration, after.timestamp_s)
                    wrist = {r.name: r.frame for r in readers if r.name != OVERHEAD}
                    yes, why = await asyncio.to_thread(claude.ask, "standing_now", after.data, relay.stack_start_rgb,
                                                       _wrists(wrist.get))
                    cv2.imwrite(str(out_dir / f"tower_check_{'standing' if yes else 'not'}.jpg"),
                                cv2.cvtColor(after.data, cv2.COLOR_RGB2BGR))
                    if yes and relay.tower_standing_pass(world, t, why):
                        nxt = relay.next_step(world, t)
                        if nxt is None:
                            stop.set()
                    else:
                        relay.resolve_confirm(False, f"after the arms moved away: {why}", t)
                    continue
                if switch:
                    # The next stage runs on another model: a new loop on its client (not a restart). Both arms go
                    # home first: every demo started there, and an arm left resting on the strip after a pull (it
                    # sags low with gravity compensation off) blocked a joint and tripped the safety stop.
                    if not dry_run:
                        relay.log(time.monotonic(), "arms home before the next stage's model")
                        homed = await asyncio.to_thread(home.return_to_home, arms, duration_s=home.HOME_DURATION_S,
                                                        sleep=ctl.interruptible_sleep)
                        if ctl.frozen:
                            raise FreezeRequested  # a freeze also makes the move report "not home"
                        if not homed:
                            relay.log(time.monotonic(), "an arm did not reach home; starting the next stage anyway")
                    continue
            except FreezeRequested:
                raise  # a freeze from a home move in here: hold, never treat it as a stall and restart
            except (rerrors.ArmError, rerrors.SafetyStop):
                raise  # a real arm fault or safety stop ends the run (the arms go home in the finally below)
            except rerrors.CameraError as exc:
                # A wrist camera's USB link dropped (it re-enumerates a second or two later). The dead reader stays
                # dead, so reopening only the loop fails at once: close every reader and open them again.
                restarts += 1
                left = relay.time_left(time.monotonic())
                relay.log(time.monotonic(), f"camera dropped ({exc}); reopening cameras, restart {restarts} "
                                            f"with {left:.0f} s left")
                if restarts > MAX_RESTARTS or left < match.stop_margin_s + 15:
                    fault = exc
                    break
                if halted():  # a freeze or an end arrived: no camera work now
                    break
                readers = await _reopen_cameras(rrun, app, readers, deadline_s=min(12.0, left - match.stop_margin_s))
                if readers is None:
                    fault = exc
                    readers = []
                    break
                # The old loop closed its client's policy when it stopped: reusing that client failed at once with
                # "no policy is open" and spent a second restart. Open a fresh one, as the stall path does.
                if halted():
                    break
                clients[current] = await open_model(current)
                _watch(view, clients[current])
                continue
            except Exception as exc:  # noqa: BLE001 - a stall, or a fault inside Dream's client (an IK hiccup)
                left = relay.time_left(time.monotonic())
                if "unsafe instruction in this scene" in str(exc):
                    # Dream-1's own safety check: a person is in the scene (a hand on the cable, say). Honour it:
                    # the arms hold, and we wait for the person to step back; every new loop is checked again
                    # by Dream's service, so this only resumes once the scene is clear.
                    relay.log(time.monotonic(), f"Dream refused while a person is in the scene ({exc}); "
                                                f"holding, retry in 3 s with {left:.0f} s left")
                    if left < match.stop_margin_s + 5:
                        break
                    with contextlib.suppress(TimeoutError):  # stop-aware: a freeze must not wait out the pause
                        await asyncio.wait_for(stop.wait(), 3.0)
                    continue
                # Dream-1 stopped answering for a while (the loop's stall guard). The arms hold; while the match
                # clock has time left, open a fresh loop and carry on from the same step.
                restarts += 1
                relay.log(time.monotonic(), f"Dream loop stopped ({exc}); restart {restarts} with {left:.0f} s left")
                if restarts > MAX_RESTARTS or left < match.stop_margin_s + 10:
                    fault = exc
                    break
                if halted():
                    break  # a freeze or an end arrived: no restart work now
                # A fresh client, its IK warmed first as dreamcli does before the arms move: skipping it crashed the
                # first answer of a restarted loop (SVD did not converge).
                clients[current] = await open_model(current)
                _watch(view, clients[current])  # keep the :8093 page following the new client
                continue
            break
        if ctl.frozen:
            raise FreezeRequested  # the loop has exited and the arms hold: wait for the end, never go home
        _set_state(state, state="homing")
        if not dry_run:
            await asyncio.to_thread(home.return_to_home, arms, duration_s=home.HOME_DURATION_S,
                                    sleep=ctl.interruptible_sleep)
        if ctl.frozen:
            raise FreezeRequested  # a freeze landed as the move finished
        if fault is not None:
            raise fault
    except FreezeRequested:
        # Never an error: hold the arms, wait for an end request, then home. A freeze must not reach the caller.
        relay.log(time.monotonic(), f"frozen at stage {relay.stage}")
        try:
            await _frozen_wait(arms, ctl, state, home, rtypes, dry_run)
            if fault is not None:
                raise fault  # the loop's fault is still the reason this run ends
        except BaseException as exc:
            error = str(exc) or type(exc).__name__
            raise
    except Exception as exc:
        error = str(exc) or type(exc).__name__
        if ctl.frozen and arms:
            # A genuine fault while frozen: parking now would drop the arms. Record it, hold, and re-raise only
            # after an explicit end request (the finally below then parks).
            fault = exc
            _log.error("fault while frozen: %s", error)
            _set_state(state, alarm=f"{type(exc).__name__}: {exc}", fault=error)
            try:
                await _frozen_wait(arms, ctl, state, home, rtypes, dry_run)
            except BaseException as exc2:
                error = str(exc2) or type(exc2).__name__
                raise
        raise
    except BaseException as exc:
        error = str(exc) or type(exc).__name__
        raise
    finally:
        if arms:
            _set_state(state, state="parking")
        for reader in readers:
            reader.stop()
        if view is not None:
            with contextlib.suppress(Exception):
                view.close()
        for name, opened in arms.items():
            control.run_teardown(f"switch {name} arm off", opened.torque_off)
        if arms and ctl.frozen:  # a freeze arrived during the park, which cannot be stopped: say so
            msg = "freeze arrived during the park; the park cannot be stopped"
            relay.log(time.monotonic(), msg)
            if error is None and not done_error:  # never `alarm`: that reads as an arm fault after the arms are off
                done_error, done_reason = msg, "freeze_during_park"
        try:
            _write_report(relay, out_dir)
        except BaseException as exc:
            error = error or str(exc) or type(exc).__name__
            raise
        finally:
            try:  # the terminal state first: lock.close() releases the lock and restores the signal handlers
                if error is None:
                    try:
                        _set_state(state, state="done", score=int(relay.score()), error=done_error,
                                   reason=done_reason, end_requested=ctl.end_was_requested)
                    except Exception as exc:  # noqa: BLE001
                        _set_state(state, state="error", error=str(exc))
                else:
                    _set_state(state, state="error", error=error, end_requested=ctl.end_was_requested)
            finally:
                lock.close()
    return relay


def _augmented(client, match: MatchConfig, spec: str | None, detector: str, out_dir: pathlib.Path):
    """The Dream client, wrapped so the target ball is marked in what Dream sees (harness/augment.py) when asked."""
    wrapped = augment.wrap(client, match.calibration, spec, detector=detector, debug_dir=out_dir / "sent")
    if wrapped is not client:
        _log.info("augmenting Dream's frames: %s (detector %s); samples in %s/sent", spec, detector, out_dir)
    return wrapped


def _debug_view(dream, client):
    """Dream's DebugView (what the model sees) on :8093, or None when it cannot start. Given the raw client under
    any augment wrapper, so the view can never replace the wrapper's set_observation and skip the augmentation."""
    try:
        view = dream.DebugView(getattr(client, "_client", client), port=VIEW_PORT)
        _log.info("what the model sees: %s", view.url)
        return view
    except Exception as exc:  # noqa: BLE001 - a debug view must never kill a run
        _log.warning("debug view cannot start on port %d (%s); carrying on without it", VIEW_PORT, exc)
        return None


def _watch(view, client) -> None:
    """Points the debug view at another client (a model switch or a restart); never fails the run."""
    if view is not None:
        try:
            view.watch(getattr(client, "_client", client))
        except Exception:  # noqa: BLE001
            _log.exception("debug view could not follow the new client")


def _warm_later_stages(relay: Relay, model_for, match: MatchConfig) -> None:
    import threading
    from harness import cloud_warm

    start = relay.m_stages.index(match.start_stage) if match.start_stage in relay.m_stages else 0
    later = list(dict.fromkeys(model_for(st) for st in relay.m_stages[start + 1:]))
    later = [m for m in later if m != model_for(relay.m_stages[start])]
    if not later:
        return
    try:
        frames, state = cloud_warm.saved_observation()
    except BaseException as exc:  # noqa: BLE001 - no saved inputs yet: nothing to warm with
        _log.warning("background warm-up skipped: %s", exc)
        return
    out: dict[str, str] = {}
    for st in relay.m_stages[start + 1:]:
        m = model_for(st)
        if m in later:
            later.remove(m)
            text = match.stage_texts.get(st) or "hold the arms still"
            threading.Thread(target=cloud_warm.warm, args=(m, text, frames, state, out), daemon=True,
                             name=f"warm-{st}").start()
            _log.info("warming %s for the %s stage in the background", m, st)


async def _probe_claude(claude, stop: asyncio.Event, timeout_s: float) -> str | None:
    """Claude's pre-run probe on a daemon thread (an abandoned probe must never delay the process's exit).

    Returns why Claude can't be used, else None. Waits at most `timeout_s`, and gives up at once when `stop` is set
    (a freeze or an end); either way an unanswered probe turns the Claude checks off."""
    aloop = asyncio.get_running_loop()
    done = asyncio.Event()
    result: dict = {}

    def work() -> None:
        try:
            result["problem"] = claude.probe()
        except BaseException as exc:  # noqa: BLE001 - nothing may escape a daemon thread
            result["error"] = exc
        finally:
            with contextlib.suppress(RuntimeError):  # the loop may be closed by now
                aloop.call_soon_threadsafe(done.set)

    threading.Thread(target=work, name="claude-probe", daemon=True).start()
    waiting = [asyncio.create_task(done.wait()), asyncio.create_task(stop.wait())]
    try:
        await asyncio.wait(waiting, timeout=timeout_s, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for t in waiting:
            t.cancel()
    if not done.is_set():
        claude.available = False
        if not stop.is_set():
            _log.warning("the Claude probe gave no answer in %.0f s; vision checks are off", timeout_s)
        return None
    if "error" in result:
        claude.available = False
        return f"probe failed: {result['error']!r}"
    return result.get("problem")


async def _frozen_wait(arms: dict, ctl: FreezeControl, state, home, rtypes, dry_run: bool) -> None:
    """Holds the arms where they are until an end is requested, then homes. Returns once the end move is done.

    The arms are commanded their own measured pose once (the driver's hold tick keeps it from there); there is no
    motion and no switch-off here. A fault is written to state.json and the hold continues. A new freeze during
    the end home move comes back here and holds again.
    """
    make_command = lambda positions, gripper: rtypes.ArmCommand(positions_rad=positions, gripper=gripper)  # noqa: E731
    while True:
        failed = await asyncio.to_thread(hold_measured, arms, make_command)
        fields = {"alarm": "; ".join(f"{n}: could not hold ({e})" for n, e in failed.items())} if failed else {}
        ctl.end_requested.clear()  # an end that landed before the hold was in place is dropped (SIGTERM included)
        _set_state(state, state="frozen", **fields)  # only now: the control loop is closed and the arms hold
        _log.warning("frozen: arms hold the measured pose; waiting for an end request")
        last_alarm = fields.get("alarm")
        next_alarm = time.monotonic() + ALARM_POLL_S
        while not ctl.end_requested.is_set():
            await asyncio.sleep(FROZEN_POLL_S)
            if time.monotonic() >= next_alarm:
                next_alarm = time.monotonic() + ALARM_POLL_S
                try:
                    alarm = await asyncio.to_thread(first_alarm, arms)
                except Exception:  # noqa: BLE001 - a failed check must never drop the hold
                    _log.exception("alarm check failed")
                    alarm = None
                if alarm and alarm != last_alarm:
                    last_alarm = alarm
                    _log.error("alarm while frozen: %s", alarm)
                    _set_state(state, alarm=alarm)
        if not ctl.begin_end():
            continue  # a freeze arrived since the end request: keep holding (arms were not touched)
        _set_state(state, state="homing")
        try:
            if not dry_run:
                await asyncio.to_thread(home.return_to_home, arms, duration_s=home.HOME_DURATION_S,
                                        sleep=ctl.interruptible_sleep)
        except FreezeRequested:
            continue  # frozen again mid-move: hold the pose it reached
        if ctl.frozen:
            continue  # a freeze landed as the move finished
        return


async def _reopen_cameras(rrun, app, readers: list, deadline_s: float) -> list | None:
    """Closes the readers and opens them all again, retrying while the dropped camera comes back on USB.
    Returns the started readers, or None when a camera is still missing at the deadline."""
    for reader in readers:
        with contextlib.suppress(Exception):
            reader.stop()
    end = time.monotonic() + max(deadline_s, 2.0)
    while time.monotonic() < end:
        await asyncio.sleep(1.0)
        try:
            fresh = await asyncio.to_thread(rrun.make_cameras, app)
        except Exception as exc:  # noqa: BLE001 - the device node isn't back yet
            _log.info("cameras not back yet: %s", exc)
            continue
        for reader in fresh:
            reader.start()
        for reader in fresh:
            reader.set_exposure_hold(True)
        try:
            for reader in fresh:
                await _first_frame(reader, timeout_s=max(1.0, min(4.0, end - time.monotonic())))
            _log.info("cameras reopened")
            return fresh
        except RuntimeError as exc:
            _log.info("a camera opened but sent nothing yet: %s", exc)
            for reader in fresh:
                with contextlib.suppress(Exception):
                    reader.stop()
    return None


async def _first_frame(reader, timeout_s: float = 10.0, ctl: FreezeControl | None = None):
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if ctl is not None and ctl.frozen:
            raise FreezeRequested
        if reader.frame is not None:
            return reader.frame
        await asyncio.sleep(0.05)
    raise RuntimeError(f"camera {reader.name!r} sent no frame in {timeout_s:.0f} s")


async def _supervise(loop, relay: Relay, match: MatchConfig, stop: asyncio.Event, out_dir: pathlib.Path,
                     claude: ClaudeVerifier, go_rgb=None, model_for=None, current: str | None = None,
                     statefile=None) -> str | None:
    """Feeds the relay until the match ends (returns None) or the next step needs another model ("switch")."""
    def needs_switch(step) -> bool:
        return model_for is not None and step is not None and model_for(step.stage) != current

    if claude.available:
        # The pass bars that, if wrong, mean a DQ or a lost +20 (stack_all: only Claude can pass it).
        relay.confirm_kinds = {"unplug", "hold", "stack_all"}
        _log.info("Claude vision checks on (%s)", "unplug and tower confirmations, 8-ball watch")
    else:
        _log.info("Claude vision checks off (no key); colour checks only")
    pending: dict[str, asyncio.Task] = {}  # question -> running Claude call
    last_unplug_ask = 0.0
    no_ball_since: float | None = None  # balls stage: when the camera last stopped seeing a coloured ball on the table
    last_balls_ask = 0.0
    saved = len(list(out_dir.glob("step_*.jpg")))  # carries on numbering after a loop restart
    stage = None  # a restarted supervisor writes the current stage on its first poll
    while not stop.is_set():
        await asyncio.sleep(POLL_S)
        if relay.stage != stage:
            stage = relay.stage
            _set_state(statefile, stage=stage)
        frame = loop.video.get(OVERHEAD)
        state = loop.state
        if frame is None or state is None:
            continue
        now = time.monotonic()
        world = vision.detect(frame.data, match.calibration, frame.timestamp_s)
        if relay.stack_start_rgb is None and relay.step is not None and relay.step.kind == "stack_all":
            relay.stack_start_rgb = frame.data.copy()  # the table as stacking begins, for Claude's comparisons
        # Collect Claude answers that came back since the last poll.
        for question, task in list(pending.items()):
            if not task.done():
                continue
            del pending[question]
            yes, why = task.result() if task.exception() is None else (None, repr(task.exception()))
            if not claude.available:
                relay.confirm_kinds.clear()  # Claude went away mid-run: the colour checks decide from here
            if question in ("confirm",):
                if relay.pending_confirm == "stack_all" and match.stack_pass != "6in":
                    if yes:
                        relay.log(now, f"Claude sees the cubes stacked: {why}")
                        for other in pending.values():
                            other.cancel()
                        return "release"
                    relay.resolve_confirm(False, why, now)
                    continue
                # No opinion (unreachable, out of credit) counts as a yes: the colour check already passed.
                relay.resolve_confirm(yes is not False, why, now)
            elif question == "unplugged_since" and yes:
                if relay.external_pass(world, now, why) and _next(loop, relay, world, now, stop, needs_switch):
                    return "switch"
            elif question == "balls_left":
                if yes is False:
                    relay.log(now, f"no coloured ball left within reach: {why}; ending the run (arms home)")
                    stop.set()
                    break
                relay.log(now, f"Claude sees balls still within reach: {why}")
                no_ball_since = None
            elif question == "eight_ball_in_bin" and yes:
                relay.log(now, f"Claude sees the 8-ball in the bin: {why}")
                relay.eight_ball_alert = True
                _set_state(statefile, eight_ball_alert=True)
        if claude.available:
            if relay.pending_confirm and "confirm" not in pending:
                if relay.pending_confirm == "unplug":  # compared with go: a cable already out at go doesn't count
                    call = (claude.ask, "unplugged_since", frame.data, go_rgb)
                elif relay.pending_confirm == "stack_all" and match.stack_pass != "6in" \
                        and relay.stack_start_rgb is not None:
                    # Is it stacked (the gripper may still be on it)? A yes stops the model and checks it standing.
                    call = (claude.ask, "stacked_now", frame.data, relay.stack_start_rgb, _wrists(loop.video.get))
                elif relay.pending_confirm == "stack_all":
                    call = (claude.ask, "tower_tall" if match.stack_pass == "6in" else "tower", frame.data)
                else:
                    call = (claude.ask, "tower", frame.data)
                pending["confirm"] = asyncio.create_task(asyncio.to_thread(*call))
            step = relay.step
            if relay.stage == "balls" and match.all_colors:
                # Balls can roll out of reach: when the camera sees none on the table for 8 s, ask Claude whether any
                # is still reachable; a no ends the run with the score kept, instead of the model wandering.
                seen = [b for c in match.calibration.ball_colors for b in world.balls_on_table(c)]
                no_ball_since = None if seen else (no_ball_since or now)
                if no_ball_since and now - no_ball_since >= 8.0 and now - last_balls_ask > 10.0 \
                        and "balls_left" not in pending:
                    last_balls_ask = now
                    wrist = _wrists(loop.video.get)
                    pending["balls_left"] = asyncio.create_task(
                        asyncio.to_thread(claude.ask, "balls_left", frame.data, None, wrist))
            if step and step.kind == "unplug" and go_rgb is not None and now - last_unplug_ask > 5.0 \
                    and "unplugged_since" not in pending:
                last_unplug_ask = now
                pending["unplugged_since"] = asyncio.create_task(
                    asyncio.to_thread(claude.ask, "unplugged_since", frame.data, go_rgb))
        grip = {"left": float(state.left.gripper), "right": float(state.right.gripper)}
        outcome = relay.observe(world, now, grip)
        if outcome is Outcome.RUNNING:
            continue
        if saved < 200:  # one annotated frame per finished step, for the review afterwards
            cv2.imwrite(str(out_dir / f"step_{saved:03d}_{outcome.value}.jpg"), vision.draw(frame.data, world))
            saved += 1
        finished = relay.step  # None after a pass; the step itself after a failure
        if claude.available and outcome is Outcome.PASSED and relay.stage == "balls" and \
                "eight_ball_in_bin" not in pending:
            pending["eight_ball_in_bin"] = asyncio.create_task(
                asyncio.to_thread(claude.ask, "eight_ball_in_bin", frame.data))
        del finished
        if _next(loop, relay, world, now, stop, needs_switch):
            for task in pending.values():
                task.cancel()
            return "switch"
    for task in pending.values():
        task.cancel()
    return None


def _wrists(get) -> list:
    """The wrist cameras' newest frames (side views of the table) for Claude, as RGB arrays; missing ones skipped."""
    out = []
    for name in ("wrist_l", "wrist_r"):
        frame = get(name)
        if frame is not None and getattr(frame, "data", None) is not None:
            out.append(frame.data)
    return out


def _next(loop, relay: Relay, world, now: float, stop: asyncio.Event, needs_switch=lambda step: False) -> bool:
    """Moves the relay on; True when the new step belongs to another model (the caller swaps loops, so the old
    model never starts on it)."""
    step = relay.next_step(world, now)
    if step is None:
        relay.log(now, f"done: score {relay.score()} (stages passed: {list(relay.passed)})")
        stop.set()
        return False
    if needs_switch(step):
        return True
    # Dream starts a fresh episode on every set_task(), even with the same words, dropping the move in progress.
    # A retry of the same instruction therefore just lets Dream-1 carry on.
    if step.instruction != loop.task:
        loop.set_task(step.instruction)
    return False


def _write_report(relay: Relay, out_dir: pathlib.Path) -> None:
    report = {
        "score": relay.score(),
        "passed": relay.passed,
        "own_in_bin": relay.own_in_bin,
        "other_in_bin": relay.other_in_bin,
        "tower": relay.placed,
        "events": [{"t_s": round(e.t_s, 2), "stage": e.stage, "what": e.what} for e in relay.events],
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
