"""The relay: which stage we are in, what to tell Dream-1 next, and whether the last instruction worked.

Pure logic over `World` snapshots, so it is tested without a robot. The executor asks `next_step()` for an
instruction, runs Dream-1 on it while feeding `observe()` fresh worlds, and the relay says when the step passed,
failed or timed out. The rubric's rules shape every decision:

- Stages run in order, and the gripper may touch a later stage's objects only once the current pass bar is met: every
  instruction names only the current stage's objects and says what to leave alone, and a target too close to a later
  stage's object is skipped.
- Unplug passes at one cable out. Stack passes at 6 in, standing on its own 3 s after the gripper lets go: we build
  to `target_height_in` and stop. Balls: +5 own colour, -5 other colour, and the black 8-ball in our bin loses.
"""

from __future__ import annotations

import dataclasses
import enum
import logging

from harness.config import MatchConfig, spoken
from harness.vision import Obj, World

_log = logging.getLogger(__name__)

# The task text the team's unplug demos are recorded with (~/hackathon_data/USB_Cable_Task): a fine-tuned model
# runs on exactly this sentence.
TRAINED_UNPLUG = "Remove Orange Cables from their port"
# The other stages' demo texts (~/hackathon_data/Stacking_Boxes, Balls_In_Basket): sent word for word to their
# fine-tuned models, the stage run as one instruction.
TRAINED_STACK = "Stack the green, purple, and blue blocks"
TRAINED_BALLS = "Place yellow, blue, green, and orange balls into basket."
STACK_ALL_MIN_S = 15.0  # a whole-tower instruction is checked by Claude only after this long

UNPLUG_WORDINGS = (
    # The strip's USB ports face up, so each plug comes out by pulling straight up (the team's wording first).
    "grab the orange USB cable's plug at the power strip, then pull it straight up out of the port",
    "pinch the orange plug where it goes into the power strip and lift it straight up until it comes out",
    "pull one orange USB plug straight up out of the power strip",
)

AVOID = {
    "unplug": "without touching the blocks or the balls",
    "stack": "without touching the balls or the cables",
    "balls": "and do not touch the white or the black balls",
}
PARK = "move both arms up and away from the tower and hold them still"
PUT_DOWN_TIMEOUT_S = 15.0


class Outcome(enum.Enum):
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"


@dataclasses.dataclass
class Step:
    stage: str
    kind: str  # unplug | rod_in_cube | place_cube | hold | stack_all | ball | take_out | park
    instruction: str
    target: str  # what the step is about, for logs
    timeout_s: float
    needs_release: bool = True  # the check counts only once the gripper has opened again
    confirm_polls: int = 3  # consecutive passing checks before the step counts as done


@dataclasses.dataclass
class Event:
    t_s: float
    stage: str
    what: str


class Relay:
    """The match plan. Call `start(world, t)` once, then loop: `next_step(world, t)` and `observe(world, t, ...)`."""

    def __init__(self, match: MatchConfig) -> None:
        self.m = match
        self.cal = match.calibration
        self.stage_index = 0
        self.stage_started_s = 0.0
        self.started_s = 0.0
        self._started = False
        self.passed: dict[str, float] = {}  # stage -> pass time since go
        self.events: list[Event] = []
        self.attempts: dict[str, int] = {}
        self.skipped: set[str] = set()
        self.step: Step | None = None
        # Claude confirmation: a step of a kind in confirm_kinds passes only after resolve_confirm(True).
        self.confirm_kinds: set[str] = set()
        self.pending_confirm: str | None = None
        self._confirmed = False
        self.eight_ball_alert = False  # set when Claude sees the 8-ball in our bin
        self.holding: str | None = None  # what a gripper still holds after a passed step (the pulled cable)
        self.holding_stage: str | None = None  # the stage whose step left it held: its model puts it down
        self.stack_start_rgb = None  # the overhead frame when stacking began: Claude compares the tower with it
        self.cables_done_s: float | None = None  # when the first cable came out, while the cable model pulls more
        self._last_grip: dict[str, float] = {}
        self._step_started_s = 0.0
        self._good_polls = 0
        self._released_since_start = False
        self._gripper_was_closed = False
        # stage memory
        self.initial_plugs = 0
        self.initial_cable_px = 0
        self.initial_off_hub = 0
        self.go_world: World | None = None
        self.abandoned: set[str] = set()  # tactics given up for this match (never retried)
        self.tower_xy: tuple[float, float] | None = None
        self.stack_order: list[str] = []  # cube colours, base first
        self.rod_cube: str | None = None  # the cube that gets a rod (the top one)
        self.rod_color: str | None = None
        self.placed: list[str] = []  # tower pieces placed so far, base first ("cube:blue", "rod:red")
        self.hold_from_s: float | None = None
        self.own_in_bin = 0
        self.other_in_bin = 0
        self._own_at_start = 0
        self._other_at_start = 0
        self.log_plan = ""

    # ------------------------------------------------------------------ bookkeeping
    @property
    def stage(self) -> str:
        return self.m_stages[self.stage_index] if self.stage_index < len(self.m_stages) else "done"

    @property
    def m_stages(self) -> tuple[str, ...]:
        return ("unplug", "stack", "balls")

    def log(self, t: float, what: str) -> None:
        if not self._started:  # before go: no match clock yet
            t_s, stage = 0.0, "startup"
        else:
            t_s, stage = t - self.started_s, self.stage
        self.events.append(Event(t_s, stage, what))
        _log.info("[%5.1f s] %s: %s", t_s, stage, what)

    def time_left(self, t: float) -> float:
        return self.m.time_limit_s - (t - self.started_s)

    def score(self) -> int:
        pts = (10 if "unplug" in self.passed else 0) + (20 if "stack" in self.passed else 0)
        other = 5 * self.other_in_bin if self.m.other_color() else 0
        return pts + 5 * self.own_in_bin - other

    # ------------------------------------------------------------------ the match
    def start(self, world: World, t: float) -> None:
        self.started_s = self.stage_started_s = t
        self._started = True
        self.go_world = world  # the cleanest view of the table: nothing has moved and no arm is in the way
        self.initial_plugs = len(world.plugs_in_hub())
        self.initial_off_hub = len(world.plugs_off_hub())
        self.initial_cable_px = world.cable_px_in_hub
        self.log(t, f"go: {world.summary()}; {self.initial_plugs} plug blob(s) at the hub")
        if self.m.start_stage != "unplug":
            self.stage_index = self.m_stages.index(self.m.start_stage)
            self.log(t, f"practice: starting at the {self.m.start_stage} stage (earlier stages skipped, not scored)")

    def _advance(self, t: float, why: str) -> None:
        stage = self.stage
        self.passed[stage] = t - self.started_s
        self.log(t, f"PASSED ({why}) at {self.passed[stage]:.1f} s")
        self.stage_index += 1
        self.stage_started_s = t
        self.step = None
        self.attempts.clear()

    def next_step(self, world: World, t: float) -> Step | None:
        """The instruction to run now, or None when the match is over for us."""
        if self.time_left(t) < self.m.stop_margin_s or self.stage == "done":
            return None
        if self.holding:
            # A gripper still holds what the last step moved (the cable it pulled): put it down to the side first,
            # or the next instruction (pick up a rod) starts with a full hand.
            # It runs on the model of the stage whose step left the gripper closed (the cable model, whose demos drop
            # each pulled cable to the side), with that model's own sentence when it has one: a switch to the next
            # stage's model with a cable still in hand froze the arms while that model started on Dream's servers.
            src = self.holding_stage or self.stage
            text = self.m.stage_texts.get(src) or (
                f"put the {self.holding} down on the empty table beside the power strip, away from the blocks and the "
                f"balls, and open the gripper")
            step = Step(src, "put_down", text, self.holding, PUT_DOWN_TIMEOUT_S, needs_release=False,
                        confirm_polls=2)
        else:
            maker = {"unplug": self._next_unplug, "stack": self._next_stack, "balls": self._next_balls}[self.stage]
            step = maker(world, t)
        if step is not None:
            key = f"{step.kind}:{step.target}"
            self.attempts[key] = self.attempts.get(key, 0) + 1
            self.step, self._step_started_s, self._good_polls = step, t, 0
            self._released_since_start, self._gripper_was_closed = False, False
            self.hold_from_s = t if step.kind == "hold" else None
            self.pending_confirm, self._confirmed = None, False
            self._own_at_start = self._own_in_bin(world)
            self._other_at_start = self._other_in_bin(world)
            self.log(t, f"step {step.kind} #{self.attempts[key]}: {step.instruction!r}")
        return step

    def observe(self, world: World, t: float, gripper_open: dict[str, float]) -> Outcome:
        """Feeds one fresh world while a step runs; says whether it passed, failed (timed out) or still runs.

        @param gripper_open Each arm's gripper opening, 0 closed to 1 open (the loop's `state`)."""
        step = self.step
        if step is None:
            return Outcome.FAILED
        self._last_grip = dict(gripper_open)
        closed = min(gripper_open.values(), default=1.0) < 0.6
        if closed:
            self._gripper_was_closed = True
        elif self._gripper_was_closed:
            self._released_since_start = True
        released = self._released_since_start or not step.needs_release
        ok = released and not closed and self._check(step, world, t)
        self._good_polls = self._good_polls + 1 if ok else 0
        if self._good_polls >= step.confirm_polls:
            if step.kind in self.confirm_kinds and not self._confirmed:
                if self.pending_confirm is None:
                    self.pending_confirm = step.kind  # the executor asks Claude, then calls resolve_confirm()
                    self.log(t, f"step {step.kind}: camera check passed; asking Claude to confirm")
            else:
                self._on_pass(step, world, t)
                return Outcome.PASSED
        if self.time_left(t) < self.m.stop_margin_s:
            self.log(t, f"match clock nearly out ({self.time_left(t):.0f} s left): ending the step")
            return Outcome.FAILED
        if step.kind == "unplug_more" and t - self._step_started_s > step.timeout_s:
            # Time's up for pulling: close the stage (already passed at the first cable) and move on.
            if min(gripper_open.values(), default=1.0) < 0.6:
                self.holding, self.holding_stage = "orange USB cable", "unplug"
            self.step = None
            self._advance(t, f"a cable out at {self.cables_done_s:.0f} s, pulled more until {t - self.started_s:.0f} s")
            return Outcome.PASSED
        if t - self._step_started_s > step.timeout_s:
            self.log(t, f"step {step.kind} timed out after {step.timeout_s:.0f} s")
            key = f"{step.kind}:{step.target}"
            if step.kind == "put_down" and self.attempts.get(key, 0) >= 2:
                self.log(t, f"could not put the {self.holding} down; carrying on")
                self.holding = self.holding_stage = None
            if self.attempts.get(key, 0) > self.m.retries:
                self.skipped.add(key)
                self.log(t, f"skipping {key} after {self.attempts[key]} tries")
            return Outcome.FAILED
        return Outcome.RUNNING

    def resolve_confirm(self, ok: bool, why: str, t: float) -> None:
        """Claude's answer to `pending_confirm`: yes lets the step pass on the next poll, no keeps it running."""
        kind, self.pending_confirm = self.pending_confirm, None
        self.log(t, f"Claude {'confirms' if ok else 'does NOT confirm'} {kind}: {why}")
        if ok:
            self._confirmed = True
        else:
            self._good_polls = 0

    def tower_standing_pass(self, world: World, t: float, why: str) -> bool:
        """Claude saw the tower standing after the arms moved away (stack_all only): the stage passes."""
        if self.step is None or self.step.kind != "stack_all":
            return False
        self.pending_confirm = None
        self.log(t, f"Claude sees the tower standing with the arms away: {why}")
        self._on_pass(self.step, world, t)
        return True

    def external_pass(self, world: World, t: float, why: str) -> bool:
        """Claude saw the current step's goal met although the colour check did not (unplug only)."""
        if self.step is None or self.step.kind != "unplug":
            return False
        self.log(t, f"Claude sees the cable out: {why}")
        self._on_pass(self.step, world, t)
        return True

    # ------------------------------------------------------------------ stage 1: unplug
    def _next_unplug(self, world: World, t: float) -> Step | None:
        if self.cables_done_s is not None:
            # More cables: the same words (Dream carries on, no reset), run until the time set, never "passed" early.
            left = max(1.0, self.m.pull_cables_until_s - (t - self.started_s))
            return Step("unplug", "unplug_more", self._unplug_step().instruction, "more", left, needs_release=False,
                        confirm_polls=10**6)
        plugs = world.plugs_in_hub()
        later = world.of("cube") + world.of("rod") + world.of("ball") + world.of("eight_ball")
        ranked = sorted(plugs, key=lambda p: -world.nearest_cm(p, later))  # the most room around it first
        for p in ranked:
            if world.nearest_cm(p, later) < self.m.clear_of_later_cm:
                continue
            if f"unplug:{p.color}" in self.skipped:
                continue
            return self._unplug_step()
        # The station's cables are all orange: name the hub, not a colour (the pass check counts the hub's cables).
        return self._unplug_step()

    def _unplug_step(self) -> Step:
        """The unplug instruction, worded differently on each try (Dream-1 sometimes acts on one wording only)."""
        text = self.m.stage_texts.get("unplug") or self.m.trained_unplug
        if text:  # the fine-tune's own demo text, word for word, and the same on every retry
            return Step("unplug", "unplug", text, "any", self.m.unplug_timeout_s)
        tries = self.attempts.get("unplug:any", 0)
        words = UNPLUG_WORDINGS[tries % len(UNPLUG_WORDINGS)]
        return Step("unplug", "unplug", f"{words} {AVOID['unplug']}", "any", self.m.unplug_timeout_s)

    # ------------------------------------------------------------------ stage 2: stack
    def _plan_stack(self, world: World) -> None:
        # Plan from whichever view sees more cubes: the go frame (no arm in the way) or the current one (a cube
        # may have been nudged since). Positions come from the current frame when it has the cube.
        go = self.go_world
        if go is not None and len(go.of("cube")) > len(world.of("cube")):
            world = go
        cubes = sorted(world.of("cube"), key=lambda o: o.brightness)  # darkest first = the base
        self.stack_order = [c.color for c in cubes]
        if cubes:
            self.tower_xy = (cubes[0].x, cubes[0].y)
        need_in = self.m.target_height_in
        cube_in = self.cal.cube_height_in * len(cubes)
        rods = world.of("rod")
        if cubes and rods and cube_in < need_in:  # a rod standing in the top cube buys the extra inches
            top = cubes[-1]
            # Prefer a rod whose colour no cube shares, so it can't be confused with a cube on camera.
            distinct = [r for r in rods if r.color not in self.cal.cube_colors] or rods
            rod = min(distinct, key=lambda r: r.dist_cm(top, self.cal.px_per_cm))
            self.rod_cube, self.rod_color = top.color, rod.color
        self.placed = [f"cube:{self.stack_order[0]}"] if self.stack_order else []
        self.log_plan = (f"tower {' / '.join(self.stack_order)} (base first)"
                         + (f", {self.rod_color} rod in the {self.rod_cube} cube" if self.rod_color else "")
                         + f": {self.plan_height_in():.1f} in planned")
        if self.plan_height_in() < self.m.min_height_in:
            self.log_plan += f" -- BELOW the {self.m.min_height_in} in we need; check the pieces and calibration"

    def height_in(self) -> float:
        cubes = sum(1 for p in self.placed if p.startswith("cube:"))
        h = cubes * self.cal.cube_height_in
        if self.rod_cube and f"cube:{self.rod_cube}" in self.placed and "rod:" + str(self.rod_color) in self.placed:
            h += self.cal.rod_length_in - self.cal.rod_insert_in
        return h

    def plan_height_in(self) -> float:
        """The height the whole plan reaches: every cube, plus the rod standing in the top one."""
        h = len(self.stack_order) * self.cal.cube_height_in
        return h + (self.cal.rod_length_in - self.cal.rod_insert_in if self.rod_color else 0.0)

    def _next_stack(self, world: World, t: float) -> Step | None:
        text = self.m.stage_texts.get("stack")
        if text:
            # A fine-tune trained on whole-tower demos: one instruction, word for word, sent again (same words, so
            # Dream carries on) until Claude confirms a tall, free-standing tower.
            return Step("stack", "stack_all", text, "tower", self.m.stack_all_timeout_s, needs_release=False,
                        confirm_polls=2)
        if not self.stack_order:
            self._plan_stack(world)
            self.log(t, f"plan: {self.log_plan}")
            if not self.stack_order:
                # Never move on without the pass: touching a ball now would be a DQ. Park and look again.
                self.log(t, "no cubes seen; parking and looking again")
                return Step("stack", "park", PARK, "none", 3.0, needs_release=False, confirm_polls=99)
        # Pass check first: tall enough, then the 3 s free-standing hold with the arms parked.
        if self.height_in() >= self.m.target_height_in or len(self.placed) >= len(self.stack_order) + (
                1 if self.rod_color else 0):
            if self.height_in() >= self.m.min_height_in:
                return Step("stack", "hold", PARK, "tower", self.m.hold_s + 6.0, needs_release=False,
                            confirm_polls=1)
        # 1. The rod goes into the top cube while that cube is still on the table, where the rod can't roll off
        #    (unless rod_first is off: then it goes into the top cube once that cube is on the tower, step 3).
        table_key = f"rod_in_cube:{self.rod_color}"
        if self.rod_color and self.attempts.get(table_key, 0) > self.m.retries and "rod_on_table" not in \
                self.abandoned:
            self.abandoned.add("rod_on_table")
            self.log(t, "rod on the table failed; switching to: cubes first, then the rod in the top cube on the tower")
        if self.m.rod_first and "rod_on_table" not in self.abandoned and self.rod_color \
                and f"rod:{self.rod_color}" not in self.placed:
            return Step("stack", "rod_in_cube",
                        f"put the {self.rod_color} rod upright in the hole of the {self.rod_cube} cube "
                        f"{AVOID['stack']}", self.rod_color, self.m.step_timeout_s)
        # 2. Next cube onto the tower.
        placed_cubes = [p.split(":")[1] for p in self.placed if p.startswith("cube:")]
        for c in self.stack_order:
            if c in placed_cubes or f"place_cube:{c}" in self.skipped:
                continue
            below = placed_cubes[-1]
            if c == self.rod_cube and f"rod:{self.rod_color}" in self.placed and "rod_on_table" not in self.abandoned:
                # The top cube carries its rod: say to hold the cube, or Dream-1 may grab the rod sticking up.
                text = (f"pick up the {c} cube with the {self.rod_color} rod standing in it by the sides of the cube "
                        f"and place it on top of the {below} cube {AVOID['stack']}")
            else:
                text = f"place the {c} cube on top of the {below} cube {AVOID['stack']}"
            return Step("stack", "place_cube", text, c, self.m.step_timeout_s)
        # 3. The rod into the top cube on the tower (rod_first off, or the table attempts failed): it rests on the
        #    cube below, so the height is the same.
        if self.rod_color and f"rod:{self.rod_color}" not in self.placed \
                and f"rod_on_tower:{self.rod_color}" not in self.skipped:
            return Step("stack", "rod_on_tower",
                        f"put the {self.rod_color} rod upright in the hole of the {self.rod_cube} cube on top of "
                        f"the tower {AVOID['stack']}", self.rod_color, self.m.step_timeout_s)
        # Nothing left to place: hold if tall enough, else the stage can't pass with what's here.
        if self.height_in() >= self.m.min_height_in:
            return Step("stack", "hold", PARK, "tower", self.m.hold_s + 6.0, needs_release=False, confirm_polls=1)
        self.log(t, f"out of pieces at {self.height_in():.1f} in; retrying skipped steps")
        self.skipped.clear()
        return self._next_stack(world, t) if self.attempts else None

    # ------------------------------------------------------------------ stage 3: balls
    def _next_balls(self, world: World, t: float) -> Step | None:
        trained = self.m.stage_texts.get("balls")
        if trained:
            # A fine-tune knows only its own sentence: send just that, every time. Its demos never took a ball back
            # out, so no take-out or park steps (their words would be new to it); the bin is still counted.
            return Step("balls", "ball", trained, self.m.own_color or "any", self.m.step_timeout_s)
        if world.eight_ball_in_bin() or self.eight_ball_alert:
            self.eight_ball_alert = False
            return Step("balls", "take_out", "take the black ball out of the bin and put it on the table",
                        "eight_ball", self.m.step_timeout_s, needs_release=True)
        other = self.m.other_color()
        if not self._every_colour() and other and world.balls_in_bin(other) and self.time_left(t) > 20:
            return Step("balls", "take_out", f"take the {spoken(other)} ping-pong ball out of the bin and put it on "
                        "the table", other, self.m.step_timeout_s)
        for white in world.avoid_in_bin():
            if self.time_left(t) > 20:
                return Step("balls", "take_out", f"take the {spoken(white.color)} ping-pong ball out of the bin and "
                            "put it on the table", white.color, self.m.step_timeout_s)
        eight, avoid = world.of("eight_ball"), world.of("avoid_ball")
        # A whole-stage ball model runs its own sentence and picks its own ball; parking because a white ball is near
        # one would only waste the clock.
        near_white_ok = self.m.all_colors and bool(self.m.stage_texts.get("balls"))
        on_table = [b for c in self._scoring_colors() for b in world.balls_on_table(c)]
        candidates = [b for b in on_table
                      if world.nearest_cm(b, eight) >= self.m.clear_of_eight_cm
                      and (near_white_ok or world.nearest_cm(b, avoid) >= self.m.clear_of_avoid_cm)]
        if not candidates:
            if on_table:
                self.log(t, "every own-colour ball left is next to a black or white ball; waiting")
            return Step("balls", "park", PARK, "none", 3.0, needs_release=False, confirm_polls=99)
        if self.m.own_color and not self.m.all_colors:
            target = self.m.own_color
        else:
            fresh = [b for b in candidates if f"ball:{b.color}" not in self.skipped]  # a skipped colour goes last
            target = (fresh or candidates)[0].color
        text = self.m.stage_texts.get("balls") or (f"pick up the {spoken(target)} ping-pong ball and drop "
                                                   f"it in the bin {AVOID['balls']}")
        return Step("balls", "ball", text, target, self.m.step_timeout_s)

    def _every_colour(self) -> bool:
        """--all-colors, or no --color at all: every ball colour but the avoid (white) balls scores."""
        return self.m.all_colors or not self.m.own_color

    def _scoring_colors(self) -> list[str]:
        if not self._every_colour():
            return [self.m.own_color]
        return [c for c in self.cal.ball_colors if c not in self.cal.avoid_balls]

    def _own_in_bin(self, world: World) -> int:
        return sum(len(world.balls_in_bin(c)) for c in self._scoring_colors())

    def _other_in_bin(self, world: World) -> int:
        other = self.m.other_color()
        return len(world.balls_in_bin(other)) if other and not self._every_colour() else 0

    # ------------------------------------------------------------------ checks
    def _check(self, step: Step, world: World, t: float) -> bool:
        cal = self.cal
        if step.kind == "unplug":
            fewer_plugs = self.initial_plugs > 0 and len(world.plugs_in_hub()) < self.initial_plugs
            fewer_px = self.initial_cable_px >= 15 and world.cable_px_in_hub < 0.6 * self.initial_cable_px
            head_off = len(world.plugs_off_hub()) > self.initial_off_hub  # a plug head now lies off the strip
            return fewer_plugs or fewer_px or head_off
        if step.kind in ("rod_in_cube", "rod_on_tower"):
            cubes = world.of("cube", self.rod_cube)
            rods = world.of("rod", self.rod_color)
            if not cubes or any(r.lying for r in rods):
                return False
            # Standing: an upright rod of that colour at the cube (or hidden behind the gripper's view of it).
            return not rods or any(r.dist_cm(c, cal.px_per_cm) < 4.0 for r in rods for c in cubes)
        if step.kind == "place_cube":
            if self.tower_xy is None:
                return False
            mine = world.of("cube", step.target)
            return any(abs(c.x - self.tower_xy[0]) + abs(c.y - self.tower_xy[1]) < 3.0 * cal.px_per_cm
                       for c in mine)
        if step.kind == "put_down":
            return bool(self._last_grip) and min(self._last_grip.values()) >= 0.6
        if step.kind == "stack_all":
            # The colour check cannot measure a tower's height, so only Claude passes it ("tower_tall"): without
            # Claude the step just runs on until the clock says stop, and the stage is never claimed.
            return "stack_all" in self.confirm_kinds and t - self._step_started_s >= STACK_ALL_MIN_S
        if step.kind == "hold":
            if not self._tower_standing(world):
                self.hold_from_s = t  # restart the 3 s whenever the tower is not seen standing
            return self.hold_from_s is not None and t - self.hold_from_s >= self.m.hold_s
        if step.kind == "ball":
            return self._own_in_bin(world) > self._own_at_start
        if step.kind == "take_out":
            if step.target == "eight_ball":
                return not world.eight_ball_in_bin()
            if step.target in self.cal.avoid_balls:
                return not world.inside(self.cal.bin_roi, world.of("avoid_ball", step.target))
            return len(world.balls_in_bin(step.target)) < self._other_at_start
        return False

    def _tower_standing(self, world: World) -> bool:
        if self.tower_xy is None:
            return False
        top = world.of("cube", self.stack_order[-1] if self.stack_order else "")
        return any(abs(c.x - self.tower_xy[0]) + abs(c.y - self.tower_xy[1]) < 4.0 * self.cal.px_per_cm for c in top)

    def _on_pass(self, step: Step, world: World, t: float) -> None:
        self.log(t, f"step {step.kind} passed ({step.target})")
        still_closed = min(self._last_grip.values(), default=1.0) < 0.6
        if step.kind == "put_down":
            self.holding = self.holding_stage = None
        elif step.kind == "unplug" and still_closed:
            self.holding, self.holding_stage = "orange USB cable", "unplug"
        if step.kind == "unplug":
            if self.m.pull_cables_until_s and t - self.started_s < self.m.pull_cables_until_s - 3.0:
                # The stage's bar is met (later objects may be touched from now on); the cable model keeps pulling the
                # rest until the time set, then the stage closes. holding is decided when it does.
                self.holding = self.holding_stage = None
                self.cables_done_s = t - self.started_s
                self.log(t, f"a cable is out; pulling more until {self.m.pull_cables_until_s:.0f} s")
            else:
                self._advance(t, "a cable is out")
        elif step.kind in ("rod_in_cube", "rod_on_tower"):
            self.placed.append(f"rod:{self.rod_color}")
        elif step.kind == "place_cube":
            self.placed.append(f"cube:{step.target}")
            self.log(t, f"tower now {self.height_in():.1f} in: {self.placed}")
        elif step.kind == "hold":
            self._advance(t, f"tower {self.height_in():.1f} in, standing {self.m.hold_s:.0f} s")
        elif step.kind == "stack_all":
            self._advance(t, "Claude sees a tall tower standing on its own")
        elif step.kind in ("ball", "take_out"):
            self.own_in_bin = self._own_in_bin(world)
            self.other_in_bin = self._other_in_bin(world)
            self.log(t, f"bin: {self.own_in_bin} own, {self.other_in_bin} other; score {self.score()}")
        self.step = None
