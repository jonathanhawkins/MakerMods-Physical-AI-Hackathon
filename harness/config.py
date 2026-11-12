"""Match settings and camera calibration for the Hill Climb relay harness.

Everything a match needs is fixed before "go" (the rubric allows no human input after it): our ball colour from the
coin flip, the time budgets, and the overhead camera calibration (where the hub, the bin and the tower spot are in the
image, and what each colour looks like under the venue's light). Calibration lives in a YAML file so it can be tuned
from real frames without touching code (`python -m harness snapshot` writes a frame and the detections over it).
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

import yaml

DEFAULT_CALIBRATION = pathlib.Path(__file__).resolve().parent / "calibration.yaml"

# Stage order is fixed by the rubric.
STAGES = ("unplug", "stack", "balls")


@dataclasses.dataclass
class ColorRange:
    """An HSV box (OpenCV units: H 0-179, S and V 0-255). `h_lo > h_hi` wraps around red."""

    name: str
    h_lo: int
    h_hi: int
    s_lo: int
    s_hi: int
    v_lo: int
    v_hi: int


@dataclasses.dataclass
class Calibration:
    """Overhead camera calibration, in the overhead frame's pixel coordinates (as the model sees it, 640x360)."""

    frame_size: tuple[int, int] = (640, 360)
    hub_roi: list[tuple[int, int]] = dataclasses.field(default_factory=list)  # polygon around the USB hub's ports
    bin_roi: list[tuple[int, int]] = dataclasses.field(default_factory=list)  # polygon of our bin's opening
    tower_roi: list[tuple[int, int]] = dataclasses.field(default_factory=list)  # where the tower is built
    work_roi: list[tuple[int, int]] = dataclasses.field(default_factory=list)  # the table area the arms work in
    ignore_rois: list[list[tuple[int, int]]] = dataclasses.field(default_factory=list)  # e.g. the grippers at rest
    colors: dict[str, ColorRange] = dataclasses.field(default_factory=dict)
    cube_colors: list[str] = dataclasses.field(default_factory=list)
    rod_colors: list[str] = dataclasses.field(default_factory=list)
    ball_colors: list[str] = dataclasses.field(default_factory=list)  # the two ping-pong colours (coin-flip choices)
    cable_colors: list[str] = dataclasses.field(default_factory=list)
    avoid_balls: list[str] = dataclasses.field(default_factory=list)  # never picked (the white balls)
    # Ball colours seen on the table outside this match's rules (other tasks' balls): known to the detectors only.
    other_balls: list[str] = dataclasses.field(default_factory=list)
    eight_ball: str = "black"
    # Blob sizes in pixels at the frame size above.
    cube_area: tuple[int, int] = (250, 4000)
    rod_area: tuple[int, int] = (120, 3000)
    ball_area: tuple[int, int] = (40, 900)
    plug_area: tuple[int, int] = (30, 1500)
    # Measured at the station (Nov 7): cube 2 x 2 x 1.3 in (lies with a 2 x 2 face up), rod 0.8 in across x 4 in.
    cube_height_in: float = 1.3
    cube_side_in: float = 2.0
    rod_length_in: float = 4.0
    rod_diameter_in: float = 0.8
    rod_insert_in: float = 1.3  # how deep a rod sinks into a cube's hole; 1.3 = all the way through (worst case)
    usb_a_width_in: float = 0.59
    usb_c_width_in: float = 0.40
    px_per_cm: float = 4.3  # on the table plane: a cube's 2 in face is ~22 px in the overhead frame

    @classmethod
    def load(cls, path: pathlib.Path = DEFAULT_CALIBRATION) -> "Calibration":
        raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
        colors = {name: ColorRange(name=name, **box) for name, box in (raw.pop("colors", {}) or {}).items()}
        cal = cls(colors=colors)
        for key, value in raw.items():
            if not hasattr(cal, key):
                raise ValueError(f"calibration: unknown key {key!r}")
            if key.endswith("_roi"):
                value = [tuple(p) for p in value]
            elif key == "ignore_rois":
                value = [[tuple(p) for p in poly] for poly in value]
            elif isinstance(getattr(cal, key), tuple):
                value = tuple(value)
            setattr(cal, key, value)
        for group in ("cube_colors", "rod_colors", "ball_colors", "cable_colors", "avoid_balls", "other_balls"):
            for c in getattr(cal, group):
                if c not in cal.colors:
                    raise ValueError(f"calibration: {group} names {c!r}, which has no HSV range under colors")
        return cal


def spoken(color: str) -> str:
    """How a calibration colour is said to Dream-1: `green_ball` is just "green"."""
    return color.split("_")[0]


@dataclasses.dataclass
class MatchConfig:
    """One match's settings."""

    own_color: str | None  # the coin-flip colour of our ping-pong balls; None: every non-white ball is ours
    time_limit_s: float = 300.0
    budgets_s: dict[str, float] = dataclasses.field(
        default_factory=lambda: {"unplug": 30.0, "stack": 120.0, "balls": 1e9})
    step_timeout_s: float = 25.0  # one instruction to Dream-1
    unplug_timeout_s: float = 45.0  # the first move: Dream-1's round trips ran 1-5 s in practice
    retries: int = 2  # per step, before a recovery move, then a skip
    target_height_in: float = 7.0  # aim: an inch over the 6 in pass bar when the pieces allow it
    min_height_in: float = 6.3  # never call the stack done below this (the bar is 6 in)
    # True: stand the rod in the top cube while it is on the table, then stack that cube last (the default).
    # False: stack every cube, then stand the rod in the top one on the tower (if the rod slides out of the
    # through-hole when its cube is lifted).
    rod_first: bool = True
    # A fine-tune learns the exact task text of its demos: when set, the unplug step sends just this, every try.
    trained_unplug: str | None = None
    # Per stage ("unplug", "stack", "balls"): the Dream model to run it with (missing: the run's --checkpoint), and
    # the task text its demos were recorded with. A stage with a text is run as one instruction, word for word,
    # instead of the step-by-step base-model plan (stack: the whole tower in one go; balls: every pick).
    stage_models: dict[str, str] = dataclasses.field(default_factory=dict)
    stage_texts: dict[str, str] = dataclasses.field(default_factory=dict)
    # Per stage, Metal gravity compensation x this while that stage's model runs (harness/gravity.py); a stage not
    # named keeps the run's --gravity-scale.
    stage_gravity: dict[str, float] = dataclasses.field(default_factory=dict)
    # Stack pass bar for a whole-tower fine-tune, checked by Claude: "6in" (the rubric as we read it) or "3-cubes"
    # (three cubes stacked, standing on their own: what the team's demos build, 3.9 in).
    # Seconds of match time the cable model keeps pulling after the first cable is out (the same sentence, so Dream
    # carries on): its demos pull every cable. 0 = move on at the first cable.
    pull_cables_until_s: float = 0.0
    # Practice: start the relay at this stage ("unplug", "stack" or "balls"); the earlier stages are skipped, not
    # scored. A real match always starts at "unplug".
    start_stage: str = "unplug"
    stack_pass: str = "6in"
    # True: every ball colour but white and black scores (what the team's ball demos do): any coloured ball in the
    # bin counts, none is taken back out, and a ball next to a white one is still a target (the model pushes whites).
    all_colors: bool = False
    stack_all_timeout_s: float = 90.0  # one whole-tower instruction, before it is sent again (same words: no reset)
    hold_s: float = 3.0  # the rubric's free-standing check after the gripper lets go
    clear_of_later_cm: float = 8.0  # never instruct a target this close to a later stage's object
    clear_of_eight_cm: float = 6.0  # never pick a ball this close to the black 8-ball
    clear_of_avoid_cm: float = 4.0  # nor this close to a never-grab (white) ball
    stop_margin_s: float = 8.0  # stop starting new steps this close to the time limit
    calibration: Calibration = dataclasses.field(default_factory=Calibration)

    def other_color(self) -> str | None:
        if self.own_color is None:
            return None
        others = [c for c in self.calibration.ball_colors if c != self.own_color]
        return others[0] if others else "other"
