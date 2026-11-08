"""Offline tests of the relay: scripted worlds stand in for the camera, so no robot or Dream call is needed.

    repos/dimos/.venv/bin/python -m pytest harness/test_relay.py -q
"""

from harness.config import Calibration, ColorRange, MatchConfig
from harness.relay import Outcome, Relay
from harness.vision import Obj, World

OPEN = {"left": 1.0, "right": 1.0}
CLOSED = {"left": 1.0, "right": 0.1}


def cal() -> Calibration:
    c = Calibration(
        hub_roi=[(0, 0), (100, 0), (100, 100), (0, 100)],
        bin_roi=[(500, 0), (600, 0), (600, 100), (500, 100)],
        tower_roi=[(200, 200), (300, 200), (300, 300), (200, 300)],
        cube_colors=["blue", "green", "purple"], rod_colors=["red"], ball_colors=["orange", "white"],
        cable_colors=["orange"], px_per_cm=4.3, cube_height_in=1.3, rod_length_in=4.0, rod_insert_in=1.3)
    c.colors = {k: ColorRange(k, 0, 1, 0, 1, 0, 1) for k in ["blue", "green", "purple", "red", "orange", "white",
                                                          "black"]}
    return c


def o(kind, color, x, y, bright=100.0, elong=1.0):
    return Obj(kind, color, x, y, 300, elong, (int(x) - 5, int(y) - 5, 10, 10), bright)


def world(objs, c):
    return World(list(objs), c)


def run_step(relay, worlds, t, grips=None):
    """Feeds worlds one poll each (0.5 s apart); returns (outcome, t)."""
    grips = grips or [OPEN] * len(worlds)
    out = Outcome.RUNNING
    for w, g in zip(worlds, grips):
        t += 0.5
        out = relay.observe(w, t, g)
        if out is not Outcome.RUNNING:
            break
    return out, t


def test_full_relay_in_order():
    c = cal()
    m = MatchConfig(own_color="orange", calibration=c)
    r = Relay(m)
    plug = o("plug", "orange", 50, 50)
    cubes = [o("cube", "blue", 250, 250, 60), o("cube", "green", 380, 300, 150), o("cube", "purple", 400, 200, 110)]
    rod = o("rod", "red", 420, 320, elong=5.0)
    balls = [o("ball", "orange", 450, 50), o("ball", "white", 460, 120), o("eight_ball", "black", 300, 30)]
    t = 0.0
    w0 = world([plug, *cubes, rod, *balls], c)
    r.start(w0, t)
    # Stage 1: only the cable is named, and the instruction says to leave the rest alone.
    s = r.next_step(w0, t)
    assert s.kind == "unplug" and "blocks" in s.instruction and "ball" in s.instruction
    w1 = world([*cubes, rod, *balls], c)  # plug gone from the hub
    out, t = run_step(r, [w0, w0, w1, w1, w1, w1], t, [OPEN, CLOSED, OPEN, OPEN, OPEN, OPEN])
    assert out is Outcome.PASSED and r.stage == "stack"
    # Stage 2 plan: darkest cube (blue) is the base, lightest (green) gets the rod and goes on top.
    s = r.next_step(w1, t)
    assert r.stack_order == ["blue", "purple", "green"]
    assert s.kind == "rod_in_cube" and "red rod" in s.instruction and "green cube" in s.instruction
    w2 = world([*cubes, o("rod", "red", 382, 296, elong=2.0), *balls], c)  # the rod stands in the green cube
    out, t = run_step(r, [w1, w2, w2, w2, w2], t, [CLOSED, OPEN, OPEN, OPEN, OPEN])
    assert out is Outcome.PASSED
    s = r.next_step(w2, t)
    assert s.kind == "place_cube" and "purple cube on top of the blue" in s.instruction
    purple_on = o("cube", "purple", 252, 251, 110)
    w3 = world([cubes[0], cubes[1], purple_on, *balls], c)
    out, t = run_step(r, [w2, w3, w3, w3, w3], t, [CLOSED, OPEN, OPEN, OPEN, OPEN])
    assert out is Outcome.PASSED
    s = r.next_step(w3, t)
    assert s.kind == "place_cube" and "green cube with the red rod" in s.instruction
    assert "by the sides of the cube" in s.instruction and "on top of the purple cube" in s.instruction
    green_on = o("cube", "green", 251, 249, 150)
    w4 = world([cubes[0], purple_on, green_on, *balls], c)
    out, t = run_step(r, [w3, w4, w4, w4, w4], t, [CLOSED, OPEN, OPEN, OPEN, OPEN])
    assert out is Outcome.PASSED and abs(r.height_in() - 6.6) < 1e-9  # 3 x 1.3 in + (4 - 1.3) in of rod
    # Hold 3 s with the arms parked; only then may a ball be touched.
    s = r.next_step(w4, t)
    assert s.kind == "hold"
    out, t = run_step(r, [w4] * 10, t)
    assert out is Outcome.PASSED and r.stage == "balls"
    s = r.next_step(w4, t)
    assert s.kind == "ball" and "orange ping-pong ball" in s.instruction and "black ball" in s.instruction
    w5 = world([cubes[0], purple_on, green_on, o("ball", "orange", 550, 50), balls[1], balls[2]], c)
    out, t = run_step(r, [w4, w5, w5, w5, w5], t, [CLOSED, OPEN, OPEN, OPEN, OPEN])
    assert out is Outcome.PASSED and r.score() == 10 + 20 + 5


def test_never_advances_without_a_pass():
    c = cal()
    r = Relay(MatchConfig(own_color="orange", calibration=c))
    w = world([o("plug", "orange", 50, 50), o("cube", "blue", 250, 250)], c)
    r.start(w, 0.0)
    r.next_step(w, 0.0)
    t = 0.0
    for _ in range(200):  # the plug never leaves; the stage must never change
        t += 0.5
        if r.observe(w, t, OPEN) is not Outcome.RUNNING:
            r.next_step(w, t)
    assert r.stage == "unplug"


def test_eight_ball_in_bin_is_fixed_first():
    c = cal()
    r = Relay(MatchConfig(own_color="orange", calibration=c))
    r.stage_index = 2
    w = world([o("eight_ball", "black", 550, 50), o("ball", "orange", 450, 50)], c)
    r.start(w, 0.0)
    s = r.next_step(w, 0.0)
    assert s.kind == "take_out" and "black ball" in s.instruction


def test_skips_balls_next_to_the_eight_ball():
    c = cal()
    r = Relay(MatchConfig(own_color="orange", calibration=c))
    r.stage_index = 2
    w = world([o("eight_ball", "black", 300, 300), o("ball", "orange", 305, 302)], c)  # ~1.4 cm apart
    r.start(w, 0.0)
    assert r.next_step(w, 0.0).kind == "park"


def test_stops_near_the_time_limit():
    c = cal()
    r = Relay(MatchConfig(own_color="orange", calibration=c, time_limit_s=300))
    w = world([o("plug", "orange", 50, 50)], c)
    r.start(w, 0.0)
    assert r.next_step(w, 295.0) is None


def test_rod_on_tower_order():
    c = cal()
    r = Relay(MatchConfig(own_color="orange", calibration=c, rod_first=False))
    cubes = [o("cube", "blue", 250, 250, 60), o("cube", "green", 380, 300, 150), o("cube", "purple", 400, 200, 110)]
    w = world([*cubes, o("rod", "red", 420, 320, elong=5.0)], c)
    r.start(w, 0.0)
    r.stage_index = 1
    s = r.next_step(w, 0.0)
    assert s.kind == "place_cube" and "purple cube on top of the blue" in s.instruction  # cubes first
    r.placed += ["cube:purple", "cube:green"]
    s = r.next_step(w, 1.0)
    assert s.kind == "rod_on_tower" and "on top of the tower" in s.instruction  # then the rod, on the tower


def test_claude_must_confirm_the_unplug():
    c = cal()
    r = Relay(MatchConfig(own_color="orange", calibration=c))
    r.confirm_kinds = {"unplug"}
    before = world([o("plug", "orange", 50, 50)], c)
    after = world([], c)
    r.start(before, 0.0)
    r.next_step(before, 0.0)
    out, t = run_step(r, [before, after, after, after, after, after], 0.0, [CLOSED] + [OPEN] * 5)
    assert out is Outcome.RUNNING and r.pending_confirm == "unplug" and r.stage == "unplug"
    r.resolve_confirm(False, "an arm hides the strip", t)
    assert r.stage == "unplug" and r.pending_confirm is None
    out, t = run_step(r, [after] * 4, t)
    r.resolve_confirm(True, "a plug lies off the strip", t)
    out, t = run_step(r, [after] * 4, t)
    assert out is Outcome.PASSED and r.stage == "stack"


def test_failed_rod_on_table_switches_to_rod_on_tower():
    c = cal()
    m = MatchConfig(own_color="orange", calibration=c, step_timeout_s=5)
    r = Relay(m)
    cubes = [o("cube", "blue", 250, 250, 60), o("cube", "green", 380, 300, 150), o("cube", "purple", 400, 200, 110)]
    w = world([*cubes, o("rod", "red", 420, 320, elong=5.0)], c)
    r.start(w, 0.0)
    r.stage_index = 1
    t, kinds = 0.0, []
    for _ in range(5):
        s = r.next_step(w, t)
        kinds.append(s.kind)
        while r.observe(w, t, OPEN) is Outcome.RUNNING:  # the rod never stands up: each try times out
            t += 0.5
    assert kinds[:3] == ["rod_in_cube"] * 3  # three tries on the table (retries = 2)
    assert kinds[3] == "place_cube" and "rod_on_table" in r.abandoned  # then cubes first...
    assert "with the red rod" not in r.next_step(w, t).instruction  # ...and the top cube carries no rod


def test_plans_from_the_go_frame_when_the_arm_hides_cubes():
    c = cal()
    r = Relay(MatchConfig(own_color="orange", calibration=c))
    cubes = [o("cube", "blue", 250, 250, 60), o("cube", "green", 380, 300, 150), o("cube", "purple", 400, 200, 110)]
    go = world([*cubes, o("rod", "red", 420, 320, elong=5.0)], c)
    r.start(go, 0.0)
    r.stage_index = 1
    hidden = world([cubes[0]], c)  # the arm covers two cubes when stage 2 starts
    r.next_step(hidden, 1.0)
    assert r.stack_order == ["blue", "purple", "green"] and abs(r.plan_height_in() - 6.6) < 1e-9


def test_puts_the_pulled_cable_down_before_stacking():
    c = cal()
    r = Relay(MatchConfig(own_color="orange", calibration=c))
    cubes = [o("cube", "blue", 250, 250, 60), o("cube", "green", 380, 300, 150), o("cube", "purple", 400, 200, 110)]
    before = world([o("plug", "orange", 50, 50), *cubes, o("rod", "red", 420, 320, elong=5.0)], c)
    after = world([*cubes, o("rod", "red", 420, 320, elong=5.0)], c)
    r.start(before, 0.0)
    r.next_step(before, 0.0)
    r.observe(after, 0.5, CLOSED)  # the gripper holds the pulled cable...
    assert r.external_pass(after, 1.0, "a plug is out")  # ...when the unplug passes
    s = r.next_step(after, 1.0)
    assert r.stage == "stack" and s.kind == "put_down" and "orange USB cable down" in s.instruction
    out, t = run_step(r, [after, after, after], 1.0, [CLOSED, OPEN, OPEN])
    assert out is Outcome.PASSED and r.holding is None
    assert r.next_step(after, t).kind == "rod_in_cube"  # only now the tower starts


def test_trained_unplug_text_is_sent_verbatim_on_every_try():
    c = cal()
    r = Relay(MatchConfig(own_color="orange", calibration=c, trained_unplug="Remove Orange Cables from their port"))
    w = world([o("plug", "orange", 50, 50), o("cube", "blue", 250, 250)], c)
    r.start(w, 0.0)
    texts = {r.next_step(w, 0.0).instruction}
    t = 0.0
    for _ in range(400):
        t += 0.5
        if r.observe(w, t, OPEN) is not Outcome.RUNNING:
            step = r.next_step(w, t)
            if step and step.kind == "unplug":
                texts.add(step.instruction)
    assert texts == {"Remove Orange Cables from their port"}


def test_plan_stages_per_stage_models_and_texts():
    from harness.__main__ import plan_stages
    base = "dreamlabs/dream-1-beta"
    models, texts = plan_stages(base, {"unplug": "dream-1-ft:aa-cables", "stack": None, "balls": "dream-1-ft:bb-balls"},
                                {"unplug": None, "stack": None, "balls": None})
    assert models == {"unplug": "dream-1-ft:aa-cables", "stack": base, "balls": "dream-1-ft:bb-balls"}
    assert set(texts) == {"unplug", "balls"}  # the base-model stack keeps its step-by-step plan
    # One all-in-one fine-tune: every stage on it, each with its recorded text; "" turns one off.
    models, texts = plan_stages("dream-1-ft:cc-all", {}, {"stack": ""})
    assert set(models.values()) == {"dream-1-ft:cc-all"} and set(texts) == {"unplug", "balls"}


def test_stack_all_passes_only_with_claude_and_then_moves_to_balls():
    c = cal()
    m = MatchConfig(own_color="orange", calibration=c, stage_texts={"stack": "Stack the green, purple, and blue blocks"})
    r = Relay(m)
    w = world([o("cube", "blue", 250, 250), o("cube", "green", 300, 250), o("ball", "orange", 100, 300)], c)
    r.start(w, 0.0)
    r.stage_index = 1  # unplug already passed
    step = r.next_step(w, 0.0)
    assert step.kind == "stack_all" and step.instruction == "Stack the green, purple, and blue blocks"
    t = 0.0
    for _ in range(60):  # without Claude: never claims the stage
        t += 0.5
        assert r.observe(w, t, OPEN) is Outcome.RUNNING
    r.confirm_kinds = {"stack_all"}
    for _ in range(4):
        t += 0.5
        r.observe(w, t, OPEN)
    assert r.pending_confirm == "stack_all"
    r.resolve_confirm(True, "tall tower", t)
    t += 0.5
    assert r.observe(w, t, OPEN) is Outcome.PASSED and r.stage == "balls"


def test_default_run_uses_the_team_fine_tunes_with_their_own_sentences():
    from harness.__main__ import DEFAULT_STAGE_MODELS, MODEL_TEXTS, plan_stages
    models, texts = plan_stages(None, {}, {})
    assert models == DEFAULT_STAGE_MODELS
    assert texts == {st: MODEL_TEXTS[models[st]] for st in models}
    # Swapping the balls model swaps its sentence too.
    models, texts = plan_stages(None, {"balls": "dream-1-ft:39478655-nuck3-balls-basket"}, {})
    assert texts["balls"] == "Place yellow, blue, green, and orange balls into basket."
    # Base Dream-1 everywhere: the step-by-step plan, no trained sentences.
    models, texts = plan_stages("dreamlabs/dream-1-beta", {}, {})
    assert set(models.values()) == {"dreamlabs/dream-1-beta"} and texts == {}


def test_put_down_runs_on_the_cable_model_with_its_own_sentence():
    c = cal()
    m = MatchConfig(own_color="orange", calibration=c,
                    stage_texts={"unplug": "Remove Orange Cables from their port", "stack": "Stack the blocks"})
    r = Relay(m)
    w = world([o("plug", "orange", 50, 50), o("cube", "blue", 250, 250)], c)
    r.start(w, 0.0)
    step = r.next_step(w, 0.0)
    r._last_grip = {"left": 1.0, "right": 0.1}  # the cable still in the right gripper
    r._on_pass(step, w, 1.0)
    assert r.stage == "stack" and r.holding
    put = r.next_step(w, 1.0)
    assert put.kind == "put_down" and put.stage == "unplug"  # the executor keeps the cable model for it
    assert put.instruction == "Remove Orange Cables from their port"
    assert r.observe(w, 2.0, OPEN) is Outcome.RUNNING
    assert r.observe(w, 2.5, OPEN) is Outcome.PASSED  # gripper open twice: put down
    nxt = r.next_step(w, 3.0)
    assert nxt.stage == "stack" and nxt.instruction == "Stack the blocks" and r.holding_stage is None


def test_all_colors_counts_every_ball_and_takes_none_back_out():
    c = cal()
    m = MatchConfig(own_color="orange", calibration=c, all_colors=True,
                    stage_texts={"balls": "Place all balls in the basket. Push away all white balls."})
    r = Relay(m)
    other = next(col for col in c.ball_colors if col != "orange")
    # an other-colour ball already in the bin, an own ball on the table next to a white one
    w = world([o("ball", other, 550, 50), o("ball", "orange", 100, 300), o("avoid_ball", "white", 108, 300)], c)
    r.start(w, 0.0)
    r.stage_index = 2
    step = r.next_step(w, 0.0)
    assert step.kind == "ball"  # not "take_out", and the white neighbour doesn't block it
    w2 = world([o("ball", other, 550, 50), o("ball", "orange", 560, 60), o("avoid_ball", "white", 108, 300)], c)
    t = 0.5
    assert r.observe(w, t, {"left": 0.1, "right": 1.0}) is Outcome.RUNNING  # grabbed
    for _ in range(4):
        t += 0.5
        out = r.observe(w2, t, OPEN)  # let go over the bin
        if out is not Outcome.RUNNING:
            break
    assert out is Outcome.PASSED and r.own_in_bin == 2 and r.other_in_bin == 0


def test_keeps_pulling_cables_until_the_time_set():
    c = cal()
    m = MatchConfig(own_color="orange", calibration=c, pull_cables_until_s=60.0,
                    stage_texts={"unplug": "Remove Orange Cables from their port", "stack": "Stack the blocks"})
    r = Relay(m)
    w = world([o("plug", "orange", 50, 50), o("cube", "blue", 250, 250)], c)
    r.start(w, 0.0)
    first = r.next_step(w, 0.0)
    r._last_grip = dict(OPEN)
    r._on_pass(first, w, 20.0)  # Claude saw the first cable out at 20 s
    assert r.stage == "unplug"
    more = r.next_step(w, 20.0)
    assert more.kind == "unplug_more" and more.instruction == first.instruction  # same words: Dream carries on
    t = 20.0
    while True:
        t += 0.5
        out = r.observe(w, t, OPEN)
        if out is not Outcome.RUNNING:
            break
    assert out is Outcome.PASSED and r.stage == "stack" and 59.0 <= t <= 61.0
    assert r.next_step(w, t).instruction == "Stack the blocks"


def test_trained_ball_model_only_ever_gets_its_own_sentence():
    c = cal()
    text = "Place yellow, blue, green, and orange balls into basket."
    r = Relay(MatchConfig(own_color="orange", calibration=c, all_colors=True, stage_texts={"balls": text}))
    other = next(col for col in c.ball_colors if col != "orange")
    empty = world([], c)  # no ball seen: the old code parked with new words
    in_bin = world([o("ball", other, 550, 50), o("avoid_ball", "white", 560, 60)], c)  # the old code took them out
    r.start(empty, 0.0)
    r.stage_index = 2
    for w in (empty, in_bin):
        assert r.next_step(w, 1.0).instruction == text


def test_practice_can_start_at_the_stack_stage():
    c = cal()
    r = Relay(MatchConfig(own_color="orange", calibration=c, start_stage="stack",
                          stage_texts={"stack": "Stack the green, purple, and blue blocks"}))
    w = world([o("plug", "orange", 50, 50), o("cube", "blue", 250, 250)], c)
    r.start(w, 0.0)
    step = r.next_step(w, 0.0)
    assert r.stage == "stack" and step.kind == "stack_all" and "unplug" not in r.passed


def _colour_cal() -> Calibration:
    c = cal()
    c.ball_colors = ["orange", "green"]
    c.avoid_balls = ["white"]
    return c


def _balls_relay(c, own=None, t=0.0, w=None):
    r = Relay(MatchConfig(own_color=own, calibration=c))
    r.stage_index = 2
    r.start(w, t)
    return r


def test_no_colour_targets_every_ball_colour():
    c = _colour_cal()
    w = world([o("ball", "orange", 450, 50), o("ball", "green", 450, 150), o("avoid_ball", "white", 450, 250)], c)
    r = _balls_relay(c, None, w=w)
    s = r.next_step(w, 0.0)
    assert s.kind == "ball" and s.target == "orange"
    w2 = world([o("ball", "green", 450, 150), o("avoid_ball", "white", 450, 250)], c)
    r2 = _balls_relay(c, None, w=w2)
    s2 = r2.next_step(w2, 0.0)
    assert s2.kind == "ball" and s2.target == "green"
    w3 = world([o("avoid_ball", "white", 450, 250)], c)
    r3 = _balls_relay(c, None, w=w3)
    assert r3.next_step(w3, 0.0).kind == "park"  # white is never a target


def test_no_colour_never_takes_out_other_colour():
    c = _colour_cal()
    assert MatchConfig(own_color=None, calibration=c).other_color() is None
    w = world([o("ball", "green", 550, 50), o("ball", "orange", 450, 50)], c)
    r = _balls_relay(c, None, w=w)
    s = r.next_step(w, 0.0)
    assert s.kind == "ball"


def test_no_colour_score_counts_all_non_white():
    c = _colour_cal()
    w = world([o("ball", "orange", 550, 50), o("ball", "green", 550, 60)], c)
    r = _balls_relay(c, None, w=w)
    r.next_step(w, 0.0)
    r.own_in_bin, r.other_in_bin = r._own_in_bin(w), r._other_in_bin(w)
    assert r.own_in_bin == 2 and r.other_in_bin == 0
    assert r.score() == 10


def test_eight_ball_take_out_unchanged_without_colour():
    c = _colour_cal()
    w = world([o("eight_ball", "black", 550, 50), o("ball", "orange", 450, 50)], c)
    r = _balls_relay(c, None, w=w)
    s = r.next_step(w, 0.0)
    assert s.kind == "take_out" and "black ball" in s.instruction


def test_colour_mode_unchanged():
    c = _colour_cal()
    w = world([o("ball", "green", 550, 50), o("ball", "orange", 450, 50)], c)
    r = _balls_relay(c, "orange", w=w)
    s = r.next_step(w, 0.0)
    assert s.kind == "take_out" and s.target == "green"
    r.other_in_bin, r.own_in_bin = 1, 2
    assert r.score() == 5


def test_no_colour_excludes_avoid_balls_listed_in_ball_colors():
    c = _colour_cal()
    c.ball_colors = ["orange", "white"]  # white is a coin-flip colour AND a never-grab colour
    c.avoid_balls = ["white"]
    w = world([o("ball", "white", 450, 50)], c)
    r = _balls_relay(c, None, w=w)
    assert r._scoring_colors() == ["orange"]
    assert r.next_step(w, 0.0).kind == "park"  # white is never targeted


def test_no_colour_skipped_colour_falls_through_to_the_next():
    c = _colour_cal()
    w = world([o("ball", "orange", 450, 50), o("ball", "green", 450, 150)], c)
    r = _balls_relay(c, None, w=w)
    r.skipped.add("ball:orange")
    s = r.next_step(w, 0.0)
    assert s.kind == "ball" and s.target == "green"


def test_log_before_start_is_startup_at_zero():
    r = Relay(MatchConfig(own_color="orange", calibration=cal()))
    r.log(123456.0, "frozen before the arms opened")
    assert (r.events[0].t_s, r.events[0].stage) == (0.0, "startup")
    r.start(world([], cal()), 1000.0)
    r.log(1002.0, "later")
    assert r.events[-1].stage == "unplug" and r.events[-1].t_s == 2.0
