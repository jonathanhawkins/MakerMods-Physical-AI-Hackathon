# MakerMods Physical AI Hackathon: Hill Climb harness

Our entry for the **Hill Climb** track of the MakerMods Physical AI Hackathon (Nov 7–8, 2026, Founders, Inc. SF Lab;
https://luma.com/pah). The track is ranked only by eval score: a bimanual 6-axis MakerMods arm station has 5 minutes,
with no human input after "go", to run a three-stage relay:

1. **Unplug** the orange cables from the power strip (5 points per cable).
2. **Stack** the green, purple and blue blocks into a free-standing tower.
3. **Balls:** put the coloured ping-pong balls in the basket, never grabbing the white ones.

We fine-tuned Dream-1 on our own teleop demos, one model per stage, and built a harness in `harness/` that runs the
relay unattended on station NUCK-3. The harness picks each stage's model, judges each stage with Claude vision, and
recovers when something drops out.

**Best result:** a full relay through all three stages (cables out, tower passed at 195 s, then the balls stage until
time ran out), scoring 30. Stacking alone passed reliably once the blocks started close together, and a balls-only
run put 2 balls in.

## How the harness works

```
 go ─▶ unplug (usb-cables model) ─▶ arms home ─▶ stack (stack-blocks model) ─▶ arms home ─▶ balls (balls-whites model)
          │                                         │                                         │
     Claude: a cable out                   Claude: 3 blocks stacked,               Claude: any balls left
     since the go frame?                   standing on their own?                  within reach?
```

- **One fine-tune per stage.** Each model is sent the exact sentence its demos were recorded with; a fine-tune
  ignores paraphrases. The harness homes the arms and switches clients between stages, and the put-down step stays
  on the model that is holding the object.
- **Claude vision referees** (`harness/claude_check.py`, `claude-opus-5-5`). It compares the overhead frame with the
  go frame for "a cable is out". It checks stacking in two steps: stacked while the gripper may still touch the
  tower, then standing on its own after the arms go home. In the balls stage it asks whether any ball is still
  reachable, and stops the stage when none is.
- **Gravity and height correction.** The demos were recorded with no gravity torque, so the arm sagged 4–7 cm
  below its commanded pose. The models learned those commanded targets, and Dream's runtime reaches them exactly,
  so every grasp landed high. `harness/gravity.py` scales the Metal arm's gravity compensation per stage (0 = as
  recorded). `harness/height_offset.py` lowers the right hand's targets near the table, with a 4 cm floor.
- **Warm models.** A fine-tune that has been idle for about an hour goes cold on Dream's servers and took up to
  12 minutes to come back. `harness/cloud_warm.py` warms models from saved inputs (no arms, no cameras). A run also
  warms the later stages' models in the background at go.
- **Recovery.** If a wrist camera drops off USB, the harness reopens the cameras and opens a fresh Dream client. If
  Dream stalls, it opens a new loop and carries on from the same step. If Dream refuses because a person is in the
  scene, it waits. `harness/autorestart.py` reruns the whole harness from the stage it stopped in, sharing what is
  left of the clock.
- **Freeze and end hooks** (`harness/freeze.py`): SIGUSR1 freezes the arms in place; SIGINT, SIGTERM and SIGHUP
  end the run. `state.json` in the run folder lets another process, such as a LiveKit voice agent, follow and
  control the run.

The overhead camera's colour detection (`harness/vision.py`, `harness/calibration.yaml`) handles timing and
targeting, but only orange and green balls are calibrated, so it undercounts yellow and blue balls in the basket.

## The fine-tunes

Five Dream-1 fine-tunes were trained from the team's teleop demos with the kit's `tools/finetune.py`. The model IDs
are `dream-1-ft:39478655-nuck3-<name>`:

| Name | Stage | Exact task text |
|---|---|---|
| `usb-cables` | unplug | `Remove Orange Cables from their port` |
| `stack-blocks` | stack | `Stack the green, purple, and blue blocks` |
| `balls-whites` | balls (used) | `Place all yellow, orange, green, and blue balls in the basket. Push away all white balls.` |
| `balls-basket` | balls (unused) | `Place yellow, blue, green, and orange balls into basket.` |
| `white-balls` | (unused) | `Move nearby white balls far away` |

`balls-basket` can't be used: Dream's server asks for a `video.overhead` camera, but the station sends the overhead
camera as `scene`.

## Running it

The harness runs on the station, in Dream's Python (`~/dream-env/bin/python`, from `~/team-harness`). **Anything
that opens the arms moves them, so keep a hand on the e-stop.**

Warm the models first (no arms):

```bash
~/dream-env/bin/python -W ignore -m harness.cloud_warm \
  dream-1-ft:39478655-nuck3-usb-cables dream-1-ft:39478655-nuck3-stack-blocks dream-1-ft:39478655-nuck3-balls-whites
```

Run a match, restarting from the last stage after a crash:

```bash
~/dream-env/bin/python -m harness.autorestart --color orange --time-limit 300 \
  --gravity-scale 0 --gravity-unplug 0.3 --lower-unplug-cm 3 \
  --stack-pass 3-cubes --all-colors --pull-cables-until 45 \
  --model-balls dream-1-ft:39478655-nuck3-balls-whites
```

`python -m harness run` takes the same options for a single run with no restarts. Other subcommands are `snapshot`
(overhead frame with detections drawn on, no arms), `detect` and `serve` (the relay's steps over HTTP, for DimOS).

| Option | What it does |
|---|---|
| `--start-stage {unplug,stack,balls}` | Practice: start at a later stage. |
| `--time-limit S` | Match clock in seconds (default 300). |
| `--pull-cables-until S` | Keep pulling cables until S seconds into the match; each cable scores. |
| `--gravity-scale`, `--gravity-<stage>` | Gravity compensation, overall and per stage (0–1.5). |
| `--lower-<stage>-cm` | Lower the right hand's targets near the table. At gravity 0 this adds to the sag. |
| `--model-<stage>`, `--<stage>-text` | Override a stage's model or its task text. |
| `--stack-pass {6in,3-cubes}` | What Claude must see to pass the stack. |
| `--all-colors` | Every ball colour except white and black scores. |

Settings that worked:
- **Cables:** gravity 0.3 with `--lower-unplug-cm 3`. At gravity 0 the arm is weak enough that pulling a cable
  trips the arm's safety stop.
- **Blocks and balls:** gravity 0. At 0.3 the gripper caught the top of a block and the block shot out.

## What we learned

- **Teleop demos sag.** Without gravity torque during recording, the commanded and measured hand heights differ by
  4–7 cm near the table. Correct for it at run time or record with gravity compensation on.
- **Fine-tunes need their exact sentence**, and a cold model can stall a stage for minutes. Keep them warm before a
  match.
- **Gravity 0 trades accuracy for strength.** A slow Dream answer (6–10 s) leaves the arm sagging against a blocked
  joint, and the safety stop fires after 5 s.
- **The layout matters as much as the model.** Stacking passed when the three blocks started close together within
  one arm's reach, and failed when they were spread out or an extra cube was on the table.
- **Wrist cameras are found by USB port** (`/dev/v4l/by-path/...` in `~/.dreamlabs/station.yaml`). Moving a cable to
  another port breaks the config, and a strained cable drops out mid-run.
- **A crash can leave a gripper motor powered.** The arm then refuses to open until its own power is cycled; a
  computer reboot doesn't clear it.

## Layout

| Path | What it is |
|---|---|
| `harness/` | The relay harness: stages and referee (`relay.py`), Dream-1 control loop (`dream_executor.py`), Claude checks, vision, gravity and height correction, warming, auto-restart, freeze hooks. Tests are `test_*.py` (pytest). |
| `dimos_hillclimb/` | The relay as Dimensional OS skills: DimOS's MCP server and LLM agent sequence the match on top of our referee and verifiers. See `dimos_hillclimb/README.md`. |
| `finetune/` | A YOLO ball-detector fine-tune from the station's episodes: pull, extract, auto-label, review in a web page, train, evaluate. |
| `sim/` | Our MuJoCo copy of the NUCK-3 station and a scripted-demo recorder. See `sim/README.md`. |
| `sim_data/` | Sim datasets (LeRobot v2.1, kit layout). |
| `save_dream_key.sh`, `save_anthropic_key.sh`, `station_install_key.sh` | Prompt for a key (hidden) and install it on the station. They contain no keys. |
| `repos/` | Upstream repos cloned by `setup.sh` (gitignored): portal-makermods, maker-arm-sdk, makermodslab, metal-arm-harness, metal-python-ros, dimos, and Dream Labs' hackathon kit. |

## Setup

```bash
./setup.sh
```

This clones the upstream repos into `repos/` if they are missing and installs each one. It needs Python 3.12 and
`uv`.

**Credentials:**
- **Dream:** `DREAM_API_KEY=dl_live_...` from https://platform.dreamlabs.ai (API Keys; shown once). On the station
  it lives in `~/.dreamlabs/credentials`.
- **Claude vision checks:** an Anthropic API key in `~/team-harness/.anthropic_key` on the station.
- **LiveKit**, for the voice agent: `LIVEKIT_URL`, `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET` in
  `repos/portal-makermods/.env`.

**Dream-1:** `pip install dreamlabs` gives `dreamcli`, and `pip install dreamlabs-client` gives `dream.DreamClient`.
Each call returns about 1.6 s of actions from camera images, robot state and a task string. Docs are at
https://platform.dreamlabs.ai/api-documentation.
