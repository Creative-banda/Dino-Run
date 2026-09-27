# Dino Run — Reinforcement Learning (PPO)

A **PPO** agent that learns to play the *existing* Pygame Dino Run game. The game itself was
not redesigned, re-skinned or re-implemented: the RL environment drives the same simulation
that `python main.py` runs, and `tests/test_physics_equivalence.py` proves the two are
frame-identical against the untouched original file.

Everything here is measured, not assumed. Where a number appears below it comes from a
command in this repository whose output is quoted next to it.

---

## 1. Quick start

```bash
# dependencies (pygame, gymnasium, stable-baselines3, numpy, psutil, torch)
python -m pip install -r requirements.txt

# 1) validate the whole pipeline (43 checks: headless, obs, reward, check_env, PPO, CUDA, render)
python training/smoke_test.py --timesteps 4096 --envs 4

# 2) measure how many parallel environments this machine can actually feed (writes models/benchmark_envs.json)
python training/benchmark_envs.py --envs 1,8,16,32,64,128 --steps 25000 --ppo-probe 3

# 3) train (--envs auto consumes the benchmark recommendation; GPU used for the policy if present)
python training/train_ppo.py --timesteps 3000000 --envs auto --frame-skip 4 --device auto --output models/ppo_dino_v1

# 4) evaluate against the untrained baselines, with fixed seeds
python training/evaluate_ppo.py --model models/ppo_dino_v1 --episodes 20 --seed 1234 --frame-skip 4

# 5) WATCH IT PLAY THE REAL GAME (normal Pygame window)
python training/play_agent.py --model models/ppo_dino_v1
```

`main.py` is unchanged from the player's point of view: `python main.py` still opens the same
game with the same menu, sprites, animations, parallax background, sounds and arrow keys.

---

## 2. Architecture / file map

```
project/
├── .gitignore                  # keeps caches/bundles/transient outputs out of the repo
├── main.py                     # the human game (now a thin entry point over game/human_play.py)
├── main_half_screen.py         # legacy half-screen variant of the original game (untouched)
├── requirements.txt
├── game/                       # THE GAME (existing gameplay, refactored so it can be driven headlessly)
│   ├── config.py               # GameConfig: resolution -> ratio_x/ratio_y/ratio, ground geometry
│   ├── actions.py              # the 3 actions + keyboard mapping (original semantics)
│   ├── clock.py                # Clock protocol: RealClock (play) / SimClock (RL)
│   ├── headless.py             # SDL dummy-driver bootstrap (no window, no audio, no keyboard)
│   ├── assets.py               # sprite/sound loading (visual vs headless)
│   ├── entities.py             # Player / Obstacle / Bird / Ground  (original physics, verbatim)
│   ├── core.py                 # DinoGame.step(action) == one iteration of the original main loop
│   ├── render.py               # parallax background + world drawing (unchanged visuals)
│   └── human_play.py           # windowed loop, menu, score/high-score, agent-playback hook
├── rl/
│   ├── environment.py          # DinoRunEnv (Gymnasium): reset/step/observation_space/action_space
│   ├── observations.py         # dino-obs-v1: the 47-float structured observation
│   ├── rewards.py              # RewardConfig / RewardFunction (documented, anti-exploit)
│   ├── wrappers.py             # frame_skip / episode stats / reward-scale wrappers
│   ├── vector_env.py           # make_vec_env (dummy | process | subproc) + benchmark recommendation
│   ├── process_vec_env.py      # multi-process backend with batched IPC (this project's own)
│   ├── env_worker.py           # torch-free worker: owns several envs, one round-trip per step
│   └── baselines.py            # random / do_nothing / geometric_heuristic / trained-model policies
├── training/
│   ├── benchmark_envs.py       # how many environments? measures raw + end-to-end PPO throughput
│   ├── train_ppo.py            # PPO training entry point (resumable, eval + checkpoints)
│   ├── evaluate_ppo.py         # evaluation vs baselines, fixed seeds, JSON report
│   ├── play_agent.py           # visual playback of a trained policy (real window)
│   ├── smoke_test.py           # the validation gate (43 checks)
│   ├── make_bundle.py          # builds + verifies the Colab zip
│   ├── artifacts.py            # model/run-config resolution, save/load helpers
│   └── resources.py            # CPU/RAM sampling used by the benchmark
├── colab/
│   ├── dino_run_ppo_colab.ipynb# the notebook (upload -> install -> benchmark -> smoke -> train -> evaluate)
│   └── README.md               # the same workflow as CLI commands
├── tests/
│   ├── test_physics_equivalence.py  # refactored sim vs reference/original_main.py, frame by frame
│   └── test_process_vec_env.py      # process backend vs single-process reference
├── reference/
│   └── original_main.py        # the original single-file game, kept ONLY as the equivalence baseline
└── models/
    ├── ppo_dino_v1/            # verified demo run (best/latest/evaluation.json/run_config.json)
    ├── ppo_dino_demo/          # the 3M-step run it was copied from (all checkpoints)
    └── benchmark_envs.json     # the machine benchmark report consumed by --envs auto
```

---

## 3. How the existing game was integrated (no rewrite)

The original game was a single 431-line script: module-scope `pygame.init()`, a `while running:`
loop, physics inline in `Player.move()`, spawning inline in the loop body, and `random` /
`pygame.time.get_ticks()` read directly. To make it drivable without a keyboard, display or wall
clock, three things are **injected** instead of edited:

| injected | human play (`main.py`) | RL (`DinoRunEnv`) |
|---|---|---|
| **clock** (`game/clock.py`) | `RealClock` wrapping `pygame.time.get_ticks()` | `SimClock`: one step = exactly one 60 Hz frame |
| **rng** | module-level `random` | per-environment `random.Random(seed)` |
| **action** (`game/actions.py`) | `action_from_keys()` reads `pygame.key.get_pressed()` | the discrete action the agent picked |

* `game/core.py::DinoGame.step(action)` performs exactly one iteration of the original loop —
  score increment, ground scrolling, the entity-arrival spawn check, obstacle/bird updates,
  player animation + physics + collision, death bookkeeping, speed increment — in the original
  order. No gameplay constant was changed: `gap = 40 + randint(0, 50)`, birds only after
  `score > 200` with 40 % probability, `+15` gap for birds, `game_speed = 4 * ratio_x` + `0.5 * ratio_x`
  every 10 s, jump `-14 * ratio_y`, gravity `0.8 * ratio_y`, fast-fall `5 * ratio_y`.
* `game/entities.py` holds `Player`/`Obstacle`/`Bird` verbatim, including the midbottom re-anchor in
  `change_animation()` that makes crouching genuinely shrink the hitbox (32 px standing → 28 px).
* `game/render.py` + `game/human_play.py` are the original drawing path; `main.py` is now
  `from game.human_play import main`. The menu, high-score file, sounds, music and the day/night
  parallax are untouched.
* **Headless** (`game/headless.py`): `SDL_VIDEODRIVER=dummy` + `SDL_AUDIODRIVER=dummy`, a 1×1
  display context (needed because sprites go through `convert_alpha()` and hitboxes come from the
  scaled images), no mixer, no fonts, no background layers, no `flip()`. Nothing opens a window and
  no key is ever read. `DinoRunEnv(render_mode="rgb_array")` can still render off-screen into a
  `pygame.Surface` for inspection.
* **Proof, not hope** — `tests/test_physics_equivalence.py` runs the original `reference/original_main.py`
  and the refactored simulation side by side, driving both with the same action script, and compares
  *every frame*: player rect, velocity, `Inair`, `ducking`, gravity, score, game speed, obstacle and
  bird rects/types, spawn counts and the death frame.

  ```
  === mode=random ===    original frames: 181, refactored frames: 181   OK - 181 frames identical
  === mode=heuristic === original frames: 2500, refactored frames: 2500 OK - 2500 frames identical
  EQUIVALENCE TEST PASSED: refactored simulation is frame-identical to the original.
  ```

**The one intentional deviation** (documented in `game/core.py`): `last_increment_time` is re-based on
every `reset()`. In the original it was captured at import, so idling in the menu for >10 s made the
first speed-up fire immediately. Re-basing makes episodes reproducible (speed-ups land on frames
600/1200/…) — required for a deterministic RL environment, and it makes the *first* 10 s no harder
than the original's first 10 s in a quick play session.

---

## 4. Observation space (`dino-obs-v1`, 47 floats, `float32`)

`gymnasium.spaces.Box(shape=(47,), low=..., high=...)` — fixed shape regardless of how many
obstacles exist (empty slots are zero-padded). Everything is **relative to the player** and divided
by a resolution-independent constant, so a model trained at 800×600 also plays the desktop-sized
window of `main.py`. No feature encodes the correct action.

| # | feature | definition |
|---|---|---|
| **player** | | |
| 0 | `player_above_ground` | `(ground_top - player.bottom) / max_jump_height` (≈0 on the ground, 1 at apex) |
| 1 | `player_velocity_y` | `velocity_y / jump_speed` (negative = rising) |
| 2 | `player_in_air` | 0/1 |
| 3 | `player_crouching` | 0/1 |
| 4 | `player_fast_falling` | 0/1 — "down" is currently engaged while airborne |
| 5 | `player_height` | `rect.h / window_height` (crouch shrinks it) |
| 6 | `player_width` | `rect.w / window_width` |
| **game** | | |
| 7 | `game_speed` | `game_speed / (3 * base_speed)` |
| 8 | `game_score` | frames survived / 6000 |
| 9 | `game_speed_progress` | 0…1 within the current 10 s difficulty interval |
| 10 | `game_entities_ahead` | `min(#entities ahead / 6, 1)` |
| **slot 0…3** (the 4 nearest entities ahead, nearest first) | | |
| +0 | `present` | 1 if this slot holds an entity |
| +1 | `is_bird` | 1 for a pterodactyl, 0 for a box |
| +2 | `box_variant` | box sprite 1…6 → 0…1 (0 for birds) |
| +3 | `dx` | `(entity.left - player.right) / window_width` — the gap in front of the dinosaur |
| +4 | `top_above_ground` | `(ground_top - entity.top) / window_height` |
| +5 | `bottom_above_ground` | `(ground_top - entity.bottom) / window_height` |
| +6 | `width` | `entity.rect.w / window_width` |
| +7 | `height` | `entity.rect.h / window_height` |
| +8 | `contact_frames` | frames until the entity reaches the player *at its own speed* (birds move 5 px/frame faster), /240 |

All values are clipped into the declared `Box` bounds. Geometry only: "is this box tall", "is this
bird high or low", "how many frames until it arrives" — the *decision* (jump / crouch / do nothing) is
what the network has to learn. Four slots give the agent look-ahead instead of pure reaction; a box
typically becomes visible ~12+ frames before contact at base speed, and fewer at higher speeds,
which is exactly the information the `dx`/`contact_frames` features carry.
`rl/observations.py::feature_names()` returns the same list programmatically.

---

## 5. Action mapping (`Discrete(3)`)

| action | meaning | game effect |
|---|---|---|
| 0 | nothing | no input |
| 1 | jump | `velocity_y = -14 * ratio_y`, only if `not Inair` |
| 2 | down | **on the ground → crouch** (`ducking = True`, hitbox 32 → 28 px); **in the air → fast-fall** (`gravity = 5 * ratio_y` instead of `0.8`) |

The dual meaning of "down" is the game's own mechanic and is preserved exactly (see
`game/actions.py`). The agent must therefore learn jump, crouch *and* the airborne fast-fall; a
short hop is executed as `jump` then `down` on a later frame (the smoke test asserts all three
behaviours, including `gravity 0.80 -> 5.00` mid-air).

A human holding *both* keys in one frame gets "jump + immediate fast-fall", which a discrete action
cannot express; the agent reaches the same physics by issuing `down` on the next frame.

`frame_skip` (default 1; training used 4) decides how many game frames one agent action covers. One
action per game frame is the human-equivalent setting; at `frame_skip=4` the agent decides 15×/s and
a step costs 4 simulation frames.

---

## 6. Reward formula

```
normal frame:  r_t = scale * ( advance * (game_speed / base_speed) + survival )
collision:     r_T = r_t + scale * death
total:         R   = Σ_t r_t  +  (scale * death if the episode ended by collision)
```

Defaults (`rl/rewards.py`): `advance = 0.1`, `survival = 0.0`, `death = -10.0`, `clear_bonus = 0.0`,
`scale_advance_by_speed = True`, `scale = 1.0`. Optional `clear_bonus` (per obstacle safely passed)
defaults to 0 — `RewardConfig` exposes it but it adds nothing the distance term does not already give.

Why each term exists:

* **`advance * (game_speed / base_speed)`** — the reward per frame is literally the distance the world
  scrolled this frame, so total return is monotone in survival time, and surviving *further* into the
  faster part of the level pays more per frame. This is the only positive term, which is what makes
  the reward un-exploitable.
* **`survival`** — optional flat per-frame bonus, left at 0 so that time and progress can be
  re-weighted later without code changes.
* **`death = -10`** — a collision ends the episode; the penalty makes dying clearly worse than one
  more frame (which pays ~0.1–0.5). It is *not* the mechanism that teaches survival — the lost future
  reward is — it just keeps the value function honest near the end of an episode.

There is **no reward for jumping, crouching, being airborne, or passing an obstacle by default**.
Jumping in front of nothing earns exactly what doing nothing earns (both merely advance), and jumping
*costs* ~35 frames of air time during which another jump is impossible, so spam is never optimal. The
smoke test verifies this (`jumping is not rewarded by itself: 0.1000 vs 0.1000`,
`the reward per survived frame does not depend on the action`, and
`surviving longer strictly dominates: heuristic 936.9 over 6000 frames vs do-nothing 8.1 / jump-spam 18.5`).

---

## 7. Episode handling

* An episode starts at `reset()` (the game's own start state) and ends **terminated** on collision.
* **Truncated** at `max_episode_steps` agent steps (default 6000 → 100 s of game time at
  `frame_skip=1`, or 400 s at `frame_skip=4`); the truncation gate guards against infinite episodes.
* Terminated and truncated are reported separately, and both carry an `info["dino_episode"]` summary
  (score, survival seconds, distance, reward, steps, collision flag, final speed, spawns, per-action counts).
* In a vector env a dead environment is auto-reset (SB3 `VecEnv` semantics); the other environments
  are unaffected, which `tests/test_process_vec_env.py` checks explicitly (`12 episodes`, per-env
  episode counts `[2,2,2,2,2,2]`).

---

## 8. Parallel environments: measured, not guessed

`training/benchmark_envs.py` sweeps environment counts and backends, samples CPU % and RSS, and then
runs **real PPO probes** (same hyperparameters as training) because with a tiny MLP the PPO update —
not the simulation — can dominate the wall clock.

Backends:

* `dummy` — SB3 `DummyVecEnv`: everything in one process (fast per step, no parallelism).
* `process` — **project-owned** `rl/process_vec_env.py`: N worker processes, each owning several
  environments, with **batched IPC** (one round-trip per vector step, not one per env). Workers live in
  `rl/env_worker.py`, which imports *no* torch/SB3, so they never pay the ~1 s torch import.
* `subproc` — SB3 `SubprocVecEnv`, kept for comparison.

Measured on the development machine (Windows 10, 12 logical / 6 physical cores, 16 GB RAM,
torch 2.12.0+cpu, **no CUDA**), 25 000 steps per configuration:

| envs | backend | steps/s | steps/s per env | CPU % | RSS MB |
|---|---|---|---|---|---|
| 1 | dummy | 16 066 | 16 066 | 99 | 220 |
| 8 | dummy | 25 846 | 3 231 | 102 | 220 |
| 8 | process | 17 718 | 2 215 | 89 | 610 |
| 16 | process | 30 462 | 1 904 | 88 | 611 |
| 32 | process | 46 108 | 1 441 | 96 | 756 |
| 64 | process | 72 067 | 1 126 | 69 | 758 |
| 128 | process | 111 715 | 873 | 54 | 762 |

Raw stepping keeps scaling, but end-to-end PPO throughput does not — hence the probes:

| envs | backend | PPO samples/s (end-to-end) | time in PPO update |
|---|---|---|---|
| 1 | dummy | 1 774 | 89 % |
| 16 | process | 6 154 | 80 % |
| **32** | **process** | **7 199** | 84 % |
| 128 | process | 7 499 | 93 % |

**Recommendation written to `models/benchmark_envs.json`:** `n_envs = 32`, backend `process`
(`96 %` of the best probed configuration, `7 199 samples/s`), chosen over 128 environments because
128 needs 4× the processes/RAM/start-up time for a ~4 % gain. `--envs auto` reads this file;
`--envs 16` / `--envs 64` override it, and a `--max-envs` flag caps the recommendation.

**Why 32 won and more did not help:** the simulation is cheap (16 k steps/s even single-threaded) and
the policy is a 14 724-parameter MLP, so from ~32 environments on the bottleneck is the gradient update
on 12 CPU threads (CPU 580 % at both 32 and 128 envs), not the game. Adding environments buys
throughput only until the update saturates; that is also why PPO on this machine is **update-bound**,
and why the same 32-env configuration is a sane default for Colab.

---

## 9. PPO configuration (Stable-Baselines3 + Gymnasium)

No PPO was written from scratch. `rl/environment.py` is a standard Gymnasium `Env`
(`check_env` passes with 0 warnings).

```python
PPO("MlpPolicy", venv, ...)
# policy_kwargs = {"net_arch": [64, 64]}     -> 47 -> 64 -> 64 -> (3 logits | 1 value), 14 724 params
learning_rate   = 5e-4      # (default in code 3e-4)
n_steps         = 512       # rollout length per environment
batch_size      = 512
n_epochs        = 6         # (default 10)
gamma           = 0.997     # long horizon: episodes can last thousands of frames
gae_lambda      = 0.95
clip_range      = 0.2
ent_coef        = 0.005
vf_coef         = 0.5
max_grad_norm   = 0.5
target_kl       = 0.05      # early-stop an update that drifts
normalize_advantage = True
```

Every value is a CLI flag (`--gamma`, `--ent-coef`, `--learning-rate`, `--net-arch`, …) and every
value is recorded in `models/<run>/run_config.json`, together with the observation definition
(`obs_version`, all 47 feature names), the action mapping, the reward config, the seed, the per-env
seed list and the environment count.

Why these values: `gamma = 0.99` with `frame_skip = 1` discounts the terminal penalty over ~180
frames so heavily that the gradient is dominated by entropy (`ent_coef`) — measured locally as "no
learning after 300 k steps". `gamma = 0.997` plus `frame_skip = 4` (one decision per 4 frames,
standard action-repeat) is what produced a policy that learns within ~2 M steps.

The device is selected by `--device auto` (`cuda` when `torch.cuda.is_available()`, else `cpu`), the
policy lives on the GPU while the simulation stays on the CPU, and the smoke test asserts the policy
device (`PPO policy is on the requested device`). `train_ppo.py` also forces torch to a single thread
per process when subprocess environments are used, so the workers do not oversubscribe the CPU.

---

## 10. Randomness and reproducibility

* `--seed` (default 42) seeds the vector env and the model.
* Each environment gets its own seed (`make_vec_env`: `env i → seed + i`), and each *episode* inside
  an environment derives a new seed (`seed * 1_000_003 + episode_index`), so no two environments — and
  no two successive episodes — replay the same obstacle sequence.
* `SimClock` removes wall-clock dependence entirely: frame 600 / 1200 / … are the speed-ups, always.
* The smoke test verifies `identical seeds give identical episodes` and
  `different seeds give different obstacle sequences`.
* Evaluation uses fixed seeds (`--seed 1234`), so runs are comparable.

---

## 11. Checkpoints and resume

```
models/ppo_dino_v1/
├── run_config.json                  # everything needed to reproduce/resume this run
├── latest.zip                       # final state -> what --resume continues from
├── best/best_model.zip              # best evaluated policy -> what evaluate/play load
├── checkpoints/dino_ppo_*_steps.zip # periodic snapshots (--checkpoint-every)
└── evaluations.npz / evaluation.json
```

`--resume models/ppo_dino_v1/latest.zip` (or the directory) continues training and keeps the
timestep count. Two footguns were found and fixed while doing this:

1. SB3's `EvalCallback` restarts from `-inf` on every construction, so resuming into the same output
   directory overwrote a good `best/best_model.zip` with a worse policy as soon as the continued
   policy produced its first evaluation. `training/train_ppo.py::PersistentEvalCallback` persists the
   best reward in `run_config.json` and restores it, so the best model on disk only ever improves.
2. Resuming *degraded* this policy (measured, not theoretical). Keep `latest.zip` for continuing and
   treat `best/best_model.zip` as the artefact to ship; the run config records both.

Nothing depends on the training process staying alive: all state is on disk, and the Colab notebook
re-runs the training cell by resuming from `latest.zip`.

---

## 12. Verified results (local, CPU-only machine)

Training run: **3,000,000 agent steps, 32 environments, `frame_skip = 4`, seed 42**, 16 m 51 s wall
clock (~3 000 samples/s *including* the evaluation callbacks — the pure stepping probe rate was
7 199 samples/s).

Evaluation — `python training/evaluate_ppo.py --model models/ppo_dino_v1 --episodes 20 --seed 1234
--frame-skip 4`, 20 fixed-seed episodes per policy:

| policy | mean score | median | best | worst | mean survival | collision rate | mean reward |
|---|---|---|---|---|---|---|---|
| **PPO (trained)** | **5 192** | 520 | 24 000 (cap) | 200 | 86.5 s | 80 % | 1 679.4 |
| random actions | 189 | 182 | 267 | 180 | 3.1 s | 100 % | 8.9 |
| do nothing | 181 | 181 | 181 | 181 | 3.0 s | 100 % | 8.1 |
| geometric heuristic (scripted) | 12 331 | 12 454 | 24 000 (cap) | 482 | 205.5 s | 50 % | 4 148.7 |

* **versus random: 27.5× the mean score** (189 → 5 192); versus do-nothing 28.7×.
* Action mix: `down 83.0 %`, `nothing 12.9 %`, `jump 4.1 %` — the agent drives almost everything with
  the fast-fall/crouch action, which is the actual learned strategy (a jump followed by a short
  fast-fall hop onto the ground, plus crouching under high birds).
* **Honest caveat:** the distribution is bimodal. 4 of 20 episodes survive the full 400 s cap
  (scores of 24 000, i.e. the episode limit, not a death), the other 16 die after ~500 frames
  (~8 s) — that is why the median (520) is far below the mean. A traced episode shows the failure
  mode: the fast-fall "low hop" is timed for the higher game speeds and mistimes the very first boxes
  at base speed. The scripted geometric heuristic (a hand-written rule-based player that reads the
  obstacle geometry directly) still beats the agent's mean, 12 331 vs 5 192.
* So: **the system demonstrably learns and plays far beyond random**, and it is not yet superhuman.
  More timesteps, and/or removing the low-speed failure mode, is the obvious next step.

---

## 13. Watching it play (the normal visual game)

```bash
python training/play_agent.py --model models/ppo_dino_v1              # real window, 3 episodes
python training/play_agent.py --model models/ppo_dino_v1 --episodes 0 # forever
python training/play_agent.py --model models/ppo_dino_v1 --stochastic --no-info
python training/play_agent.py --model models/ppo_dino_best            # or a .zip / any output dir
```

This opens the same window, sprites, animations, parallax, sounds and score display as `python main.py`,
drives the dinosaur from the loaded policy, prints each episode's survival time and shows a small
agent overlay (episode number, mean score, current action, score). It works at the desktop resolution
(the observation is resolution-invariant) or with `--resolution 800x600`.

**Pacing.** By default playback is **paced exactly like `python main.py`**: one game frame per drawn
frame, capped at the game's designed `60 * ratio_y` frames/s. Whatever speed you see in the human game at
a given window size is the speed you see here, on the same machine.

The game's *designed* rate is not always reachable: at 2560x1440 it is 144 frames/s, which means 144
full-window renders of five alpha-blended parallax layers, and no CPU can draw that. `python main.py`
therefore runs below its own design on a large window (that is the speed players know as "normal"), and
`--realtime` is the opt-in that instead keeps the world on the game's clock and drops drawn frames, so
the agent plays at the designed rate even when the window is too big to render it:

| window | one drawn frame | draw ceiling | designed rate | `main.py` / default | `--realtime` |
|---|---|---|---|---|---|
| 800x600 | 2.3 ms | 433 fps | 60 fps | 1.00x | 1.00x |
| 1280x720 | 5.1 ms | 198 fps | 72 fps | 1.00x | 1.00x |
| 2560x1440 | 20.2 ms | 49 fps | 144 fps | 0.32x (same as `main.py`) | **1.00x** |

Measured with the trained agent over 10 s of game time at 2560x1440: default `10.00 s / 30.99 s wall =
0.32x of the designed rate` (i.e. what `main.py` does), `--realtime` `10.01 s / 10.06 s wall = 1.00x`.
At the small windows the two modes are indistinguishable (verified `1.00x` at 800x600 and 1280x720).

Pacing never changes the game itself: every simulated frame is bit-identical either way and the agent
gets the same observation per frame, so it makes exactly the same decisions. Only how many of those
frames reach the screen per real second differs.

```bash
python training/play_agent.py --model models/ppo_dino_v1 --selftest --frames 12000   # no window
python training/play_agent.py --model models/ppo_dino_v1 --realtime     # designed 144 frames/s at 4K
python training/play_agent.py --model models/ppo_dino_v1 --resolution 1280x720  # full speed, no drops
```
`--selftest` runs the identical render loop against an off-screen surface (verified: `frames run:
1,500 | episodes finished: 3 | best score: 835`), which is how the render path is checked on a machine
without a display.

---

## 13b. Rendering performance (a real bug that was found and fixed)

The parallax background is five full-window layers blitted every frame, so its cost dominates
everything else (measured at 2560x1440: 162 ms of a 164 ms frame; the model itself costs 0.6 ms).

The cause was in `game/assets.py`: images were loaded with `pygame.image.load()` and scaled **without**
`convert()` / `convert_alpha()`, so every scaled surface kept the PNG file's pixel format
(channel order `0xff, 0xff00, 0xff0000` instead of the display's `0xff0000, 0xff00, 0xff`). Pygame then
falls back to a per-pixel format conversion on *every blit*: one full-window layer cost **33.5 ms** as
loaded versus **2.6 ms** after conversion — a 13x penalty. The original script called
`.convert_alpha()` on every sprite, so this was a fidelity regression introduced by the refactor, not a
game property.

`game/assets.py::to_display_format()` now matches each image to the display format (`convert_alpha()`
when the file has an alpha channel, plain `convert()` when it is fully opaque — neither changes a single
pixel). Effect on the whole frame at 2560x1440: **166 ms → 20 ms** (8.5x), and at the RL geometry
800x600 the frame now costs 2.3 ms instead of 22.5 ms. This speeds up the human game too, at every
resolution.

`game/render.py::update_parallax_background(steps=1)` accepts a step count so that when playback
simulates several frames at once (see §13) the background is advanced the same number of frames and
still drawn exactly once — i.e. it never lags behind the entities. With `steps=1` it is bit-identical
to the original arithmetic.

Pacing note (`game/human_play.py::play(catch_up=...)`): the number of frames simulated per loop
iteration is the number the game clock says are *due*, and it is allowed to be **zero**. Clamping it to
a minimum of one simulated frame per iteration was a bug: whenever the renderer was faster than the
game clock (e.g. 430 fps at 800x600), the world was fast-forwarded to the render rate. Both failure
modes are now covered by measurement (see the table in §13): too few frames simulated = slow motion,
too many = fast forward. `catch_up` defaults to False, so `main.py` and `play_agent.py` share the
original one-frame-per-drawn-frame pacing.

---

## 14. Google Colab

```bash
python training/make_bundle.py --verify      # -> dist/dino_run_rl_colab.zip (3.42 MB, 90 files)
```

The bundle contains the game, the assets, `rl/`, `training/`, `tests/`, `colab/`, `reference/`, the
already-trained `models/ppo_dino_v1` and the benchmark report. `--verify` extracts it into a temporary
directory and runs a real headless rollout **from inside the extracted copy**:

```
single env ok: 60 steps, 1 episode ends, reward 13.70, last score 56, episode seed 7000021
4-env vector env ok
display: {'pygame': '2.6.1', 'sdl': '2.28.4', 'driver': 'dummy', 'headless': True, ...}
BUNDLE VERIFY OK
```

In Colab (T4 GPU runtime): upload the zip, then either open
`colab/dino_run_ppo_colab.ipynb` (it walks through machine check → extract → install → headless probe →
**benchmark** → **smoke test** → 200 k-step run → evaluate → the long resumable run → download), or run
the same commands from `colab/README.md`:

```bash
!unzip -q -o dino_run_rl_colab.zip -d /content/dino_rl && cd /content/dino_rl
!python -m pip install -q -r requirements.txt
!python -u training/benchmark_envs.py --envs 1,2,4,8,16,32 --steps 12000 --max-seconds 15 \
    --backends dummy,process --ppo-probe 2 --ppo-steps 8192 --json models/benchmark_envs.json
!python -u training/smoke_test.py --timesteps 4096 --envs 4 --device auto
!python -u training/train_ppo.py --timesteps 3000000 --envs 32 --backend process --frame-skip 4 \
    --n-steps 512 --batch-size 512 --n-epochs 6 --gamma 0.997 --ent-coef 0.005 --learning-rate 5e-4 \
    --eval-every 250000 --checkpoint-every 500000 --device auto --seed 42 --output models/ppo_dino_colab
```

`--envs auto` is preferred over the hard-coded 32 in Colab: it reads the recommendation from the
benchmark you just ran, which on a 2-vCPU Colab runtime will be smaller than 32. Long runs should
write to Google Drive (`--output /content/drive/MyDrive/...`) so checkpoints survive a disconnect,
and can be continued with `--resume .../latest.zip`.

**Not verified here:** the CUDA path could not be exercised on this machine (CPU-only torch). The code
selects the device (`--device auto`), places the policy on `cuda` when
`torch.cuda.is_available()`, and the smoke test's `CUDA is used for the policy` check is the gate that
proves it on a GPU runtime (it prints `SKIP` here).

---

## 15. Validation / testing (what was actually run)

| command | what it proves | result |
|---|---|---|
| `python tests/test_physics_equivalence.py` | refactored simulation == original game, frame by frame | **PASSED** (181 + 2500 identical frames) |
| `python tests/test_process_vec_env.py` | multi-process backend == single-process reference, env isolation, no leaked children | **PASSED** (11/11) |
| `python training/smoke_test.py --timesteps 4096 --envs 4` | headless, obs, actions (jump/crouch/fast-fall), reward anti-exploit, `check_env`, vector envs, PPO update + policy change, checkpoint round-trip, CUDA, render | **PASSED** (43/43, 1 skipped: CUDA) |
| `python training/make_bundle.py --verify` | the Colab zip is self-contained | **PASSED** |
| `python training/benchmark_envs.py ...` | how many environments this machine can feed | 32 (process) recommended |
| `python training/evaluate_ppo.py ...` | the agent beats the untrained baselines | 27.5× random, 0.4× the scripted heuristic |
| `python training/play_agent.py --selftest --frames 12000` | the visual playback path runs off-screen | **PASSED** |

---

## 16. Known limitations

* The trained policy is bimodal (see §12): reliable at speed, mistimed at base speed.
* `frame_skip=4` means the agent cannot make a decision on every game frame; that is what made
  learning feasible, at the price of a coarser control granularity than a human has.
* The reward has no "pass the obstacle" term by design; if the low-speed mistiming turns out to be a
  credit-assignment problem rather than a data problem, a small, non-exploitable progress bonus is the
  first knob to reach for (it is already available via `RewardConfig.clear_bonus`).
* Multi-process training ramps to ~760 MB RSS and ~6 busy cores on this machine; on a 2-vCPU Colab
  runtime the recommendation will be smaller, and `--backend dummy` is the fallback if process
  spawning is unavailable.
* The scripted `geometric_heuristic` baseline is intentionally strong; it is the bar for the next run.
