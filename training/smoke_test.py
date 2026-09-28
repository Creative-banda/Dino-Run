"""Short validation pipeline -- run this before any long training job.

It answers, with actual measurements instead of assumptions:

1. is pygame really running headless (dummy video driver, no window, no audio)?
2. does the environment reset/step correctly, with valid observations and rewards?
3. are the actions valid, and do they still mean jump / crouch / fast-fall?
4. does the reward function refuse to reward jumping or crouching by itself?
5. does ``check_env`` (stable-baselines3's environment checker) pass?
6. do many environments run concurrently and independently, and does the
   multi-process backend agree with the single-process one?
7. does PPO actually update the policy, and does the policy *change*?
8. does checkpoint save/load round-trip (so training can resume)?
9. is CUDA used for the policy when available (skipped on a CPU-only machine)?
10. do the visual-rendering code paths still work (the normal game and rgb_array)?

Usage::

    python training/smoke_test.py
    python training/smoke_test.py --timesteps 8192 --envs 4 --device auto
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from game.actions import ACTION_DOWN, ACTION_JUMP, ACTION_NOTHING  # noqa: E402
from game.config import rl_config  # noqa: E402
from game.headless import describe_display  # noqa: E402
from rl.environment import DinoRunEnv  # noqa: E402
from rl.rewards import RewardConfig, RewardFunction  # noqa: E402
from rl.vector_env import make_vec_env  # noqa: E402

RESULTS: List[Tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    RESULTS.append((name, bool(condition), detail))
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}{(' -- ' + detail) if detail else ''}", flush=True)
    return bool(condition)


def skip(name: str, detail: str = "") -> None:
    RESULTS.append((name, True, f"skipped: {detail}"))
    print(f"[SKIP] {name} -- {detail}", flush=True)


# --------------------------------------------------------------------------- 1
def test_headless() -> None:
    import pygame

    info = describe_display()
    check("pygame video driver is dummy (no window)", info["driver"] == "dummy", str(info))
    check("display mode is 1x1 (offscreen)", pygame.display.get_surface().get_size() == (1, 1))
    check("audio mixer is not initialised", not info["mixer_initialised"])
    check("no keyboard input is required", not pygame.key.get_focused())


# --------------------------------------------------------------------------- 2
def test_environment() -> DinoRunEnv:
    env = DinoRunEnv(seed=7)
    check("observation space is a fixed-size Box", env.observation_space.shape == (47,), str(env.observation_space.shape))
    check("action space is Discrete(3)", env.action_space.n == 3)
    obs, info = env.reset(seed=1000)
    check("reset returns a finite float32 observation", obs.shape == (47,) and obs.dtype == np.float32 and np.isfinite(obs).all())
    check("reset info contains the episode seed", info.get("seed") is not None, str(info))

    obs, reward, terminated, truncated, info = env.step(ACTION_NOTHING)
    check("step returns the gymnasium 5-tuple", isinstance(terminated, bool) and isinstance(truncated, bool))
    check("step returns a finite reward", np.isfinite(reward), f"reward={reward:.4f}")
    check(
        "step info reports score/distance/spawns",
        all(key in info for key in ("score", "distance", "obstacles_spawned", "birds_spawned")),
    )
    check("observation stays inside the declared Box", env.observation_space.contains(obs))

    # determinism
    env_a = DinoRunEnv(seed=7)
    env_b = DinoRunEnv(seed=7)
    obs_a, _ = env_a.reset(seed=1234)
    obs_b, _ = env_b.reset(seed=1234)
    same = np.array_equal(obs_a, obs_b)
    for _ in range(300):
        obs_a, ra, ta, tra, _ = env_a.step(ACTION_NOTHING)
        obs_b, rb, tb, trb, _ = env_b.step(ACTION_NOTHING)
        if ta or tb:
            break
    check("identical seeds give identical episodes", same and ra == rb and ta == tb, f"reward {ra:.3f} vs {rb:.3f}")
    # different seeds must produce different obstacle sequences (the first frames are
    # identical by construction: nothing has spawned yet)
    env_c = DinoRunEnv(seed=7)
    env_d = DinoRunEnv(seed=7)
    env_c.reset(seed=4321)
    env_d.reset(seed=9999)
    for _ in range(400):
        _, _, term_c, trunc_c, _ = env_c.step(ACTION_NOTHING)
        _, _, term_d, trunc_d, _ = env_d.step(ACTION_NOTHING)
        if (term_c or trunc_c) or (term_d or trunc_d):
            break
    same_scene = np.array_equal(env_c.builder.build(env_c.game), env_d.builder.build(env_d.game))
    check(
        "different seeds give different obstacle sequences",
        not same_scene or env_c.game.score != env_d.game.score,
        f"survived {env_c.game.score} vs {env_d.game.score} frames with the same no-op policy",
    )
    env_c.close()
    env_d.close()

    # truncation
    short = DinoRunEnv(seed=3, max_episode_steps=25)
    short.reset(seed=5)
    truncated_seen = False
    for _ in range(30):
        _, _, terminated, truncated, info = short.step(ACTION_NOTHING)
        if truncated:
            truncated_seen = True
            check("truncated episodes still report an episode summary", "dino_episode" in info)
            break
        if terminated:
            break
    check("max_episode_steps truncates instead of terminating", truncated_seen)
    short.close()
    env_a.close()
    env_b.close()
    return env


# --------------------------------------------------------------------------- 3
def test_actions(env: DinoRunEnv) -> None:
    env.reset(seed=2000)
    # step to the ground first
    for _ in range(40):
        env.step(ACTION_NOTHING)
    game = env.game
    check("player is on the ground before the action test", not game.player.Inair, f"rect={tuple(game.player.rect)}")

    env.step(ACTION_JUMP)
    check("action 1 (jump) launches the dinosaur", game.player.Inair and game.player.velocity_y < 0, f"vy={game.player.velocity_y:.2f}")
    env.step(ACTION_NOTHING)
    before = game.player.gravity
    env.step(ACTION_DOWN)
    check(
        "action 2 while airborne engages the fast fall",
        game.player.gravity > before and game.player.gravity > 0.8 * env.config.ratio_y,
        f"gravity {before:.2f} -> {game.player.gravity:.2f}",
    )

    env.reset(seed=2001)
    for _ in range(40):
        env.step(ACTION_NOTHING)
    env.step(ACTION_DOWN)
    check("action 2 on the ground crouches", game.player.ducking)
    env.step(ACTION_NOTHING)
    check("releasing down stops crouching", not game.player.ducking)

    # crouching must shrink the hitbox (the animation drives the rect)
    env.reset(seed=2002)
    for _ in range(40):
        env.step(ACTION_NOTHING)
    standing_h = game.player.rect.h
    env.step(ACTION_DOWN)
    check("crouching changes the collision box", game.player.rect.h < standing_h, f"{standing_h} -> {game.player.rect.h}")


# --------------------------------------------------------------------------- 4
def test_rewards(env: DinoRunEnv) -> None:
    reward_fn = RewardFunction(env.reward_config, base_game_speed=4.0 * env.config.ratio_x)
    env.reset(seed=3000)
    advance = reward_fn.frame_reward(env.game.game_speed)
    check("surviving one frame pays the advance reward", abs(advance - 0.1) < 1e-9, f"r={advance:.4f}")

    # jumping in an empty field must not pay more than doing nothing
    env.reset(seed=3001)
    _, noop_reward, _, _, _ = env.step(ACTION_NOTHING)
    env.reset(seed=3001)
    _, jump_reward, _, _, _ = env.step(ACTION_JUMP)
    check("jumping is not rewarded by itself", abs(noop_reward - jump_reward) < 1e-9, f"{noop_reward:.4f} vs {jump_reward:.4f}")

    # do nothing until the first box arrives -> collision must be punished
    env.reset(seed=3002)
    total = 0.0
    frames = 0
    for _ in range(1200):
        _, reward, terminated, truncated, _ = env.step(ACTION_NOTHING)
        total += reward
        frames += 1
        if terminated or truncated:
            break
    check("a collision terminates the episode", terminated, f"survived {env.game.score} frames")
    if terminated:
        check(
            "the death penalty is applied exactly once",
            abs(reward + reward_fn.frame_reward(env.game.game_speed) - 0.0) < 1e-6 or reward < -9.0,
            f"terminal frame reward {reward:.3f}",
        )

    # every survived frame must pay exactly advance * (speed / base_speed) -- the same
    # amount no matter which action was taken (that is what makes exploits pointless)
    sequences = {}
    formula_ok = True
    total_jumping = 0.0
    for tag, action_for in (
        ("noop", lambda index: ACTION_NOTHING),
        ("jump", lambda index: ACTION_JUMP if index % 2 == 0 else ACTION_NOTHING),
    ):
        env.reset(seed=3002)
        rewards = []
        for index in range(1200):
            _, reward, terminated, truncated, _ = env.step(action_for(index))
            if terminated or truncated:
                break
            rewards.append(reward)
            if index < 1199:
                total_jumping += reward if tag == "jump" else 0.0
            expected = reward_fn.frame_reward(env.game.game_speed)
            if abs(reward - expected) > 1e-9:
                formula_ok = False
        sequences[tag] = np.asarray(rewards)
    check("the per-frame reward follows the documented formula", formula_ok, f"r = 0.1 * game_speed / 4")
    # the two policies may survive a different number of frames (that is the physics),
    # but at any given simulation frame the reward must be identical
    overlap = min(len(sequences["noop"]), len(sequences["jump"]))
    check(
        "the reward per survived frame does not depend on the action",
        overlap > 0 and np.allclose(sequences["noop"][:overlap], sequences["jump"][:overlap]),
        f"first rewards {sequences['noop'][:3]} vs {sequences['jump'][:3]} "
        f"(survived {len(sequences['noop'])} vs {len(sequences['jump'])} frames)",
    )

    # the only way to earn more is to survive longer: a competent policy must score
    # far higher than both action-spamming and doing nothing
    from rl.baselines import GeometricHeuristicPolicy

    heuristic = GeometricHeuristicPolicy()
    env.reset(seed=3002)
    total_heuristic = 0.0
    rng = np.random.default_rng(0)
    for _ in range(6500):
        action = heuristic.act(env, None, rng)
        _, reward, terminated, truncated, info = env.step(action)
        total_heuristic += reward
        if terminated or truncated:
            break
    check(
        "surviving longer strictly dominates (no reward exploit beats playing)",
        total_heuristic > 5 * max(total, total_jumping),
        f"heuristic {total_heuristic:.1f} over {info['frames']} frames vs do-nothing {total:.1f} / jump-spam {total_jumping:.1f}",
    )


# --------------------------------------------------------------------------- 5/6
def test_vector_envs() -> None:
    from stable_baselines3.common.env_checker import check_env

    env = DinoRunEnv(seed=11)
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_env(env, warn=True)
    check("check_env passes without warnings", len(caught) == 0, f"{len(caught)} warnings")
    env.close()

    from rl.process_vec_env import ProcessGroupVecEnv
    from rl.vector_env import make_env

    n_envs = 4
    vec_dummy = make_vec_env(n_envs, seed=500, backend="dummy")
    vec_process = make_vec_env(n_envs, seed=500, backend="process", envs_per_worker=2)
    obs_d = vec_dummy.reset()
    obs_p = vec_process.reset()
    check("vector envs agree on the initial observations", np.allclose(obs_d, obs_p))
    seeds = [info.get("seed") for info in vec_dummy.reset_infos]
    check(
        "each environment has its own seed",
        len(set(seeds)) == n_envs and None not in seeds,
        str(seeds),
    )
    rng = np.random.default_rng(0)
    consistent = True
    for _ in range(200):
        actions = rng.integers(0, 3, size=n_envs)
        obs_d, r_d, done_d, _ = vec_dummy.step(actions)
        obs_p, r_p, done_p, _ = vec_process.step(actions)
        if not (np.allclose(obs_d, obs_p) and np.allclose(r_d, r_p) and np.array_equal(done_d, done_p)):
            consistent = False
            break
    check("multi-process backend is numerically identical to single-process", consistent)
    vec_dummy.close()
    vec_process.close()
    del ProcessGroupVecEnv


# --------------------------------------------------------------------------- 7/8/9
def test_ppo(timesteps: int, envs: int, device: str, output: Path) -> None:
    import torch

    from stable_baselines3 import PPO

    from training.train_ppo import build_ppo_kwargs

    vec = make_vec_env(envs, seed=900, backend="process" if envs > 1 else "dummy", envs_per_worker=2)
    kwargs = build_ppo_kwargs(n_steps=256, batch_size=128, n_epochs=4, net_arch=[64, 64])
    device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    model = PPO("MlpPolicy", vec, device=device, seed=900, verbose=0, **kwargs)
    check("PPO policy is on the requested device", str(model.policy.device).startswith(device), str(model.policy.device))
    if torch.cuda.is_available():
        check("CUDA is available and used for the policy", str(model.policy.device).startswith("cuda"))
    else:
        skip("CUDA is used for the policy", "no CUDA device on this machine (CPU fallback in use)")

    # a fixed probe state set to compare the policy before/after training
    probe_states = []
    probe_env = DinoRunEnv(seed=1)
    obs, _ = probe_env.reset(seed=42)
    rng = np.random.default_rng(1)
    for _ in range(64):
        obs, _, terminated, truncated, _ = probe_env.step(int(rng.integers(0, 3)))
        probe_states.append(obs.copy())
        if terminated or truncated:
            obs, _ = probe_env.reset()
    probe_env.close()
    # the probe tensor must live on the same device as the policy (cpu or cuda)
    probe_tensor = torch.as_tensor(np.asarray(probe_states), dtype=torch.float32, device=model.policy.device)

    def action_probs() -> np.ndarray:
        with torch.no_grad():
            dist = model.policy.get_distribution(probe_tensor)
            return dist.distribution.probs.cpu().numpy()

    before = action_probs()
    before_params = torch.cat([p.detach().flatten().clone() for p in model.policy.parameters()])
    started = time.time()
    model.learn(total_timesteps=timesteps, progress_bar=False)
    elapsed = time.time() - started
    after = action_probs()
    after_params = torch.cat([p.detach().flatten().clone() for p in model.policy.parameters()])
    param_delta = float((after_params - before_params).norm())
    policy_delta = float(np.abs(after - before).max())
    check("PPO ran and consumed the requested timesteps", model.num_timesteps >= timesteps, f"{model.num_timesteps} in {elapsed:.1f}s")
    check("network weights changed during training", param_delta > 0.0, f"|delta| = {param_delta:.4f}")
    check(
        "the policy actually changed (not just the weights)",
        policy_delta > 1e-4,
        f"max action-probability change {policy_delta:.4f}",
    )

    # checkpoint round trip
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "smoke_model.zip"
    model.save(str(checkpoint))
    reloaded = PPO.load(str(checkpoint), env=None if False else vec, device=device)
    reloaded_probs = None
    with torch.no_grad():
        reloaded_probs = reloaded.policy.get_distribution(probe_tensor).distribution.probs.cpu().numpy()
    check("checkpoint save/load reproduces the policy", np.allclose(after, reloaded_probs, atol=1e-6))
    check("reloaded model keeps its timestep count", int(reloaded.num_timesteps) == int(model.num_timesteps))

    from training.artifacts import resolve_model_path

    check("model path resolution accepts directories", resolve_model_path(output) == checkpoint, str(checkpoint))
    vec.close()


# --------------------------------------------------------------------------- 10
def test_rendering() -> None:
    import pygame

    env = DinoRunEnv(seed=5, render_mode="rgb_array")
    env.reset(seed=7)
    for _ in range(5):
        env.step(ACTION_NOTHING)
    frame = env.render()
    check(
        "rgb_array rendering works offscreen",
        isinstance(frame, np.ndarray) and frame.shape == (env.config.height, env.config.width, 3),
        str(None if frame is None else frame.shape),
    )
    env.close()

    # the human-playable game loop must still run (here: onto a dummy surface)
    from game.config import GameConfig
    from game.human_play import play

    config = GameConfig(width=640, height=480, write_high_score=False)
    stats = play(
        config=config,
        action_fn=lambda game, frame: ACTION_JUMP if frame % 30 == 0 else ACTION_NOTHING,
        max_frames=120,
        max_episodes=1,
        screen=pygame.Surface((config.width, config.height)),
        death_pause_ms=0,
        fps_limit=False,
    )
    check("the visual game loop still runs (parallax, sprites, score, fonts)", stats["frames"] >= 120, str(stats))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate the Dino Run RL pipeline end to end.")
    parser.add_argument("--timesteps", type=int, default=4096)
    parser.add_argument("--envs", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="models/smoke")
    args = parser.parse_args(argv)

    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    print("=" * 92)
    print("Dino Run RL smoke test")
    print("=" * 92)
    import platform

    print(f"python {platform.python_version()} on {platform.platform()}")
    print(f"config: {rl_config().to_dict()}")
    print()

    env = test_environment()
    test_headless()
    test_actions(env)
    test_rewards(env)
    env.close()
    test_vector_envs()
    test_ppo(args.timesteps, args.envs, args.device, Path(args.output))
    test_rendering()

    failed = [name for name, ok, _ in RESULTS if not ok]
    skipped = [name for name, ok, detail in RESULTS if ok and str(detail).startswith("skipped")]
    print()
    print("=" * 92)
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed ({len(skipped)} skipped)")
    if failed:
        print("FAILED:")
        for name in failed:
            print(f"  - {name}")
        return 1
    print("SMOKE TEST PASSED -- the pipeline is valid; a long training run can be started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
