"""Evaluate a trained PPO agent on the real game and compare it to baselines.

Evaluation is learning-free: the model is loaded and used with ``predict``.  Every
episode uses a fixed seed, so the same evaluation can be re-run after further
training and compared directly.

Reported per policy: survival time, frames, distance, score, obstacles/birds
encountered, collisions, actions taken and their distribution, mean/best/worst
episode reward.  The trained agent is compared against:

* ``random``         -- uniform random actions (the "no learning" baseline),
* ``do_nothing``     -- never acts (dies at the first obstacle),
* ``geometric_heuristic`` -- a hand-coded geometry controller (a reference for what
  simple non-learned rules achieve, so "better than random" is not mistaken for
  "plays well").

Examples
--------
::

    python training/evaluate_ppo.py --model models/ppo_dino_v1 --episodes 20
    python training/evaluate_ppo.py --model models/ppo_dino_v1 --episodes 10 --no-baselines
    python training/evaluate_ppo.py --baselines-only --episodes 10      # baselines only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from game.config import GameConfig, rl_config  # noqa: E402
from rl.baselines import PolicySource, build_policy  # noqa: E402
from rl.rewards import RewardConfig  # noqa: E402
from rl.vector_env import make_vec_env  # noqa: E402
from rl.wrappers import EpisodeStatsWrapper  # noqa: E402
from training.artifacts import load_model, load_run_config, resolve_model_path, write_json  # noqa: E402


def evaluate_policy(
    policy: PolicySource,
    episodes: int,
    seed: int,
    config: GameConfig,
    envs: int,
    max_episode_steps: int,
    frame_skip: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run ``episodes`` fixed-seed episodes and return aggregate statistics."""
    from rl.environment import DinoRunEnv

    collector: List[Dict[str, Any]] = []
    rng = np.random.default_rng(seed + 999)
    started = time.time()

    def record(summary: Dict[str, Any]) -> None:
        collector.append(summary)
        if verbose:
            print(
                f"  episode {len(collector):>3}: score {summary.get('score', 0):>6} "
                f"({summary.get('survival_seconds', 0):.1f}s, {summary.get('distance', 0):,.0f}px, "
                f"collision={summary.get('collision')})",
                flush=True,
            )

    if envs <= 1:
        # single environment: sequential, fully deterministic, easy to follow
        env = EpisodeStatsWrapper(
            DinoRunEnv(
                config=config,
                reward_config=RewardConfig(),
                frame_skip=frame_skip,
                max_episode_steps=max_episode_steps,
                seed=seed,
            )
        )
        policy.reset()
        observation, _ = env.reset(seed=seed)
        while len(collector) < episodes:
            action = policy.act(env, observation, rng)
            observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                record(info.get("dino_episode", {}))
                observation, _ = env.reset()
        env.close()
    else:
        # parallel evaluation: one worker per evaluation environment
        vec = make_vec_env(
            envs,
            seed=seed,
            backend="process",
            config=config,
            reward_config=RewardConfig(),
            frame_skip=frame_skip,
            max_episode_steps=max_episode_steps,
            monitor=False,
            envs_per_worker=1,
        )
        policy.reset()
        observations = vec.reset()
        # heuristic policies need the game object, which lives in the worker process
        if policy.name == "geometric_heuristic":
            vec.close()
            raise SystemExit("the geometric heuristic baseline only supports --envs 1")
        while len(collector) < episodes:
            actions = np.array([policy.act(None, observations[index], rng) for index in range(vec.num_envs)])
            observations, _, dones, infos = vec.step(actions)
            for done, info in zip(dones, infos):
                if not done:
                    continue
                summary = info.get("dino_episode")
                if summary is None:
                    continue
                record(summary)
        vec.close()

    elapsed = time.time() - started
    collector = collector[:episodes]
    if not collector:
        return {"policy": policy.name, "episodes": 0, "error": "no episodes finished"}

    def values(key: str) -> np.ndarray:
        return np.asarray([float(episode.get(key, 0.0)) for episode in collector], dtype=np.float64)

    score = values("score")
    survival = values("survival_seconds")
    distance = values("distance")
    reward = values("reward")
    obstacles = values("obstacles_spawned")
    birds = values("birds_spawned")
    collisions = np.asarray([bool(episode.get("collision", False)) for episode in collector])
    action_totals: Dict[str, int] = {}
    for episode in collector:
        for name, count in (episode.get("actions") or {}).items():
            action_totals[name] = action_totals.get(name, 0) + int(count)
    total_actions = sum(action_totals.values()) or 1

    best_index = int(np.argmax(score))
    worst_index = int(np.argmin(score))
    return {
        "policy": policy.name,
        "policy_info": policy.describe(),
        "episodes": len(collector),
        "wall_seconds": elapsed,
        "steps_per_second": float(score.sum() / elapsed) if elapsed else 0.0,
        "score_mean": float(score.mean()),
        "score_std": float(score.std()),
        "score_median": float(np.median(score)),
        "score_best": float(score.max()),
        "score_worst": float(score.min()),
        "frames_mean": float(values("frames").mean()),
        "survival_seconds_mean": float(survival.mean()),
        "survival_seconds_best": float(survival.max()),
        "survival_seconds_worst": float(survival.min()),
        "distance_mean": float(distance.mean()),
        "distance_total": float(distance.sum()),
        "obstacles_mean": float(obstacles.mean()),
        "birds_mean": float(birds.mean()),
        "obstacles_total": int(obstacles.sum()),
        "collisions": int(collisions.sum()),
        "collision_rate": float(collisions.mean()),
        "truncations": int((~collisions).sum()),
        "actions_total": int(total_actions),
        "action_distribution": {name: count / total_actions for name, count in action_totals.items()},
        "action_counts": action_totals,
        "reward_mean": float(reward.mean()),
        "reward_std": float(reward.std()),
        "reward_best": float(reward.max()),
        "reward_worst": float(reward.min()),
        "best_episode": collector[best_index],
        "worst_episode": collector[worst_index],
        "episodes_detail": [
            {
                "seed": episode.get("seed"),
                "score": episode.get("score"),
                "survival_seconds": episode.get("survival_seconds"),
                "distance": episode.get("distance"),
                "reward": episode.get("reward"),
                "collision": episode.get("collision"),
                "obstacles_spawned": episode.get("obstacles_spawned"),
                "birds_spawned": episode.get("birds_spawned"),
            }
            for episode in collector
        ],
    }


def _format_table(results: List[Dict[str, Any]]) -> str:
    header = (
        f"{'policy':<22} {'eps':>4} {'score mean':>10} {'best':>7} {'worst':>7} "
        f"{'survival s':>11} {'distance':>10} {'boxes':>6} {'birds':>6} {'collide':>8} {'reward':>9}"
    )
    lines = [header, "-" * len(header)]
    for result in results:
        if result.get("error"):
            lines.append(f"{result['policy']:<22} ERROR: {result['error']}")
            continue
        lines.append(
            f"{result['policy']:<22} {result['episodes']:>4} {result['score_mean']:>10,.0f} "
            f"{result['score_best']:>7,.0f} {result['score_worst']:>7,.0f} "
            f"{result['survival_seconds_mean']:>11.1f} {result['distance_mean']:>10,.0f} "
            f"{result['obstacles_mean']:>6.1f} {result['birds_mean']:>6.1f} "
            f"{result['collision_rate'] * 100:>7.0f}% {result['reward_mean']:>9,.1f}"
        )
    return "\n".join(lines)


def _format_actions(result: Dict[str, Any]) -> str:
    distribution = result.get("action_distribution") or {}
    counts = result.get("action_counts") or {}
    parts = [f"{name}: {100 * share:.1f}% ({counts.get(name, 0):,})" for name, share in sorted(distribution.items())]
    return " | ".join(parts)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO Dino Run agent (no learning).")
    parser.add_argument("--model", default=None, help="model .zip, training output directory, or 'none'")
    parser.add_argument("--episodes", type=int, default=20, help="episodes per policy")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--envs", type=int, default=1, help="parallel evaluation environments")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of taking the argmax")
    parser.add_argument("--resolution", default=None)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=6000)
    parser.add_argument("--no-baselines", action="store_true")
    parser.add_argument("--baselines-only", action="store_true")
    parser.add_argument("--json", default=None, help="where to write the raw results")
    parser.add_argument("--verbose", action="store_true", help="print every episode")
    args = parser.parse_args(argv)

    config = GameConfig.parse_resolution(args.resolution) if args.resolution else rl_config()
    run_config = {}
    if args.model and args.model.lower() not in ("none", "null"):
        run_config = load_run_config(Path(args.model)) if Path(args.model).is_dir() else {}

    print("=" * 118)
    print("Dino Run - PPO evaluation (no learning, fixed seeds)")
    print("=" * 118)

    results: List[Dict[str, Any]] = []
    model = None
    if args.model and args.model.lower() not in ("none", "null") and not args.baselines_only:
        resolved = resolve_model_path(args.model)
        print(f"model      : {resolved}")
        model = load_model(resolved, env=None, device=args.device)
        policy = build_policy("model", model=model, deterministic=not args.stochastic)
        print(f"device     : {model.policy.device} | timesteps trained: {int(model.num_timesteps):,}")
        if run_config:
            print(
                f"run config : {run_config.get('n_envs')} envs, seed {run_config.get('seed')}, "
                f"frame_skip {run_config.get('frame_skip')}, obs {run_config.get('environment', {}).get('obs_version')}"
            )
        print(f"episodes   : {args.episodes} per policy, seeds {args.seed}..{args.seed + args.episodes - 1}")
        print()
        print(f"Evaluating PPO agent ({'deterministic' if not args.stochastic else 'stochastic'} actions)...")
        results.append(
            evaluate_policy(
                policy,
                args.episodes,
                args.seed,
                config,
                args.envs,
                args.max_episode_steps,
                args.frame_skip,
                verbose=args.verbose,
            )
        )

    if not args.no_baselines:
        for policy in [build_policy("random"), build_policy("do_nothing"), build_policy("geometric_heuristic")]:
            print(f"Evaluating baseline '{policy.name}'...")
            results.append(
                evaluate_policy(
                    policy,
                    args.episodes,
                    args.seed,
                    config,
                    args.envs,
                    args.max_episode_steps,
                    args.frame_skip,
                    verbose=args.verbose,
                )
            )

    print()
    print(_format_table(results))
    print()
    for result in results:
        if not result.get("error"):
            print(f"actions   {result['policy']:<22} {_format_actions(result)}")

    comparison: Dict[str, Any] = {}
    agent = next((r for r in results if r.get("policy") == "ppo"), None)
    if agent:
        print()
        print("Improvement over the untrained baselines (mean score / mean survival):")
        for result in results:
            if result is agent or result.get("error"):
                continue
            score_ratio = agent["score_mean"] / result["score_mean"] if result["score_mean"] else float("inf")
            survival_ratio = (
                agent["survival_seconds_mean"] / result["survival_seconds_mean"]
                if result["survival_seconds_mean"]
                else float("inf")
            )
            comparison[result["policy"]] = {
                "score_ratio": score_ratio,
                "survival_ratio": survival_ratio,
                "baseline_score_mean": result["score_mean"],
                "agent_score_mean": agent["score_mean"],
            }
            print(
                f"  vs {result['policy']:<22} score x{score_ratio:>8.1f}   "
                f"survival x{survival_ratio:>8.1f}   "
                f"({result['score_mean']:,.0f} -> {agent['score_mean']:,.0f})"
            )

    payload = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": str(args.model) if args.model else None,
        "settings": {
            "episodes": args.episodes,
            "seed": args.seed,
            "deterministic": not args.stochastic,
            "envs": args.envs,
            "resolution": f"{config.width}x{config.height}",
            "frame_skip": args.frame_skip,
            "max_episode_steps": args.max_episode_steps,
        },
        "results": results,
        "comparison": comparison,
    }
    if args.json:
        write_json(args.json, payload)
        print(f"\nresults written to {args.json}")
    elif args.model and Path(args.model).is_dir():
        default_path = Path(args.model) / f"evaluation{'random' if args.stochastic else ''}.json"
        write_json(default_path, payload)
        print(f"\nresults written to {default_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
