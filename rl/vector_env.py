"""Vectorised Dino Run environments.

Many independent Dino Run instances run at once so PPO sees far more experience per
wall-clock second.  Each is fully independent: its own game state, player, obstacle
list, score, episode, RNG seed and reset (see :class:`~rl.environment.DinoRunEnv`).

Two backends:

``dummy``
    All environments stepped sequentially in one process (:class:`DummyVecEnv`).
    Zero communication overhead, but the GIL means the pure-python physics cannot
    run in parallel across cores.
``subproc``
    One OS process per environment (:class:`SubprocVecEnv`).  This is what actually
    scales the simulation across CPU cores -- the recommended backend whenever more
    than one core is available.

The number of environments is *not* guessed: ``training/benchmark_envs.py`` measures
throughput and writes a report that ``resolve_n_envs`` reads.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from game.config import GameConfig, rl_config

from .environment import DinoRunEnv
from .rewards import RewardConfig
from .wrappers import EpisodeRewardWrapper

DEFAULT_BENCHMARK_FILE = "models/benchmark_envs.json"


def make_env(
    env_id: int = 0,
    seed: Optional[int] = None,
    config: Optional[GameConfig] = None,
    reward_config: Optional[RewardConfig] = None,
    frame_skip: int = 1,
    max_episode_steps: int = 6000,
    render_mode: Optional[str] = None,
    monitor: bool = True,
) -> Callable[[], "gym.Env"]:
    """Return a thunk that builds one environment (picklable: safe for subprocesses)."""

    def _init() -> "gym.Env":
        env = DinoRunEnv(
            config=config,
            reward_config=reward_config,
            frame_skip=frame_skip,
            max_episode_steps=max_episode_steps,
            render_mode=render_mode,
            seed=seed,
            env_id=env_id,
        )
        if monitor:
            # Per-episode return/length statistics that ``verbose=1`` training logs
            # print ("rollout/ep_rew_mean", ...).  Deliberately *not* SB3's Monitor:
            # this wrapper is dependency-free, so the worker processes do not import
            # torch/stable-baselines3 and stay ~3x smaller in RAM.
            env = EpisodeRewardWrapper(env, info_keywords=("score",))
        return env

    return _init


def make_vec_env(
    n_envs: int,
    seed: int = 0,
    backend: str = "auto",
    config: Optional[GameConfig] = None,
    reward_config: Optional[RewardConfig] = None,
    frame_skip: int = 1,
    max_episode_steps: int = 6000,
    start_method: Optional[str] = None,
    monitor: bool = True,
    envs_per_worker: Optional[int] = None,
):
    """Build ``n_envs`` independent environments.

    Seeds follow the usual convention ``env i -> seed + i`` so no two environments
    ever replay the same obstacle sequence.

    Backends: ``dummy`` (one process), ``process`` (this project's batched-IPC
    multi-process backend -- the default for ``auto`` when more than one core is
    usable), ``subproc`` (SB3's ``SubprocVecEnv``, kept for comparison).
    """
    if n_envs < 1:
        raise ValueError("n_envs must be >= 1")

    # imported lazily so that worker processes never import torch / SB3
    from stable_baselines3.common.vec_env import DummyVecEnv

    config = config or rl_config()
    reward_config = reward_config or RewardConfig()
    env_fns = [
        make_env(
            env_id=index,
            seed=seed + index,
            config=config,
            reward_config=reward_config,
            frame_skip=frame_skip,
            max_episode_steps=max_episode_steps,
            monitor=monitor,
        )
        for index in range(n_envs)
    ]

    if backend == "dummy" or n_envs == 1:
        vec = DummyVecEnv(env_fns)
        vec.seed(seed)  # type: ignore[attr-defined]
        return vec

    if backend == "auto":
        backend = "process"

    if backend == "process":
        from .process_vec_env import ProcessGroupVecEnv, default_grouping

        grouping = default_grouping(n_envs, envs_per_worker)
        try:
            vec = ProcessGroupVecEnv(
                env_fns,
                envs_per_worker=grouping,
                start_method=start_method or "spawn",
            )
        except Exception as exc:  # pragma: no cover - platform dependent
            print(f"[vector_env] process backend unavailable ({exc}); using DummyVecEnv", file=sys.stderr)
            vec = DummyVecEnv(env_fns)
        vec.seed(seed)  # type: ignore[attr-defined]
        return vec

    if backend == "subproc":
        from stable_baselines3.common.vec_env import SubprocVecEnv

        kwargs: Dict[str, Any] = {}
        if start_method:
            kwargs["start_method"] = start_method
        try:
            vec = SubprocVecEnv(env_fns, **kwargs)
        except Exception as exc:  # pragma: no cover - platform dependent
            print(f"[vector_env] SubprocVecEnv unavailable ({exc}); falling back to DummyVecEnv", file=sys.stderr)
            vec = DummyVecEnv(env_fns)
        vec.seed(seed)  # type: ignore[attr-defined]
        return vec

    raise ValueError(f"unknown backend {backend!r} (use 'auto', 'dummy', 'process' or 'subproc')")


def seed_vec_env(vec, seed: int) -> None:
    """Seed every sub-environment deterministically (``env i -> seed + i``)."""
    try:
        vec.seed(seed)
    except Exception:  # pragma: no cover
        for index, env in enumerate(getattr(vec, "envs", [])):
            try:
                env.reset(seed=seed + index)
            except Exception:
                pass


def recommended_env_count(
    benchmark_file: Union[str, Path] = DEFAULT_BENCHMARK_FILE,
    fallback: int = 8,
    max_envs: Optional[int] = None,
) -> int:
    """Read the benchmark report and return the recommended number of environments."""
    path = Path(benchmark_file)
    if not path.exists():
        return max(1, min(fallback, os.cpu_count() or 1))
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return fallback
    recommended = report.get("recommendation") or {}
    count = int(recommended.get("n_envs") or fallback)
    if max_envs is not None:
        count = min(count, max_envs)
    return max(1, count)


def describe_backends() -> Dict[str, Any]:
    """Which backends are usable on this machine (used by the benchmark report)."""
    cpu = os.cpu_count() or 1
    return {
        "cpu_count": cpu,
        "platform": sys.platform,
        "default_start_method": "spawn" if sys.platform == "win32" else "fork",
        "subproc_available": True,
        "suggested_max_envs": cpu,
    }


def env_configs_from_args(args) -> Dict[str, Any]:
    """Build the shared environment kwargs from parsed CLI arguments."""
    config = GameConfig.parse_resolution(args.resolution) if getattr(args, "resolution", None) else rl_config()
    return {
        "config": config,
        "reward_config": RewardConfig(),
        "frame_skip": getattr(args, "frame_skip", 1),
        "max_episode_steps": getattr(args, "max_episode_steps", 6000),
    }


def parse_env_list(text: Union[str, Sequence[int]]) -> List[int]:
    """Parse ``"1,4,8,16"`` (or a sequence) into a list of environment counts."""
    if isinstance(text, (list, tuple)):
        return [int(x) for x in text]
    counts: List[int] = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:  # range syntax, e.g. "1-16" would be odd; support "4-16:4"? keep simple
            start, _, end = chunk.partition("-")
            step = 1
            if ":" in end:
                end, _, step_text = end.partition(":")
                step = int(step_text)
            counts.extend(range(int(start), int(end) + 1, step))
        else:
            counts.append(int(chunk))
    return sorted(set(counts))
