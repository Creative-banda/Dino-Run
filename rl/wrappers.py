"""Small Gymnasium wrappers used by training and evaluation.

The environment itself already reports everything needed (score, distance,
survival time, spawns, action counts) in ``info["dino_episode"]`` when an episode
ends; these wrappers only *aggregate* that across episodes so the evaluation script
and the training callbacks can report without bookkeeping of their own.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

import gymnasium as gym


class EpisodeStatsWrapper(gym.Wrapper):
    """Records every finished episode and exposes aggregate statistics."""

    def __init__(self, env: gym.Env, max_episodes: Optional[int] = None) -> None:
        super().__init__(env)
        self.episodes: List[Dict[str, Any]] = []
        self.max_episodes = max_episodes

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        summary = info.get("dino_episode")
        if summary is not None:
            self.episodes.append(dict(summary))
            if self.max_episodes is not None and len(self.episodes) > self.max_episodes:
                self.episodes = self.episodes[-self.max_episodes :]
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    # ---------------------------------------------------------------- reporting
    def clear(self) -> None:
        self.episodes.clear()

    @property
    def episode_count(self) -> int:
        return len(self.episodes)

    def _values(self, key: str) -> np.ndarray:
        return np.asarray([episode.get(key, 0.0) for episode in self.episodes], dtype=np.float64)

    def action_distribution(self) -> Dict[str, float]:
        counts: Dict[str, float] = {}
        total = 0.0
        for episode in self.episodes:
            for name, value in (episode.get("actions") or {}).items():
                counts[name] = counts.get(name, 0.0) + float(value)
                total += float(value)
        if total <= 0:
            return {name: 0.0 for name in counts} or {}
        return {name: value / total for name, value in counts.items()}

    def summary(self) -> Dict[str, Any]:
        """Aggregated metrics over all recorded episodes."""
        if not self.episodes:
            return {"episodes": 0}
        score = self._values("score")
        survival = self._values("survival_seconds")
        distance = self._values("distance")
        reward = self._values("reward")
        frames = self._values("frames")
        obstacles = self._values("obstacles_spawned")
        birds = self._values("birds_spawned")
        collisions = np.asarray([bool(e.get("collision", False)) for e in self.episodes])
        best_index = int(np.argmax(score))
        worst_index = int(np.argmin(score))
        return {
            "episodes": len(self.episodes),
            "score_mean": float(score.mean()),
            "score_std": float(score.std()),
            "score_max": float(score.max()),
            "score_min": float(score.min()),
            "score_median": float(np.median(score)),
            "frames_mean": float(frames.mean()),
            "survival_seconds_mean": float(survival.mean()),
            "distance_mean": float(distance.mean()),
            "distance_total": float(distance.sum()),
            "reward_mean": float(reward.mean()),
            "reward_std": float(reward.std()),
            "reward_max": float(reward.max()),
            "reward_min": float(reward.min()),
            "obstacles_mean": float(obstacles.mean()),
            "birds_mean": float(birds.mean()),
            "collisions": int(collisions.sum()),
            "collision_rate": float(collisions.mean()),
            "best_episode": dict(self.episodes[best_index]),
            "worst_episode": dict(self.episodes[worst_index]),
            "action_distribution": self.action_distribution(),
        }


class EpisodeRewardWrapper(gym.Wrapper):
    """Lightweight, SB3-``Monitor``-compatible episode return/length logging.

    SB3's ``Monitor`` writes CSV files and drags extra imports into every worker
    process.  Worker processes hold one pygame simulation each, so keeping them lean
    directly raises how many can run in parallel (RAM is the binding constraint on
    Colab).  The accounting below matches ``Monitor``: the reported episode return is
    the sum of the rewards since the last reset, *including* the terminal step's
    reward (i.e. the death penalty), and ``info["episode"]`` is only attached on the
    final step of an episode.
    """

    def __init__(self, env: gym.Env, info_keywords=()) -> None:
        super().__init__(env)
        self.needs_reset = True
        self.rewards: List[float] = []
        self.episode_returns: List[float] = []
        self.episode_lengths: List[int] = []
        self.info_keywords = tuple(info_keywords)
        self.total_steps = 0
        self._t_start = time.perf_counter()

    def reset(self, **kwargs):
        self.needs_reset = False
        self.rewards = []
        self._t_start = time.perf_counter()
        return self.env.reset(**kwargs)

    def step(self, action):
        if self.needs_reset:
            raise RuntimeError("step() called on an episode that already finished")
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.rewards.append(float(reward))
        if terminated or truncated:
            self.needs_reset = True
            episode_info = {
                "r": round(sum(self.rewards), 6),
                "l": len(self.rewards),
                "t": round(time.perf_counter() - self._t_start, 6),
            }
            for key in self.info_keywords:
                episode_info[key] = info.get(key)
            self.episode_returns.append(episode_info["r"])
            self.episode_lengths.append(episode_info["l"])
            info["episode"] = episode_info
        self.total_steps += 1
        return observation, reward, terminated, truncated, info

    def get_total_steps(self) -> int:
        return self.total_steps


class NumpyObsWrapper(gym.ObservationWrapper):
    """Guarantees a contiguous ``np.float32`` observation of the declared shape."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        low = np.asarray(self.observation_space.low, dtype=np.float32)
        high = np.asarray(self.observation_space.high, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation):
        return np.asarray(observation, dtype=np.float32)
