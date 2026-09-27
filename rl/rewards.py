"""Reward design for Dino Run.

The only real objective is: **stay alive as long as possible**.  Everything below is
built around that, with deliberately *no action shaping*.

Formula
-------
For a normal frame ``t``::

    r_t = advance * (game_speed / SPEED_BASE)          # progress through the level
        + survival                                     # optional flat per-frame term
        + clear_bonus * obstacles_passed_this_frame    # optional, off by default

and on collision::

    r_T = r_T_frame + death                            # death == a large negative

Total::

    R = scale * ( sum_t r_t + death_if_collision )

Why each term exists
--------------------
``advance`` (default ``0.1``)
    Dense progress signal proportional to *distance actually travelled* this frame,
    which is the game's own scroll speed.  A player who reaches a higher game speed
    covers more ground per frame, so the term also rewards getting further into the
    harder part of the level.  Because it is paid per frame, total return is
    monotone in survival time -- the agent's return can only grow by surviving
    longer (or by surviving into faster sections).

``survival`` (default ``0.0``)
    Optional flat per-frame bonus.  Left at zero because ``advance`` already covers
    "time alive"; it exists only so the trade-off between time and progress can be
    re-weighted without touching the code.

``death`` (default ``-10``)
    Collision ends the episode; the terminal penalty makes dying clearly worse than
    surviving one more frame (which pays ``advance``-ish, ~0.1-0.5).  ``-10`` is
    small relative to a long episode's return, but it is *not* the mechanism that
    teaches survival -- the lost future reward is.  It mainly keeps the value
    function honest at the end of an episode.

``clear_bonus`` (default ``0.0`` -- deliberately off)
    Optional reward for an obstacle that passes behind the dinosaur safely.  It is
    not exploitable (entities only appear at a fixed spawn cadence, so the bonus is
    proportional to survival time anyway), but it adds nothing the progress term
    does not already give, so the default keeps the signal minimal.

``scale_advance_by_speed`` (default True)
    Divides nothing, multiplies by ``game_speed / SPEED_BASE``: reward per frame
    equals distance per frame, so the reward is literally "pixels of progress".

Exploit analysis
----------------
* **No reward for jumping.**  Jumping while nothing is ahead earns exactly the same
  as doing nothing (both just advance).  Jumping into an obstacle still pays the
  death penalty.  So spamming jumps is *never* better than running, and it is worse
  whenever it prevents a needed jump (jumping costs ~35 frames of air time during
  which the agent cannot jump again).
* **No reward for crouching** for the same reason.
* **No reward for "being in the air"** or for any action/state that could be farmed
  in place: the only positive term is proportional to distance, and distance only
  accumulates while the episode continues.
* **No reward for passing an obstacle** by default, so the agent cannot learn a
  local "dodge" micro-behaviour that ignores survival.
* **No reward for standing still** (there is no such state -- the world scrolls at
  ``game_speed`` no matter what the agent does).
* The agent can never benefit from dying early: death both ends the progress stream
  and pays a negative terminal reward.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

from game.core import BASE_GAME_SPEED


@dataclass
class RewardConfig:
    """Weights of the reward function (see the module docstring)."""

    advance: float = 0.1
    survival: float = 0.0
    death: float = -10.0
    clear_bonus: float = 0.0
    scale_advance_by_speed: bool = True
    scale: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "RewardConfig":
        allowed = {k: v for k, v in (data or {}).items() if k in RewardConfig.__dataclass_fields__}
        return RewardConfig(**allowed)

    def describe(self) -> str:
        speed_part = "advance * (game_speed / base_speed)" if self.scale_advance_by_speed else "advance"
        pieces = [speed_part]
        if self.survival:
            pieces.append(f"survival ({self.survival})")
        if self.clear_bonus:
            pieces.append(f"clear_bonus * obstacles_cleared ({self.clear_bonus})")
        formula = " + ".join(pieces)
        return f"r_t = {self.scale} * ({formula});  death = {self.scale} * ({self.death})"


class RewardFunction:
    """Stateless reward evaluator (kept as a class so it can be extended/versioned)."""

    def __init__(self, config: RewardConfig, base_game_speed: float = BASE_GAME_SPEED) -> None:
        self.config = config
        self.base_game_speed = base_game_speed

    def base_frame_reward(self, game_speed: float) -> float:
        """Reward for one survived frame (no terminal component)."""
        cfg = self.config
        progress = self.base_game_speed
        if cfg.scale_advance_by_speed and progress:
            advance = cfg.advance * (game_speed / progress)
        else:
            advance = cfg.advance
        return cfg.scale * (advance + cfg.survival)

    def frame_reward(self, game_speed: float, obstacles_cleared: int = 0) -> float:
        reward = self.base_frame_reward(game_speed)
        if self.config.clear_bonus and obstacles_cleared:
            reward += self.config.scale * self.config.clear_bonus * obstacles_cleared
        return reward

    def terminal_reward(self, terminated: bool) -> float:
        """``terminated`` is True only for a collision (not for time-limit truncation)."""
        if not terminated:
            return 0.0
        return self.config.scale * self.config.death

    def describe(self) -> str:
        return self.config.describe()
