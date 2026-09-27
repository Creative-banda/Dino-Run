"""Policy sources used to compare a trained PPO agent against reference behaviours.

Every policy is wrapped in the same tiny interface so ``training/evaluate_ppo.py``
can measure them identically (same seeds, same episode budget, same reward):

* :class:`RandomPolicy`          -- uniform random actions (the "no learning" baseline)
* :class:`DoNothingPolicy`       -- never acts at all (dies at the first box)
* :class:`GeometricHeuristicPolicy`
  -- a hand-written controller that jumps for ground boxes and crouches under
  mid-height birds, using **only** the recorded geometry.  It is not trained, not a
  language model and not imitation learning: it is a reference upper-ish bound that
  says "this game is playable with simple geometry rules", which makes it a much
  more informative comparison than random actions alone.
* :class:`ModelPolicy`           -- a Stable-Baselines3 model (the PPO agent)
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from game.actions import ACTION_DOWN, ACTION_JUMP, ACTION_NOTHING, NUM_ACTIONS
from game.core import DinoGame
from game.entities import Bird


class PolicySource:
    """Base class: turn an environment (+observation) into a discrete action."""

    name = "policy"

    def reset(self) -> None:
        """Called at every episode start."""

    def act(self, env, observation: Optional[np.ndarray], rng: np.random.Generator) -> int:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"name": self.name}


class RandomPolicy(PolicySource):
    name = "random"

    def act(self, env, observation, rng) -> int:
        return int(rng.integers(0, NUM_ACTIONS))


class DoNothingPolicy(PolicySource):
    name = "do_nothing"

    def act(self, env, observation, rng) -> int:
        return ACTION_NOTHING


class GeometricHeuristicPolicy(PolicySource):
    """Jump the boxes, crouch under the mid birds, ignore the high birds.

    Geometry only -- exactly the rules a human reads off the screen.  The dinosaur can
    jump ~122 px high but a bird at ``ground - 60..80`` px passes *above* it, so the
    heuristic only reacts to entities that would actually hit a standing player.
    """

    name = "geometric_heuristic"

    def __init__(self, jump_lead_frames: float = 11.0, crouch_lead_frames: float = 30.0) -> None:
        self.jump_lead_frames = jump_lead_frames
        self.crouch_lead_frames = crouch_lead_frames

    def act(self, env, observation, rng) -> int:
        game: DinoGame = env.unwrapped.game
        player = game.player
        px, py, pw, ph = player.rect
        ground_top = game.ground_top
        speed = game.game_speed or 1.0
        best = None
        for entity in game.entities_ahead():
            gap = entity.rect.left - (px + pw)
            if best is None or gap < best[0]:
                best = (gap, entity)
        if best is None:
            return ACTION_NOTHING
        gap, entity = best
        frames = gap / speed
        is_bird = isinstance(entity, Bird)
        if not is_bird:
            return ACTION_JUMP if -2 <= frames <= self.jump_lead_frames else ACTION_NOTHING
        bottom_above_ground = ground_top - entity.rect.bottom
        if bottom_above_ground <= 2:  # skimming the ground: jump it
            return ACTION_JUMP if -2 <= frames <= self.jump_lead_frames else ACTION_NOTHING
        if bottom_above_ground <= 34:  # mid height: crouch under it
            return ACTION_DOWN if frames <= self.crouch_lead_frames else ACTION_NOTHING
        return ACTION_NOTHING  # high bird: passes above a standing dinosaur


class ModelPolicy(PolicySource):
    """A trained Stable-Baselines3 model."""

    def __init__(self, model, deterministic: bool = True, name: str = "ppo") -> None:
        self.model = model
        self.deterministic = deterministic
        self.name = name

    def act(self, env, observation, rng) -> int:
        action, _ = self.model.predict(observation, deterministic=self.deterministic)
        return int(np.asarray(action).reshape(-1)[0])

    def describe(self) -> dict:
        info = {"name": self.name, "deterministic": self.deterministic}
        if hasattr(self.model, "num_timesteps"):
            info["num_timesteps"] = int(self.model.num_timesteps)
        return info


def build_policy(name: str, model=None, deterministic: bool = True) -> PolicySource:
    """Factory used by the CLI (``--policy random|do_nothing|geometric_heuristic|model``)."""
    name = (name or "").lower()
    if name in ("random", "rand"):
        return RandomPolicy()
    if name in ("none", "noop", "do_nothing", "nothing"):
        return DoNothingPolicy()
    if name in ("heuristic", "geometric", "geometric_heuristic"):
        return GeometricHeuristicPolicy()
    if name in ("model", "ppo", "agent"):
        if model is None:
            raise ValueError("build_policy('model') requires a loaded model")
        return ModelPolicy(model, deterministic=deterministic, name="ppo")
    raise ValueError(f"unknown policy {name!r}")


def baseline_policies() -> list:
    """Policies evaluated alongside the trained agent for comparison."""
    return [RandomPolicy(), DoNothingPolicy(), GeometricHeuristicPolicy()]
