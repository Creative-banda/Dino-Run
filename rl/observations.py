"""The structured observation consumed by PPO (no pixels).

Design goals
------------
* **Fixed shape.** Always :data:`OBS_SIZE` floats, whatever the game state, so a
  plain MLP policy can consume it.  Empty obstacle slots are zero-padded.
* **Relative and normalized.** Everything is expressed relative to the player and
  divided by window size / jump height / base game speed, so the same numbers mean
  the same thing at any resolution -- a model trained at 800x600 also plays the
  desktop-sized window of ``main.py``.
* **Geometry only, no answers.** No feature encodes "you should jump/crouch", and
  the correct action is never fed to the network; the policy has to discover the
  state -> action relationship itself.
* **Future context.** The four nearest obstacles *ahead* (including ones already
  overlapping the player) are always in the observation, so the agent can act
  early instead of only reacting when a box is on top of it.

Layout (version ``dino-obs-v1``, 47 features)
---------------------------------------------
==============================  ====  ==========================================
index                           size  meaning
==============================  ====  ==========================================
``player_*``                      7   dinosaur state (position, velocity, flags)
``game_*``                        4   speed, score, progress, entity count
``slot0..slot3``                4\*9  nearest 4 obstacles/birds, nearest first
==============================  ====  ==========================================

Per obstacle slot:
``present``, ``is_bird``, ``box_variant``, ``dx`` (gap to the player's front edge),
``top_above_ground``, ``bottom_above_ground``, ``width``, ``height``,
``contact_frames`` (frames until the entity reaches the player at its own speed).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

try:  # gymnasium is only needed for the observation space object
    from gymnasium import spaces
except ImportError:  # pragma: no cover - game-only usage
    spaces = None  # type: ignore[assignment]

from game.config import GameConfig
from game.core import BASE_GAME_SPEED, DinoGame
from game.entities import Bird

OBS_VERSION = "dino-obs-v1"
NUM_OBSTACLE_SLOTS = 4

PLAYER_FEATURES: Tuple[str, ...] = (
    "player_above_ground",  # (ground_top - player.bottom) / max_jump_height
    "player_velocity_y",  # velocity_y / jump_speed  (negative = rising)
    "player_in_air",
    "player_crouching",
    "player_fast_falling",  # "down" is currently engaged while airborne
    "player_height",  # rect height / window height (crouching shrinks the hitbox)
    "player_width",  # rect width / window width
)
GAME_FEATURES: Tuple[str, ...] = (
    "game_speed",  # game_speed / (3 * base speed)
    "game_score",  # frames survived / 6000
    "game_speed_progress",  # 0..1 within the current 10 s difficulty interval
    "game_entities_ahead",  # min(#entities ahead / 6, 1)
)
SLOT_FEATURES: Tuple[str, ...] = (
    "present",
    "is_bird",
    "box_variant",  # 1..6 -> 0..1 (0 for birds)
    "dx",  # (entity.left - player.right) / window width
    "top_above_ground",  # (ground_top - entity.top) / window height
    "bottom_above_ground",  # (ground_top - entity.bottom) / window height
    "width",  # entity.rect.w / window width
    "height",  # entity.rect.h / window height
    "contact_frames",  # frames until horizontal contact, / 240
)

OBS_SIZE = len(PLAYER_FEATURES) + len(GAME_FEATURES) + NUM_OBSTACLE_SLOTS * len(SLOT_FEATURES)

#: Axis-aligned bounds (every feature is clipped into these), used for the Box space.
PLAYER_BOUNDS = (
    (-0.5, 1.5),
    (-2.0, 3.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
)
GAME_BOUNDS = (
    (0.0, 1.5),
    (0.0, 1.5),
    (0.0, 1.0),
    (0.0, 1.0),
)
SLOT_BOUNDS = (
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (-1.0, 2.0),
    (0.0, 1.5),
    (-0.5, 1.5),
    (0.0, 0.5),
    (0.0, 0.5),
    (-0.5, 2.0),
)

#: normalization constants (kept here so they are documented and reproducible)
SCORE_NORM = 6000.0  # 100 simulated seconds at 60 fps
CONTACT_FRAMES_NORM = 240.0  # 4 simulated seconds
ENTITIES_AHEAD_NORM = 6.0
SPEED_NORM_FACTOR = 3.0  # game speed / (3 * base speed)


def feature_names() -> List[str]:
    """Human-readable name of every observation element (used in reports/docs)."""
    names = list(PLAYER_FEATURES) + list(GAME_FEATURES)
    for slot in range(NUM_OBSTACLE_SLOTS):
        names.extend(f"slot{slot}_{name}" for name in SLOT_FEATURES)
    return names


def bounds_arrays() -> Tuple[np.ndarray, np.ndarray]:
    lows = [lo for lo, _ in PLAYER_BOUNDS] + [lo for lo, _ in GAME_BOUNDS]
    highs = [hi for _, hi in PLAYER_BOUNDS] + [hi for _, hi in GAME_BOUNDS]
    for _ in range(NUM_OBSTACLE_SLOTS):
        lows.extend(lo for lo, _ in SLOT_BOUNDS)
        highs.extend(hi for _, hi in SLOT_BOUNDS)
    return np.asarray(lows, dtype=np.float32), np.asarray(highs, dtype=np.float32)


def observation_space(config: GameConfig):
    """The gymnasium ``Box`` describing this observation."""
    if spaces is None:  # pragma: no cover
        raise RuntimeError("gymnasium is required for observation_space()")
    low, high = bounds_arrays()
    del config  # bounds are resolution independent by construction
    return spaces.Box(low=low, high=high, shape=(OBS_SIZE,), dtype=np.float32)


class ObservationBuilder:
    """Turns a :class:`~game.core.DinoGame` state into an ``np.float32`` vector."""

    def __init__(self, config: GameConfig) -> None:
        self.config = config
        self.ratio_x = config.ratio_x
        self.ratio_y = config.ratio_y
        self.width = float(config.width)
        self.height = float(config.height)
        self.ground_top = float(config.ground_top)
        # 14 * ratio_y jump speed against 0.8 * ratio_y gravity -> 122.5 * ratio_y apex
        self.jump_speed = 14.0 * self.ratio_y
        self.max_jump_height = (self.jump_speed ** 2) / (2 * 0.8 * self.ratio_y)
        self.base_speed = BASE_GAME_SPEED * self.ratio_x
        self.bird_extra_speed = 5.0 * self.ratio_x
        self._low, self._high = bounds_arrays()
        self._bounds = (self._low, self._high)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def is_bird(entity) -> bool:
        return isinstance(entity, Bird)

    def entity_speed(self, game: DinoGame, entity) -> float:
        """Horizontal speed of an entity in pixels per frame (birds are faster)."""
        return game.game_speed + (self.bird_extra_speed if self.is_bird(entity) else 0.0)

    def _slots(self, game: DinoGame) -> Sequence:
        return game.entities_ahead()[:NUM_OBSTACLE_SLOTS]

    # -------------------------------------------------------------------- build
    def build(self, game: DinoGame) -> np.ndarray:
        player = game.player
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        ground_top = float(game.ground_top)

        obs[0] = (ground_top - player.rect.bottom) / self.max_jump_height
        obs[1] = player.velocity_y / self.jump_speed
        obs[2] = 1.0 if player.Inair else 0.0
        obs[3] = 1.0 if player.ducking else 0.0
        obs[4] = 1.0 if player.gravity > 0.8 * self.ratio_y + 1e-9 else 0.0
        obs[5] = player.rect.h / self.height
        obs[6] = player.rect.w / self.width

        offset = len(PLAYER_FEATURES)
        obs[offset + 0] = game.game_speed / (SPEED_NORM_FACTOR * self.base_speed)
        obs[offset + 1] = game.score / SCORE_NORM
        elapsed = game.clock.ticks() - game.last_increment_time
        obs[offset + 2] = min(max(elapsed, 0) / 10000.0, 1.0)
        ahead = game.entities_ahead()
        obs[offset + 3] = min(len(ahead) / ENTITIES_AHEAD_NORM, 1.0)

        offset += len(GAME_FEATURES)
        for index, entity in enumerate(self._slots(game)):
            base = offset + index * len(SLOT_FEATURES)
            rect = entity.rect
            is_bird = self.is_bird(entity)
            dx_px = rect.left - player.rect.right
            speed = self.entity_speed(game, entity)
            obs[base + 0] = 1.0
            obs[base + 1] = 1.0 if is_bird else 0.0
            obs[base + 2] = 0.0 if is_bird else (getattr(entity, "variant", 0) or 0) / 6.0
            obs[base + 3] = dx_px / self.width
            obs[base + 4] = (ground_top - rect.top) / self.height
            obs[base + 5] = (ground_top - rect.bottom) / self.height
            obs[base + 6] = rect.w / self.width
            obs[base + 7] = rect.h / self.height
            obs[base + 8] = (max(dx_px, 0.0) / speed / CONTACT_FRAMES_NORM) if speed > 0 else 0.0

        np.clip(obs, self._low, self._high, out=obs)
        return obs

    def describe(self) -> dict:
        return {
            "version": OBS_VERSION,
            "size": OBS_SIZE,
            "obstacle_slots": NUM_OBSTACLE_SLOTS,
            "features": feature_names(),
            "player_features": list(PLAYER_FEATURES),
            "game_features": list(GAME_FEATURES),
            "slot_features": list(SLOT_FEATURES),
            "normalizers": {
                "score": SCORE_NORM,
                "contact_frames": CONTACT_FRAMES_NORM,
                "entities_ahead": ENTITIES_AHEAD_NORM,
                "speed_factor": SPEED_NORM_FACTOR,
                "max_jump_height_px": self.max_jump_height,
                "jump_speed_px": self.jump_speed,
                "base_game_speed_px": self.base_speed,
                "ground_top_px": self.ground_top,
            },
        }
