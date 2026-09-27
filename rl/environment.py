"""``DinoRunEnv`` -- a Gymnasium environment around the existing Dino Run game.

The environment does not re-implement anything: every ``step`` calls
``game.core.DinoGame.step(action)``, which is the original main-loop body (proven
frame-identical to ``reference/original_main.py`` by
``tests/test_physics_equivalence.py``).

* **Headless.**  Under ``SDL_VIDEODRIVER=dummy`` pygame has an off-screen display
  context (needed because hitboxes come from scaled sprites) but no window is ever
  opened, no frame is flipped, no keyboard is read and no audio is initialised.
* **Deterministic.**  A per-environment :class:`~game.clock.SimClock` makes one step
  exactly one 60 Hz frame of simulated time, and a per-environment
  :class:`random.Random` makes obstacle/bird spawning reproducible.  ``reset(seed=s)``
  fixes the whole episode; successive auto-resets derive new seeds from ``s`` so a
  vectorised run replays identically.
* **One action = one game frame** (configurable via ``frame_skip``), so the controls
  the agent has are exactly the controls a human has.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Optional, Tuple

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from game.actions import ACTION_NAMES, NUM_ACTIONS
from game.assets import Assets
from game.clock import SimClock
from game.config import GameConfig, rl_config
from game.core import BASE_GAME_SPEED, DinoGame
from game.headless import init_headless_pygame

from .observations import OBS_VERSION, ObservationBuilder, observation_space
from .rewards import RewardConfig, RewardFunction

DEFAULT_MAX_EPISODE_STEPS = 6000  # 100 simulated seconds at 60 fps


class DinoRunEnv(gym.Env):
    """Single Dino Run instance as a Gymnasium environment."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(
        self,
        config: Optional[GameConfig] = None,
        reward_config: Optional[RewardConfig] = None,
        frame_skip: int = 1,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        headless: bool = True,
        env_id: int = 0,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"unsupported render_mode {render_mode!r} (use None or 'rgb_array')")
        if frame_skip < 1:
            raise ValueError("frame_skip must be >= 1")
        # ``max_episode_steps`` counts agent steps, i.e. up to
        # ``max_episode_steps * frame_skip`` simulated game frames per episode.

        self.config = config or rl_config()
        self.reward_config = reward_config or RewardConfig()
        self.frame_skip = int(frame_skip)
        self.max_episode_steps = int(max_episode_steps)
        self.render_mode = render_mode
        self.env_id = int(env_id)

        # Headless pygame: dummy SDL drivers, 1x1 hidden video mode, no mixer.
        if headless and render_mode != "rgb_array":
            init_headless_pygame()
        elif render_mode == "rgb_array":
            init_headless_pygame()

        visual = render_mode == "rgb_array"
        self.assets = Assets(self.config, visual=visual, load_sounds=False)
        self.game = DinoGame(
            config=self.config,
            clock=SimClock(),
            rng=random.Random(),
            assets=self.assets,
            visual=visual,
        )

        self.builder = ObservationBuilder(self.config)
        self.reward_function = RewardFunction(self.reward_config, base_game_speed=BASE_GAME_SPEED * self.config.ratio_x)

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = observation_space(self.config)
        self.action_meanings = {index: name for index, name in ACTION_NAMES.items()}
        self.obs_version = OBS_VERSION

        # seeding
        self._base_seed: Optional[int] = seed
        self._episode_index = 0
        self._rng = np.random.default_rng(seed)

        # rendering
        self._screen = None
        self._renderer = None

        # per-episode trackers
        self._steps = 0
        self._episode_reward = 0.0
        self._ahead_uids = set()
        self._action_counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
        self._episode_stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------ seeding
    def _next_game_seed(self) -> Optional[int]:
        """A reproducible per-episode game seed derived from the env's base seed."""
        index = self._episode_index
        self._episode_index += 1
        if self._base_seed is None:
            # no explicit seed: draw entropy once so auto-resets still differ
            return int(self._rng.integers(0, 2 ** 31 - 1))
        return int((int(self._base_seed) * 1_000_003 + index) % (2 ** 31 - 1))

    # -------------------------------------------------------------------- reset
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._base_seed = int(seed)
            self._episode_index = 0
            self._rng = np.random.default_rng(seed)

        game_seed = self._next_game_seed()
        self.game.reset(seed=game_seed)

        self._steps = 0
        self._episode_reward = 0.0
        self._ahead_uids = self.game.active_uids()
        self._action_counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
        self._episode_stats = {}

        observation = self.builder.build(self.game)
        info = {
            "seed": game_seed,
            "env_id": self.env_id,
            "obs_version": OBS_VERSION,
            "frame_skip": self.frame_skip,
        }
        del options
        return observation, info

    # --------------------------------------------------------------------- step
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if not self.game.is_alive:
            raise RuntimeError("step() called on a finished episode; call reset() first")

        action = int(np.asarray(action).reshape(-1)[0])
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r}; expected one of {self.action_meanings}")

        reward = 0.0
        cleared = 0
        alive = True
        for _ in range(self.frame_skip):
            alive = self.game.step(action)
            active = self.game.active_uids()
            cleared = len(self._ahead_uids - active)
            self._ahead_uids = active
            reward += self.reward_function.frame_reward(self.game.game_speed, cleared)
            if not alive:
                break

        self._steps += 1
        self._action_counts[action] += 1

        terminated = not alive
        truncated = (not terminated) and self._steps >= self.max_episode_steps
        reward += self.reward_function.terminal_reward(terminated)
        self._episode_reward += reward

        observation = self.builder.build(self.game)
        info: Dict[str, Any] = {
            "score": self.game.score,
            "frames": self.game.score,
            "survival_seconds": self.game.score / 60.0,
            "distance": self.game.distance,
            "game_speed": self.game.game_speed,
            "obstacles_spawned": self.game.spawned_obstacles,
            "birds_spawned": self.game.spawned_birds,
            "obstacles_cleared": int(cleared),
            "action": action,
            "crouching": bool(self.game.player.ducking),
            "in_air": bool(self.game.player.Inair),
        }
        if terminated or truncated:
            info["dino_episode"] = self.episode_summary(terminated=terminated)
        return observation, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ summary
    def episode_summary(self, terminated: bool = False) -> Dict[str, Any]:
        """Statistics for the episode that just ended (used for logs and evaluation)."""
        self._episode_stats = {
            "env_id": self.env_id,
            "seed": self.game.episode_seed,
            "frames": int(self.game.score),
            "score": int(self.game.score),
            "survival_seconds": self.game.score / 60.0,
            "distance": float(self.game.distance),
            "reward": float(self._episode_reward),
            "steps": int(self._steps),
            "collision": bool(terminated),
            "game_speed_final": float(self.game.game_speed),
            "obstacles_spawned": int(self.game.spawned_obstacles),
            "birds_spawned": int(self.game.spawned_birds),
            "actions": {self.action_meanings[i]: int(self._action_counts[i]) for i in range(NUM_ACTIONS)},
        }
        return dict(self._episode_stats)

    @property
    def last_episode(self) -> Dict[str, Any]:
        return dict(self._episode_stats)

    def action_distribution(self) -> Dict[str, float]:
        total = float(self._action_counts.sum()) or 1.0
        return {self.action_meanings[i]: float(self._action_counts[i]) / total for i in range(NUM_ACTIONS)}

    # ------------------------------------------------------------------ render
    def render(self):  # noqa: D401 - gymnasium API
        """Off-screen RGB frame (never opens a window)."""
        if self.render_mode != "rgb_array":
            return None
        import pygame

        from game.render import Renderer

        if self._screen is None:
            # Draws into a plain off-screen surface; the display mode is never touched
            # here, so this can never open or resize a window.
            self._screen = pygame.Surface((self.config.width, self.config.height))
            self._renderer = Renderer(self.game, self._screen)
        assert self._renderer is not None
        self._renderer.update_parallax_background()
        self._renderer.draw_world()
        return np.transpose(np.array(pygame.surfarray.array3d(self._screen)), (1, 0, 2)).copy()

    # ------------------------------------------------------------------- misc
    def close(self) -> None:
        self.game.close()

    def describe(self) -> Dict[str, Any]:
        """Full, JSON-serialisable description of the environment (for checkpoints)."""
        return {
            "env_version": "dino-rl-env-v1",
            "obs_version": OBS_VERSION,
            "observation_size": int(self.observation_space.shape[0]),
            "observations": self.builder.describe(),
            "action_space": {"type": "Discrete", "n": NUM_ACTIONS, "meanings": self.action_meanings},
            "frame_skip": self.frame_skip,
            "max_episode_steps": self.max_episode_steps,
            "game_config": self.config.to_dict(),
            "reward": self.reward_config.to_dict(),
            "reward_formula": self.reward_function.describe(),
        }

    def action_name(self, action: int) -> str:
        return ACTION_NAMES[int(action)]
