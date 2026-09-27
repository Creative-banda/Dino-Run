"""Dino Run -- the original Pygame game, refactored to be embeddable.

The gameplay, physics, art, animation and spawning logic are unchanged (see
``game/core.py`` and ``game/entities.py`` for the line-by-line mapping to the
original ``main.py``).  What changed is only *how the game is driven*:

* the clock (real time for humans, simulated 60 Hz time for RL),
* the random number generator (per instance, therefore seedable),
* the action source (arrow keys for humans, an integer 0/1/2 for RL),
* optional headless operation (no window, no audio) for training.

Public API::

    from game import DinoGame, GameConfig, rl_config, Assets
    from game.human_play import play, main
"""

from .actions import ACTION_DOWN, ACTION_JUMP, ACTION_NOTHING, NUM_ACTIONS, action_name
from .assets import Assets
from .clock import FRAME_MS, Clock, RealClock, SimClock
from .config import BASE_HEIGHT, BASE_WIDTH, GameConfig, rl_config
from .core import DinoGame
from .headless import init_headless_pygame, is_headless, set_headless_env_vars
from .render import Renderer

__all__ = [
    "ACTION_DOWN",
    "ACTION_JUMP",
    "ACTION_NOTHING",
    "NUM_ACTIONS",
    "action_name",
    "Assets",
    "Clock",
    "RealClock",
    "SimClock",
    "FRAME_MS",
    "BASE_HEIGHT",
    "BASE_WIDTH",
    "GameConfig",
    "rl_config",
    "DinoGame",
    "Renderer",
    "init_headless_pygame",
    "is_headless",
    "set_headless_env_vars",
]

__version__ = "1.1.0-rl"
