"""Headless pygame bootstrap for reinforcement learning.

Google Colab (and any server) has no display, no keyboard and no sound card.  The
RL simulation must nevertheless run the *same* game code, and that code needs a
pygame video mode because sprites are scaled with ``convert_alpha()`` and the
player/bird/obstacle hitboxes are derived from the loaded image sizes.

``SDL_VIDEODRIVER=dummy`` gives us exactly that: a real pygame display context that
never opens an OS window.  Everything is then off-screen and unflipped, so physics
and collision behave normally while nothing is rendered.

What we deliberately do *not* need in headless mode (and therefore skip):

* audio (``pygame.mixer`` is never initialised),
* fonts / text rendering,
* the 5 full-screen parallax background layers and the title image.
"""

from __future__ import annotations

import os
from typing import Tuple

import pygame

_INITIALISED = False


def set_headless_env_vars(forcing: bool = True) -> None:
    """Force the SDL dummy drivers.

    ``forcing`` overrides any value the caller may have exported, because a stray
    ``SDL_VIDEODRIVER=windows`` in the environment would otherwise pop up windows in
    a training run.  ``PYGAME_HIDE_SUPPORT_PROMPT`` keeps stdout clean for logs.
    """
    if forcing:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    else:  # pragma: no cover - convenience path
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


def init_headless_pygame(video_size: Tuple[int, int] = (1, 1)) -> bool:
    """Make pygame usable without a window.

    Returns ``True`` if this call installed the headless display, ``False`` if a
    display was already initialised (e.g. a human play session) and nothing was
    touched -- so calling this from an RL environment never breaks visual playback.
    """
    global _INITIALISED

    if pygame.display.get_init() and pygame.display.get_surface() is not None:
        return False

    set_headless_env_vars(forcing=True)
    pygame.display.init()
    # 1x1 is enough: the surface exists purely so that ``convert_alpha()`` works.
    # Nothing is ever blitted to it unless the caller asks for rgb_array rendering.
    pygame.display.set_mode(video_size)
    _INITIALISED = True
    pygame.event.pump()
    return True


def is_headless() -> bool:
    """True when the SDL dummy video driver is in use."""
    try:
        return pygame.display.get_driver() == "dummy"
    except Exception:  # pragma: no cover - display not initialised
        return os.environ.get("SDL_VIDEODRIVER") == "dummy"


def describe_display() -> dict:
    """Small diagnostic used by the smoke test / benchmark reports."""
    info = {"pygame": pygame.version.ver, "sdl": ".".join(str(v) for v in pygame.get_sdl_version())}
    try:
        info["driver"] = pygame.display.get_driver()
    except Exception:
        info["driver"] = None
    info["headless"] = is_headless()
    info["display_initialised"] = bool(pygame.display.get_init())
    info["mixer_initialised"] = bool(pygame.mixer.get_init()) if hasattr(pygame, "mixer") else False
    return info
