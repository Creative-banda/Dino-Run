"""Asset loading for the game, with a process-wide cache.

The original script loaded assets at import time and stored them in globals.  For RL
we spin up many game instances (and many processes), so:

* every scaled surface is cached by ``(path, size)`` and therefore loaded **once per
  process** and shared by every environment in it (surfaces are never drawn onto,
  so sharing is safe),
* render-only assets (parallax layers, title, fonts, ground bitmap) are skipped
  entirely in headless mode,
* audio is optional -- it is never initialised for RL.

The exact same files, scales and iteration order as the original are used, because
hitbox sizes are derived from the scaled images.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import pygame

from .config import PROJECT_ROOT, GameConfig

_IMAGE_CACHE: Dict[Tuple[str, Optional[Tuple[int, int]]], pygame.Surface] = {}
_FRAME_CACHE: Dict[Tuple[str, str, int, int], Tuple[pygame.Surface, ...]] = {}
_FONT_CACHE: Dict[Tuple[str, int], "pygame.font.Font"] = {}
_SOUND_CACHE: Dict[str, "pygame.mixer.Sound"] = {}

SOUND_FILES = {
    "jump": "sounds/jump.mp3",
    "background_music": "sounds/background_music.mp3",
    "loose": "sounds/loose.wav",
    "game_start": "sounds/game_start.mp3",
    "main_menu": "sounds/starting_background.mp3",
}


def to_display_format(surface: pygame.Surface) -> pygame.Surface:
    """Match a loaded image to the display's pixel format (``convert()``/``convert_alpha()``).

    ``pygame.image.load`` returns a surface in the *file's* pixel format.  Blitting such
    a surface makes pygame fall back to a per-pixel format conversion on every single
    blit, which is dramatically slower than blitting a display-format surface -- measured
    on the same machine, one full-window parallax layer at 2560x1440 cost **33.5 ms**
    as loaded versus **2.6 ms** after conversion (13x).  Five such layers per frame is
    the difference between the game running at its intended speed and running in slow
    motion.

    The original script called ``.convert_alpha()`` on every sprite for exactly this
    reason; this helper does the same thing, but picks the right conversion for the
    file's content (alpha channel kept for transparent sprites, plain ``convert()`` for
    fully opaque images, e.g. the ground strip).  Neither conversion changes any pixel,
    so this is purely a performance fix.

    Falls back to the untouched surface when there is no display context (unit tests
    that load assets before ``pygame.display`` exists).
    """
    try:
        has_alpha = surface.get_masks()[3] != 0 or surface.get_flags() & pygame.SRCALPHA
        return surface.convert_alpha() if has_alpha else surface.convert()
    except pygame.error:
        return surface


class AssetError(RuntimeError):
    """Raised when a required sprite/sound file is missing."""


class Assets:
    """Loads and caches every asset the game needs for a given :class:`GameConfig`."""

    def __init__(self, config: GameConfig, visual: bool = False, load_sounds: Optional[bool] = None) -> None:
        self.config = config
        self.visual = visual
        self.load_sounds = visual if load_sounds is None else load_sounds
        root = Path(config.assets_root)
        self.root = root if root.is_absolute() else PROJECT_ROOT / root

    # ------------------------------------------------------------------ paths
    def path(self, relative: str) -> Path:
        p = self.root / relative
        if not p.exists():
            raise AssetError(f"missing asset: {p}")
        return p

    def _image(self, relative: str, size: Optional[Tuple[int, int]] = None) -> pygame.Surface:
        """Load (and cache) an image, optionally scaled."""
        key = (relative, size)
        cached = _IMAGE_CACHE.get(key)
        if cached is not None:
            return cached
        surface = pygame.image.load(str(self.path(relative)))
        surface = to_display_format(surface)
        if size is not None:
            surface = pygame.transform.scale(surface, size)
        _IMAGE_CACHE[key] = surface
        return surface

    def _sprite(self, relative: str, factor: float) -> pygame.Surface:
        """Load an image and scale it by ``factor`` (matching the original expressions)."""
        unscaled = self._image(relative)
        size = (int(unscaled.get_width() * factor), int(unscaled.get_height() * factor))
        return self._image(relative, size)

    # ----------------------------------------------------------------- frames
    def _frames(self, kind: str, name: str, filenames: Sequence[str], factor: float) -> Tuple[pygame.Surface, ...]:
        cfg = self.config
        key = (kind, name, cfg.width, cfg.height)
        cached = _FRAME_CACHE.get(key)
        if cached is not None:
            return cached
        frames = tuple(self._sprite(relative, factor) for relative in filenames)
        _FRAME_CACHE[key] = frames
        return frames

    def dino_frames(self, action: str) -> Tuple[pygame.Surface, ...]:
        """Frames of one dino animation (``running`` / ``jump`` / ``duck``).

        Identical to ``Player.load_animation``: the frame count comes from
        ``os.listdir`` and each image is scaled by ``2 * ratio``.
        """
        directory = self.root / "images" / "dino" / action
        num_of_frames = len(os.listdir(directory))
        filenames = [f"images/dino/{action}/{i}.png" for i in range(num_of_frames)]
        return self._frames("dino", action, filenames, 2 * self.config.ratio)

    def bird_frames(self) -> Tuple[pygame.Surface, ...]:
        directory = self.root / "images" / "bird"
        num_of_frames = len(os.listdir(directory))
        filenames = [f"images/bird/{i}_BirdSprite.png" for i in range(num_of_frames)]
        return self._frames("bird", "bird", filenames, 2 * self.config.ratio)

    def obstacle_image(self, index: int) -> pygame.Surface:
        """One of the six box images, scaled like ``Obstacle.__init__`` (``1.5 * ratio``)."""
        return self._sprite(f"images/boxes/{index}.png", 1.5 * self.config.ratio)

    # ------------------------------------------------------ render-only assets
    def ground_surface(self) -> Optional[pygame.Surface]:
        """Scaled ground strip (visual mode only; headless derives the rect arithmetically)."""
        if not self.visual:
            return None
        cfg = self.config
        return self._sprite_plain("images/background/ground.png", (cfg.width, cfg.ground_pixels))

    def _sprite_plain(self, relative: str, size: Tuple[int, int]) -> pygame.Surface:
        """Load and scale an opaque image (the ground strip / title) for display."""
        key = (relative, size)
        cached = _IMAGE_CACHE.get(key)
        if cached is not None:
            return cached
        surface = pygame.image.load(str(self.path(relative)))
        surface = to_display_format(surface)
        surface = pygame.transform.scale(surface, size)
        _IMAGE_CACHE[key] = surface
        return surface

    def background_surfaces(self) -> Sequence[pygame.Surface]:
        """The five parallax layers, scaled to the full window (visual mode only)."""
        if not self.visual:
            return ()
        cfg = self.config
        # The original used convert_alpha() for these five layers.
        return tuple(self._image(f"images/background/bg_{i}.png", (cfg.width, cfg.height)) for i in range(5))

    def title_surface(self) -> Optional[pygame.Surface]:
        """Menu title; ``None`` when missing, exactly like the original try/except."""
        if not self.visual:
            return None
        cfg = self.config
        try:
            unscaled = self._image("images/title.png")
        except AssetError:
            return None
        size = (int(unscaled.get_width() * cfg.ratio_x), int(unscaled.get_height() * cfg.ratio_y))
        return self._image("images/title.png", size)

    def font(self, name: str, size: int) -> "pygame.font.Font":
        if not pygame.font.get_init():
            pygame.font.init()
        key = (name, size)
        cached = _FONT_CACHE.get(key)
        if cached is None:
            cached = pygame.font.Font(str(self.path(f"font/{name}.ttf")), size)
            _FONT_CACHE[key] = cached
        return cached

    # ------------------------------------------------------------------ audio
    def sound(self, name: str) -> Optional["pygame.mixer.Sound"]:
        """A sound effect, or ``None`` when audio is disabled/unavailable."""
        if not self.load_sounds:
            return None
        if not pygame.mixer.get_init():
            try:
                pygame.mixer.init()
            except pygame.error:
                self.load_sounds = False
                return None
        relative = SOUND_FILES[name]
        cached = _SOUND_CACHE.get(relative)
        if cached is None:
            try:
                cached = pygame.mixer.Sound(str(self.path(relative)))
            except (pygame.error, AssetError):
                self.load_sounds = False
                return None
            _SOUND_CACHE[relative] = cached
        return cached

    def play_sound(self, name: str) -> None:
        sound = self.sound(name)
        if sound is not None:
            sound.play()

    def stop_sound(self, name: str) -> None:
        sound = self.sound(name)
        if sound is not None:
            sound.stop()


def clear_caches() -> None:
    """Drop every cached surface (used by tests)."""
    _IMAGE_CACHE.clear()
    _FRAME_CACHE.clear()
    _FONT_CACHE.clear()


def cache_stats() -> Dict[str, int]:
    return {"images": len(_IMAGE_CACHE), "frames": len(_FRAME_CACHE)}
