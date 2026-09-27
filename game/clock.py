"""Time sources for the game.

The original code called ``pygame.time.get_ticks()`` for three things:

1. the animation cooldowns (``Player.update`` / ``Bird.update``),
2. the 10-second difficulty increase (``increment_game_speed``),
3. the initial timestamps stored at object construction.

Wall-clock time makes the simulation non-reproducible: a faster-than-real-time RL
rollout would never trigger a speed-up, and a lagging renderer would trigger them
sooner.  So the clock is injected:

* :class:`RealClock` -- the original behaviour, used by the human-playable game.
* :class:`SimClock`  -- one frame is exactly ``1000/60`` ms, so the game advances
  at 60 simulated frames per second regardless of how fast the RL loop runs.
  A speed-up then happens every 600 frames, which is what the original produced at
  its 60 fps cap.
"""

from __future__ import annotations

import pygame

#: 60 frames per second, exactly as the original ``clock.tick(60 * ratio_y)`` intends.
FRAME_MS = 1000.0 / 60.0


class Clock:
    """Minimal clock interface used by the simulation."""

    def ticks(self) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def begin_frame(self) -> None:
        """Called once at the top of every simulation step."""

    def reset(self) -> None:
        """Called on every episode reset."""


class RealClock(Clock):
    """Real wall-clock time (``pygame.time.get_ticks()``), used for human play."""

    def ticks(self) -> int:
        return pygame.time.get_ticks()


class SimClock(Clock):
    """Deterministic frame-based time: 1 step == 1000/60 ms of simulated time."""

    def __init__(self, start_ms: int = 0) -> None:
        self._start_ms = int(start_ms)
        self.frames = 0
        self._ms = self._start_ms

    def ticks(self) -> int:
        return self._ms

    def begin_frame(self) -> None:
        # Frame 1 runs at ``start_ms`` (exactly like the original, where the objects
        # are created and then immediately stepped); frame k runs at
        # ``round((k-1) * 1000/60)``.  This mirrors a 60 fps wall clock, so difficulty
        # speed-ups fire on the same frame as a real-time run of the original game.
        self._ms = self._start_ms + int(round(self.frames * FRAME_MS))
        self.frames += 1

    def reset(self) -> None:
        self.frames = 0
        self._ms = self._start_ms
