"""Central configuration for the Dino Run simulation.

The original game (``main.py`` / ``game_breakdown/11_Final_Responsive.py``) derived
every constant from the desktop resolution::

    ratio_x = WIDTH / 800
    ratio_y = HEIGHT / 600
    ratio   = min(ratio_x, ratio_y)

That scheme is preserved *exactly* here, but the resolution is made explicit so that
the RL environment can pin it.  With ``width=800, height=600`` we get
``ratio_x == ratio_y == ratio == 1.0``, which is the clean "base" geometry of the
game (and matches ``main_half_screen.py``'s constants: gravity 0.8, jump 14,
fast-fall 5, base game speed 4).

No gameplay constant lives here: every timing/physics number stays inline in the
entities/core modules so it can be diffed against the original file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict

BASE_WIDTH = 800
BASE_HEIGHT = 600

# The original hard-coded these two numbers when scaling the ground strip
# (``int(40 * ratio_y)`` for the strip height and ``HEIGHT - 40 * ratio_y`` for its top).
GROUND_HEIGHT = 40.0

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class GameConfig:
    """Resolution + side-effect flags for one Dino Run game instance."""

    width: int = BASE_WIDTH
    height: int = BASE_HEIGHT
    #: Height (in *base* units) of the ground strip at the bottom of the screen.
    ground_height: float = GROUND_HEIGHT
    #: The original writes ``high_score.txt`` on death. RL turns this off so training
    #: never touches the player's save file (and never blocks on file IO).
    write_high_score: bool = True
    high_score_path: str = "high_score.txt"
    #: Asset folder, relative to the project root (the game originally required cwd==repo root).
    assets_root: str = "assets"

    # ---------------------------------------------------------------- ratios
    @property
    def ratio_x(self) -> float:
        return self.width / BASE_WIDTH

    @property
    def ratio_y(self) -> float:
        return self.height / BASE_HEIGHT

    @property
    def ratio(self) -> float:
        """The sprite scale used by the original code (``min(ratio_x, ratio_y)``)."""
        return min(self.ratio_x, self.ratio_y)

    @property
    def ground_pixels(self) -> int:
        """``int(40 * ratio_y)`` -- height of the scaled ground image."""
        return int(self.ground_height * self.ratio_y)

    @property
    def ground_top(self) -> float:
        """Y coordinate of the top of the ground strip (``HEIGHT - 40 * ratio_y``)."""
        return self.height - self.ground_height * self.ratio_y

    # ---------------------------------------------------------------- helpers
    def replace(self, **changes: Any) -> "GameConfig":
        return replace(self, **changes)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.update(
            ratio_x=self.ratio_x,
            ratio_y=self.ratio_y,
            ratio=self.ratio,
            ground_pixels=self.ground_pixels,
            ground_top=self.ground_top,
        )
        return data

    @staticmethod
    def parse_resolution(text: str) -> "GameConfig":
        """Parse ``"800x600"`` / ``"800,600"`` into a config."""
        for sep in ("x", "X", ","):
            if sep in text:
                w, h = text.lower().split(sep, 1)
                return GameConfig(width=int(w), height=int(h))
        raise ValueError(f"could not parse resolution {text!r}; expected e.g. 800x600")


#: Configuration used by the RL side: base geometry, deterministic, no side effects.
def rl_config(width: int = BASE_WIDTH, height: int = BASE_HEIGHT) -> GameConfig:
    return GameConfig(width=width, height=height, write_high_score=False)
