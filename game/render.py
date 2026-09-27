"""Drawing, ported 1:1 from the original script.

Nothing visual changed: the same five parallax layers scroll at
``game_speed * (0.25 * i)``, the same ground strip is drawn, the same score /
high-score text is rendered with ``Retro.ttf`` at ``int(25 * ratio_x)``, the same
title image and "Press SPACE to Start" text make up the menu, and layers are drawn in
the original order (background -> ground -> scores -> obstacles -> birds -> player).

The renderer is only constructed for the human-playable game and for the optional
``rgb_array`` RL rendering mode; headless training never touches it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import pygame

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .core import DinoGame


class Renderer:
    def __init__(self, game: "DinoGame", screen: pygame.Surface) -> None:
        self.game = game
        self.screen = screen
        cfg = game.config
        # bg = parallax layers, exactly like the original module-level list
        self.bg: List[dict] = [
            {"x": 0, "img": img, "scroll": i * 0.25}
            for i, img in enumerate(game.assets.background_surfaces())
        ]
        self.score_font = game.assets.font("Retro", int(25 * cfg.ratio_x))
        self.big_font = game.assets.font("Retro", int(30 * cfg.ratio_x))
        self.title_image = game.assets.title_surface()
        self.small_font: Optional[pygame.font.Font] = None

    # ------------------------------------------------------------ background
    def update_parallax_background(self, steps: int = 1) -> None:
        """Scroll and draw the five parallax layers.

        ``steps`` advances the layers by that many simulation frames and still draws
        them exactly once.  It is 1 for the normal game (bit-identical to the original
        arithmetic); agent playback passes more when it had to simulate several frames
        to keep up with real time, so the background never lags behind the world.
        """
        screen = self.screen
        width, height = self.screen.get_size()
        for layer in self.bg:
            for _ in range(max(1, steps)):
                layer["x"] -= self.game.game_speed * layer["scroll"]
                if layer["x"] <= -width:
                    layer["x"] = 0
            screen.blit(layer["img"], (layer["x"], 0))
            screen.blit(layer["img"], (layer["x"] + width, 0))

    def draw_ground(self) -> None:
        for ground in self.game.ground_group:
            ground.draw(self.screen)

    def draw_scores(self) -> None:
        game = self.game
        score_text = self.score_font.render(f"Score: {game.score}", True, (255, 255, 255))
        high_score_text = self.score_font.render(f"High Score: {game.high_score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(high_score_text, (self.screen.get_width() - high_score_text.get_width() - 10, 10))

    def draw_entities(self) -> None:
        self.game.obstacle_group.draw(self.screen)
        self.game.enemy_group.draw(self.screen)

    def draw_player(self) -> None:
        self.game.player.draw(self.screen)

    # --------------------------------------------------------------- overlays
    def draw_agent_overlay(self, lines: List[str]) -> None:
        """Small extra HUD used *only* by ``play_agent.py`` (never by the normal game)."""
        if not lines:
            return
        if self.small_font is None:
            self.small_font = self.game.assets.font("Retro", max(10, int(14 * self.game.config.ratio_x)))
        y = 45
        for line in lines:
            text = self.small_font.render(line, True, (255, 255, 255))
            self.screen.blit(text, (10, y))
            y += int(18 * self.game.config.ratio_y)

    def draw_main_menu(self) -> None:
        """One menu frame: parallax background, title image, start text."""
        self.update_parallax_background()
        width = self.screen.get_width()
        height = self.screen.get_height()
        if self.title_image is not None:
            title_rect = self.title_image.get_rect(center=(width // 2, height // 3))
            self.screen.blit(self.title_image, title_rect)
        start_text = self.big_font.render("Press SPACE to Start", True, (255, 255, 255))
        start_rect = start_text.get_rect(center=(width // 2, height - 100))
        self.screen.blit(start_text, start_rect)

    # ----------------------------------------------------------- full frames
    def draw_world(self) -> None:
        """The gameplay layer stack, in the original drawing order."""
        self.draw_ground()
        self.draw_scores()
        self.draw_entities()
        self.draw_player()

    def draw_game_over(self) -> None:
        """Same palette as the game over state (nothing new is drawn today)."""
        self.draw_world()
